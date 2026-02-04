"""
BM25 keyword search index over clinical notes.

Features:
  - Medical-aware tokenization (preserves drug names, ICD codes)
  - doc_id → index position mapping for canonical ID tracing
  - rebuild() method for hot-reload after new notes are ingested
"""

import logging
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from backend.config import BM25_TOP_K, DB_PATH

logger = logging.getLogger(__name__)


class BM25Index:
    """In-memory BM25 index built from clinical notes."""

    def __init__(self):
        self.documents: List[Dict] = []  # [{doc_id, note_id, content, metadata}]
        self.doc_id_to_idx: Dict[str, int] = {}
        self.bm25: Optional[BM25Okapi] = None
        self._built = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Medical-aware tokenization.
        Preserves drug names, ICD codes (e.g., I10, E11.9), dosages (e.g., 500mg).
        """
        text = text.lower()
        # Keep alphanumeric + dots (for ICD codes like E11.9) + hyphens (for drug names)
        tokens = re.findall(r"[a-z0-9][a-z0-9.\-]*", text)
        return tokens

    def build(self):
        """Build the BM25 index from all clinical notes in SQLite."""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        rows = cur.execute("""
            SELECT
                n.note_id,
                n.patient_id,
                n.visit_date,
                n.doctor_name,
                n.diagnosis_code,
                n.condition_name,
                n.note_text,
                p.full_name AS patient_name,
                c.name AS clinic_name
            FROM clinical_notes n
            JOIN patients p ON n.patient_id = p.patient_id
            JOIN clinics c ON p.clinic_id = c.clinic_id
        """).fetchall()
        conn.close()

        self.documents = []
        self.doc_id_to_idx = {}
        tokenized_corpus = []

        for row in rows:
            text = (row["note_text"] or "").strip()
            if not text:
                continue

            doc_id = f"note:{row['note_id']}"
            idx = len(self.documents)

            self.documents.append({
                "doc_id": doc_id,
                "note_id": row["note_id"],
                "content": text,
                "metadata": {
                    "doc_id": doc_id,
                    "note_id": row["note_id"],
                    "patient_id": row["patient_id"],
                    "patient_name": row["patient_name"],
                    "doctor_name": row["doctor_name"],
                    "condition_name": row["condition_name"],
                    "diagnosis_code": row["diagnosis_code"],
                    "visit_date": row["visit_date"],
                    "clinic_name": row["clinic_name"],
                },
            })
            self.doc_id_to_idx[doc_id] = idx
            tokenized_corpus.append(self._tokenize(text))

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._built = True
            logger.info(f"BM25 index built with {len(self.documents)} documents")
        else:
            logger.warning("BM25 index: no documents to index")

    def rebuild(self):
        """Rebuild the index from scratch (call after new notes are added)."""
        logger.info("Rebuilding BM25 index...")
        self.build()

    def search(
        self,
        query: str,
        top_k: int = BM25_TOP_K,
        where_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search the BM25 index.

        Returns list of {doc_id, content, metadata, score} dicts,
        sorted by BM25 score descending.
        """
        if not self._built or self.bm25 is None:
            logger.warning("BM25 index not built. Returning empty results.")
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Pair indices with scores, filter out zero-score
        scored = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:top_k * 2]:  # over-fetch for post-filtering
            doc = self.documents[idx]

            # Apply metadata filter if provided
            if where_filter:
                skip = False
                for key, val in where_filter.items():
                    if doc["metadata"].get(key) != val:
                        skip = True
                        break
                if skip:
                    continue

            results.append({
                "doc_id": doc["doc_id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(score),
            })

            if len(results) >= top_k:
                break

        return results


# Module-level singleton
_bm25_index = None


def get_bm25_index() -> BM25Index:
    """Get or create the singleton BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
        _bm25_index.build()
    return _bm25_index

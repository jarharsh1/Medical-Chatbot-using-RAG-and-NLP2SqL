"""
LLM-based reranker (NOT a true cross-encoder — call it "LLM reranker" in interviews).

Features:
  - Structured JSON output with strict parsing
  - Score normalization to 0.0–1.0 with clipping
  - Rerank cache in SQLite: (query_hash, doc_id) → score
  - Batches 5 docs per LLM call for efficiency
  - Graceful fallback to input order on malformed output
  - Configurable enable/disable via config.RERANK_ENABLED
"""

import hashlib
import json
import logging
import sqlite3
from typing import Dict, List, Optional

from backend.config import (
    DB_PATH,
    get_active_model,
    RERANK_BATCH_SIZE,
    RERANK_ENABLED,
    RERANK_TOP_K,
)

logger = logging.getLogger(__name__)

RERANK_PROMPT = """You are a medical document relevance scorer.

Given a QUERY and a list of DOCUMENTS, rate how relevant each document is to answering the query.

QUERY: {query}

DOCUMENTS:
{documents}

Return a JSON array with one object per document, in the same order:
[
  {{"doc_id": "note:1042", "relevance_score": 0.85}},
  ...
]

Rules:
- relevance_score must be between 0.0 (not relevant) and 1.0 (perfectly relevant)
- Return ONLY the JSON array, no other text
- Keep the exact same doc_ids as provided
"""


def _query_hash(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def _init_rerank_cache():
    """Create the rerank_cache table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rerank_cache (
            query_hash TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (query_hash, doc_id)
        )
    """)
    conn.commit()
    conn.close()


def _get_cached_scores(qhash: str, doc_ids: List[str]) -> Dict[str, float]:
    """Look up cached rerank scores for a query+doc combination."""
    if not doc_ids:
        return {}
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join(["?"] * len(doc_ids))
    cur = conn.execute(
        f"SELECT doc_id, score FROM rerank_cache WHERE query_hash = ? AND doc_id IN ({placeholders})",
        [qhash] + doc_ids,
    )
    result = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return result


def _store_cached_scores(qhash: str, scores: Dict[str, float]):
    """Store rerank scores in cache."""
    conn = sqlite3.connect(DB_PATH)
    for doc_id, score in scores.items():
        conn.execute(
            "INSERT OR REPLACE INTO rerank_cache (query_hash, doc_id, score) VALUES (?, ?, ?)",
            (qhash, doc_id, score),
        )
    conn.commit()
    conn.close()


def _parse_scores(raw_output: str, expected_doc_ids: List[str]) -> Optional[Dict[str, float]]:
    """
    Parse LLM output into {doc_id: score} dict.
    Returns None if parsing fails.
    """
    try:
        # Strip markdown code fences if present
        text = raw_output.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        scores_list = json.loads(text)
        if not isinstance(scores_list, list):
            return None

        result = {}
        for item in scores_list:
            doc_id = item.get("doc_id", "")
            score = item.get("relevance_score", 0.0)
            # Clip to 0-1
            score = max(0.0, min(1.0, float(score)))
            if doc_id:
                result[doc_id] = score

        return result if result else None

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        logger.warning(f"Failed to parse rerank output: {e}")
        return None


def rerank(
    query: str,
    documents: List[Dict],
    top_k: int = RERANK_TOP_K,
) -> List[Dict]:
    """
    Re-rank documents using the LLM.

    Args:
        query: The user's query
        documents: List of {doc_id, content, metadata, score} dicts from RRF
        top_k: Number of documents to return after re-ranking

    Returns:
        Re-sorted list of documents with updated scores, trimmed to top_k.
        Falls back to input order if re-ranking fails.
    """
    if not RERANK_ENABLED:
        return documents[:top_k]

    if not documents:
        return []

    _init_rerank_cache()
    qhash = _query_hash(query)

    # Check cache for all docs
    all_doc_ids = [d["doc_id"] for d in documents]
    cached = _get_cached_scores(qhash, all_doc_ids)

    uncached_docs = [d for d in documents if d["doc_id"] not in cached]
    all_scores = dict(cached)

    cache_hit = len(cached) > 0 and len(uncached_docs) == 0

    if uncached_docs:
        # Score uncached docs in batches
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import HumanMessage

            llm = ChatOllama(model=get_active_model(), temperature=0)

            for i in range(0, len(uncached_docs), RERANK_BATCH_SIZE):
                batch = uncached_docs[i : i + RERANK_BATCH_SIZE]

                docs_text = "\n\n".join(
                    f"[{d['doc_id']}]: {d['content'][:300]}"
                    for d in batch
                )

                prompt = RERANK_PROMPT.format(query=query, documents=docs_text)
                response = llm.invoke([HumanMessage(content=prompt)])
                raw = response.content or ""

                expected_ids = [d["doc_id"] for d in batch]
                parsed = _parse_scores(raw, expected_ids)

                if parsed:
                    all_scores.update(parsed)
                    _store_cached_scores(qhash, parsed)
                else:
                    # Fallback: assign decreasing scores based on input order
                    logger.warning("Reranker output parsing failed. Using fallback scores.")
                    for j, d in enumerate(batch):
                        fallback_score = 1.0 - (j * 0.05)
                        all_scores[d["doc_id"]] = max(0.0, fallback_score)

        except Exception as e:
            logger.error(f"Reranker LLM call failed: {e}. Returning input order.")
            return documents[:top_k]

    # Re-sort by rerank score
    for doc in documents:
        doc["rerank_score"] = all_scores.get(doc["doc_id"], 0.0)

    documents.sort(key=lambda d: d.get("rerank_score", 0.0), reverse=True)

    result = documents[:top_k]
    logger.info(
        f"Reranked {len(documents)} → {len(result)} docs "
        f"(cache_hit={cache_hit}, cached={len(cached)}, scored={len(uncached_docs)})"
    )
    return result

"""
Long-Term Memory: successful query pattern cache.

Stores question → SQL mappings with question embeddings for few-shot retrieval.
On SQL generation, retrieves similar past patterns as examples.
"""

import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config import DB_PATH, MAX_FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_patterns (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    question_embedding BLOB,
    sql_query TEXT NOT NULL,
    query_type TEXT DEFAULT 'sql',
    success INTEGER DEFAULT 1,
    created_at REAL,
    usage_count INTEGER DEFAULT 1
)
"""


class QueryCache:
    """SQLite-backed cache for successful query patterns."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute(_TABLE_SQL)
        conn.commit()
        conn.close()

    def store_pattern(
        self,
        question: str,
        sql_query: str,
        query_type: str = "sql",
        question_embedding: Optional[List[float]] = None,
    ):
        """Store a successful question → SQL pattern."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Check for duplicate question
            existing = conn.execute(
                "SELECT pattern_id FROM query_patterns WHERE question = ?",
                (question,),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE query_patterns SET usage_count = usage_count + 1 WHERE pattern_id = ?",
                    (existing[0],),
                )
            else:
                embedding_blob = None
                if question_embedding:
                    embedding_blob = np.array(question_embedding, dtype=np.float32).tobytes()

                conn.execute(
                    """INSERT INTO query_patterns
                       (question, question_embedding, sql_query, query_type, success, created_at, usage_count)
                       VALUES (?, ?, ?, ?, 1, ?, 1)""",
                    (question, embedding_blob, sql_query, query_type, time.time()),
                )

            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store query pattern: {e}")
        finally:
            conn.close()

    def get_similar_patterns(
        self,
        question_embedding: Optional[List[float]] = None,
        top_k: int = MAX_FEW_SHOT_EXAMPLES,
    ) -> List[str]:
        """
        Retrieve the most similar past SQL patterns as few-shot examples.

        Uses cosine similarity on question embeddings.
        Falls back to most-used patterns if no embedding provided.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if question_embedding is not None:
                return self._similarity_search(conn, question_embedding, top_k)
            else:
                return self._fallback_search(conn, top_k)
        except Exception as e:
            logger.error(f"Failed to retrieve patterns: {e}")
            return []
        finally:
            conn.close()

    def _similarity_search(
        self, conn: sqlite3.Connection, query_emb: List[float], top_k: int
    ) -> List[str]:
        """Find patterns by cosine similarity."""
        query_vec = np.array(query_emb, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return self._fallback_search(conn, top_k)

        rows = conn.execute(
            "SELECT question, sql_query, question_embedding FROM query_patterns WHERE question_embedding IS NOT NULL"
        ).fetchall()

        if not rows:
            return self._fallback_search(conn, top_k)

        scored = []
        for question, sql_query, emb_blob in rows:
            stored_vec = np.frombuffer(emb_blob, dtype=np.float32)
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm == 0:
                continue
            similarity = float(np.dot(query_vec, stored_vec) / (query_norm * stored_norm))
            scored.append((similarity, question, sql_query))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            f"Q: {q}\nSQL: {sql}"
            for _, q, sql in scored[:top_k]
        ]

    def _fallback_search(self, conn: sqlite3.Connection, top_k: int) -> List[str]:
        """Fall back to most-used patterns."""
        rows = conn.execute(
            "SELECT question, sql_query FROM query_patterns ORDER BY usage_count DESC LIMIT ?",
            (top_k,),
        ).fetchall()
        return [f"Q: {q}\nSQL: {sql}" for q, sql in rows]


# Module-level singleton
_cache: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache

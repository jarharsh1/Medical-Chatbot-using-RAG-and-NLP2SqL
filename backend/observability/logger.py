"""
Observability: per-request run logging to SQLite.

Logs all retrieval artifacts, generation results, quality signals, and timing
so that debugging accuracy issues takes minutes, not hours.

Query the `runs` table to trace any past request end-to-end.
"""

import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

from backend.config import DB_PATH

logger = logging.getLogger(__name__)

_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp REAL,
    session_id TEXT,
    question TEXT,
    query_type TEXT,
    route_confidence REAL,
    retrieved_doc_ids TEXT,
    bm25_scores TEXT,
    semantic_scores TEXT,
    rerank_scores TEXT,
    final_doc_ids TEXT,
    sql_generated TEXT,
    sql_result TEXT,
    rag_answer TEXT,
    final_answer TEXT,
    confidence REAL,
    grounding_score REAL,
    is_grounded INTEGER,
    unsupported_claims TEXT,
    retrieval_time_ms INTEGER,
    rerank_time_ms INTEGER,
    generation_time_ms INTEGER,
    grounding_time_ms INTEGER,
    total_time_ms INTEGER,
    error TEXT
)
"""

_initialized = False


def _ensure_table():
    global _initialized
    if _initialized:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(_RUNS_TABLE_SQL)
        conn.commit()
        conn.close()
        _initialized = True
    except Exception as e:
        logger.error(f"Failed to create runs table: {e}")


def log_run(
    run_id: str,
    session_id: Optional[str],
    question: str,
    query_type: str,
    result: Dict[str, Any],
) -> None:
    """
    Log a complete query run to the runs table.

    Args:
        run_id: Unique run identifier.
        session_id: Conversation session ID.
        question: Original user question.
        query_type: "sql", "rag", or "hybrid".
        result: The full response dict from _process_query.
    """
    _ensure_table()

    metadata = result.get("metadata", {})
    grounding = result.get("grounding") or {}
    sources = result.get("sources", [])

    # Extract doc_ids from sources
    retrieved_doc_ids = [s.get("doc_id", "") for s in sources]

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """INSERT OR REPLACE INTO runs (
                run_id, timestamp, session_id, question, query_type,
                route_confidence, retrieved_doc_ids,
                sql_generated, sql_result, rag_answer, final_answer,
                confidence, grounding_score, is_grounded, unsupported_claims,
                retrieval_time_ms, generation_time_ms, grounding_time_ms,
                total_time_ms, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                time.time(),
                session_id,
                question,
                query_type,
                result.get("confidence"),
                json.dumps(retrieved_doc_ids),
                result.get("sql_generated"),
                None,  # sql_result is embedded in answer for now
                None,  # rag_answer extracted in hybrid mode
                result.get("answer"),
                result.get("confidence"),
                grounding.get("score"),
                1 if grounding.get("is_grounded") else 0,
                json.dumps(grounding.get("unsupported_claims", [])),
                metadata.get("retrieval_time_ms", 0),
                metadata.get("generation_time_ms", 0),
                metadata.get("grounding_time_ms", 0),
                metadata.get("total_time_ms", 0),
                result.get("error"),
            ),
        )
        conn.commit()
        conn.close()
        logger.debug(f"Logged run {run_id}")

    except Exception as e:
        logger.error(f"Failed to log run {run_id}: {e}")


def get_recent_runs(limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve recent runs for debugging."""
    _ensure_table()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific run by ID."""
    _ensure_table()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    conn.close()

    return dict(row) if row else None

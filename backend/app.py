"""
Medical AI Backend — FastAPI server.

Routes:
  /api/query     → AI query (routed: SQL, RAG, or Hybrid)
  /api/dashboard → Patient table with KPIs + pagination
  /api/filters   → Distinct clinics, doctors, conditions for dropdowns
"""

import csv
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import (
    DATA_DIR,
    DB_PATH,
    PROJECT_ROOT,
    SERVER_HOST,
    SERVER_PORT,
)

logger = logging.getLogger(__name__)

# ---------------------------
# APP SETUP
# ---------------------------
app = FastAPI(title="Medical AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# DATABASE INITIALIZATION
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clinics (
            clinic_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            dob TEXT,
            gender TEXT,
            insurance_provider TEXT,
            clinic_id INTEGER,
            FOREIGN KEY(clinic_id) REFERENCES clinics(clinic_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clinical_notes (
            note_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            visit_date TEXT,
            doctor_name TEXT,
            diagnosis_code TEXT,
            condition_name TEXT,
            note_text TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            rx_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            medication_name TEXT,
            dosage TEXT,
            days_supply INTEGER,
            refills_remaining INTEGER,
            last_filled_date TEXT,
            status TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM clinics")
    count = cursor.fetchone()[0]

    if count == 0 and DATA_DIR:
        _load_csv_to_table(conn, "clinics.csv", "clinics")
        _load_csv_to_table(conn, "patients.csv", "patients")
        _load_csv_to_table(conn, "clinical_notes.csv", "clinical_notes")
        _load_csv_to_table(conn, "prescriptions.csv", "prescriptions")

    conn.close()


def _load_csv_to_table(conn: sqlite3.Connection, filename: str, table_name: str):
    if not DATA_DIR:
        return
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        return

    cursor = conn.cursor()
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if not headers:
            return
        rows = list(reader)
        if not rows:
            return
        placeholders = ",".join(["?"] * len(headers))
        sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
        cursor.executemany(sql, rows)
        conn.commit()


# Initialize DB on import (CSV bootstrap if empty)
init_db()


# ---------------------------
# RAG INITIALIZATION (lazy, on first query)
# ---------------------------
_rag_initialized = False


def _ensure_rag_ready():
    """Initialize vector store and BM25 index once."""
    global _rag_initialized
    if _rag_initialized:
        return

    try:
        from backend.rag.vectorstore import populate_vectorstore
        from backend.rag.bm25 import get_bm25_index

        logger.info("Initializing RAG pipeline (vectorstore + BM25)...")
        populate_vectorstore()
        get_bm25_index()
        _rag_initialized = True
        logger.info("RAG pipeline ready.")
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}")
        _rag_initialized = True  # don't retry on every request


# ---------------------------
# QUERY PROCESSING
# ---------------------------
def _process_query(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Route and execute a user question through the appropriate agent.
    Applies guardrails (grounding, confidence, attribution) to RAG/hybrid answers.

    For multi-part questions, uses the orchestrator to decompose and handle each part.

    Returns the unified response dict.
    """
    from backend.agents.router import classify_query
    from backend.agents.sql_agent import generate_and_execute
    from backend.agents.rag_agent import retrieve_and_generate
    from backend.agents.hybrid_agent import run_hybrid
    from backend.agents.orchestrator import orchestrate_query
    from backend.guardrails.grounding import check_grounding
    from backend.guardrails.confidence import compute_confidence, should_refuse, REFUSAL
    from backend.guardrails.attribution import format_sources
    from backend.memory.conversation import get_conversation_memory
    from backend.memory.query_cache import get_query_cache
    from backend.memory.query_result_cache import get_cache

    run_id = str(uuid.uuid4())

    # Check result cache first (for identical queries)
    result_cache = get_cache()
    cached_result = result_cache.get(question)
    if cached_result is not None:
        logger.info(f"[{run_id}] Cache hit for query: {question[:50]}...")
        cached_result["from_cache"] = True
        cached_result["metadata"] = cached_result.get("metadata", {})
        cached_result["metadata"]["run_id"] = run_id
        cached_result["metadata"]["cache_hit"] = True
        return cached_result

    # Ensure RAG index is built for rag/hybrid queries
    _ensure_rag_ready()

    # Load conversation context (short-term memory)
    conversation_context = ""
    if session_id:
        memory = get_conversation_memory()
        conversation_context = memory.get_context(session_id)

    # Load few-shot examples (long-term memory)
    cache = get_query_cache()
    few_shot_examples = cache.get_similar_patterns()

    # Try orchestration for multi-part questions first
    orchestrated = orchestrate_query(
        question=question,
        session_id=session_id,
        conversation_context=conversation_context,
        few_shot_examples=few_shot_examples,
    )
    if orchestrated is not None:
        logger.info(f"[{run_id}] Used orchestrator for multi-part question")
        # Save to conversation memory
        if session_id:
            get_conversation_memory().add_turn(
                session_id, question, orchestrated["answer"],
                query_type="orchestrated",
                sql_query=orchestrated.get("sql_generated"),
            )
        # Cache orchestrated result
        result_cache.set(question, orchestrated)
        return orchestrated

    # Simple question: use direct routing
    query_type = classify_query(question)
    logger.info(f"[{run_id}] Query type: {query_type} — '{question[:80]}'")

    # Execute based on route
    if query_type == "sql":
        result = generate_and_execute(
            question=question,
            conversation_context=conversation_context,
            few_shot_examples=few_shot_examples,
        )
        answer = result.get("query_result") or ""
        if not answer or answer == "[]":
            answer = "No matching records found. Try simplifying your query."

        conf = compute_confidence(query_type="sql", llm_self_confidence=1.0 if not result.get("error") else 0.0)

        # Save to memory
        if session_id:
            get_conversation_memory().add_turn(
                session_id, question, answer, query_type="sql",
                sql_query=result.get("sql_query"),
            )
        # Cache successful SQL pattern
        if not result.get("error") and result.get("sql_query"):
            cache.store_pattern(question, result["sql_query"], query_type="sql")

        sql_response = {
            "query_type": "sql",
            "answer": answer,
            "result": answer,  # backward compat
            "sql_generated": result.get("sql_query", ""),
            "confidence": conf["score"],
            "sources": [],
            "grounding": None,
            "clarification": None,
            "error": result.get("error"),
            "metadata": {
                "run_id": run_id,
                "iterations": result.get("iterations", 0),
                "generation_time_ms": result.get("generation_time_ms", 0),
                "confidence_detail": conf,
            },
        }
        # Cache successful SQL result
        if not result.get("error"):
            result_cache.set(question, sql_response)
        return sql_response

    elif query_type == "rag":
        result = retrieve_and_generate(
            question=question,
            conversation_context=conversation_context,
        )
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        retrieved_docs = result.get("retrieved_docs", [])

        # Grounding check
        grounding = check_grounding(answer, sources)

        # Confidence scoring
        conf = compute_confidence(
            query_type="rag",
            retrieved_docs=retrieved_docs,
            grounding_result=grounding,
            llm_self_confidence=result.get("llm_self_confidence", 0.5),
        )

        # Refusal policy
        if should_refuse("rag", conf["score"], grounding):
            answer = REFUSAL
            if sources:
                snippets = "\n".join(f"- {s.get('text_snippet', '')[:100]}" for s in sources[:3])
                answer += f"\n\nClosest matches found:\n{snippets}"

        # Add disclaimer if medium confidence
        elif conf.get("disclaimer"):
            answer += f"\n\n_{conf['disclaimer']}_"

        formatted_sources = format_sources(sources)

        # Save to memory
        if session_id:
            source_ids = [s.get("doc_id", "") for s in sources if s.get("cited")]
            get_conversation_memory().add_turn(
                session_id, question, answer, query_type="rag",
                source_doc_ids=source_ids,
            )

        rag_response = {
            "query_type": "rag",
            "answer": answer,
            "result": answer,  # backward compat
            "sql_generated": None,
            "confidence": conf["score"],
            "sources": formatted_sources,
            "grounding": {
                "is_grounded": grounding.get("is_grounded"),
                "score": grounding.get("grounding_score"),
                "supported_sentences": grounding.get("supported_sentences"),
                "total_sentences": grounding.get("total_sentences"),
                "unsupported_claims": grounding.get("unsupported_claims", []),
            },
            "clarification": None,
            "error": None,
            "metadata": {
                "run_id": run_id,
                "retrieval_time_ms": result.get("retrieval_time_ms", 0),
                "generation_time_ms": result.get("generation_time_ms", 0),
                "grounding_time_ms": grounding.get("grounding_time_ms", 0),
                "confidence_detail": conf,
            },
        }
        # Cache RAG result (longer TTL since documents don't change often)
        if conf["score"] > 0.3:  # Only cache confident results
            result_cache.set(question, rag_response, ttl_override=7200)  # 2 hours
        return rag_response

    else:  # hybrid
        result = run_hybrid(
            question=question,
            conversation_context=conversation_context,
            few_shot_examples=few_shot_examples,
        )
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        retrieved_docs = result.get("retrieved_docs", [])

        # Grounding check (on the RAG portion)
        rag_answer = result.get("rag_answer", "")
        grounding = check_grounding(rag_answer, sources) if rag_answer else None

        # Confidence scoring
        conf = compute_confidence(
            query_type="hybrid",
            retrieved_docs=retrieved_docs,
            grounding_result=grounding,
            llm_self_confidence=result.get("confidence", 0.5),
        )

        if conf.get("disclaimer"):
            answer += f"\n\n_{conf['disclaimer']}_"

        formatted_sources = format_sources(sources)

        # Save to memory
        if session_id:
            source_ids = [s.get("doc_id", "") for s in sources if s.get("cited")]
            get_conversation_memory().add_turn(
                session_id, question, answer, query_type="hybrid",
                sql_query=result.get("sql_generated"),
                source_doc_ids=source_ids,
            )
        # Cache successful SQL pattern from hybrid
        sql_query = result.get("sql_generated")
        if sql_query and not result.get("error"):
            cache.store_pattern(question, sql_query, query_type="hybrid")

        grounding_response = None
        if grounding:
            grounding_response = {
                "is_grounded": grounding.get("is_grounded"),
                "score": grounding.get("grounding_score"),
                "supported_sentences": grounding.get("supported_sentences"),
                "total_sentences": grounding.get("total_sentences"),
                "unsupported_claims": grounding.get("unsupported_claims", []),
            }

        hybrid_response = {
            "query_type": "hybrid",
            "answer": answer,
            "result": answer,  # backward compat
            "sql_generated": result.get("sql_generated", ""),
            "confidence": conf["score"],
            "sources": formatted_sources,
            "grounding": grounding_response,
            "clarification": None,
            "hybrid_mode": result.get("hybrid_mode"),
            "error": result.get("error"),
            "metadata": {
                "run_id": run_id,
                "retrieval_time_ms": result.get("retrieval_time_ms", 0),
                "generation_time_ms": result.get("generation_time_ms", 0),
                "total_time_ms": result.get("total_time_ms", 0),
                "grounding_time_ms": grounding.get("grounding_time_ms", 0) if grounding else 0,
                "confidence_detail": conf,
            },
        }
        # Cache hybrid result
        if not result.get("error"):
            result_cache.set(question, hybrid_response)
        return hybrid_response


# ---------------------------
# API ROUTES
# ---------------------------
class FilterRequest(BaseModel):
    clinic: Optional[str] = None
    doctor: Optional[str] = None
    condition: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


def _norm(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip()
    return v if v != "" else None


def _build_where_and_params(f: FilterRequest) -> Tuple[str, List[Any]]:
    where = " WHERE 1=1 "
    params: List[Any] = []
    if f.clinic:
        where += " AND c.name = ?"
        params.append(f.clinic)
    if f.doctor:
        where += " AND n.doctor_name = ?"
        params.append(f.doctor)
    if f.condition:
        where += " AND n.condition_name = ?"
        params.append(f.condition)
    return where, params


@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "Medical AI Backend Running"}


@app.get("/api/cache/stats")
def cache_stats():
    """Get cache statistics for monitoring."""
    from backend.memory.query_result_cache import get_cache
    cache = get_cache()
    return cache.get_info()


@app.post("/api/cache/clear")
def cache_clear():
    """Clear all cached query results."""
    from backend.memory.query_result_cache import get_cache
    cache = get_cache()
    cache.clear()
    return {"status": "ok", "message": "Cache cleared"}


@app.post("/api/db/refresh-views")
def refresh_materialized_views():
    """Refresh materialized views after data changes."""
    try:
        from backend.scripts.optimize_database import refresh_materialized_views as refresh_mv
        refresh_mv()
        return {"status": "ok", "message": "Materialized views refreshed"}
    except Exception as e:
        raise HTTPException(500, f"Failed to refresh views: {e}")


@app.get("/api/filters")
def filters():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        clinics = [r[0] for r in cur.execute("SELECT DISTINCT name FROM clinics ORDER BY name").fetchall()]
        doctors = [r[0] for r in cur.execute("SELECT DISTINCT doctor_name FROM clinical_notes ORDER BY doctor_name").fetchall()]
        conditions = [r[0] for r in cur.execute("SELECT DISTINCT condition_name FROM clinical_notes ORDER BY condition_name").fetchall()]
        data = {"clinics": clinics, "doctors": doctors, "conditions": conditions}
    except Exception:
        data = {"clinics": [], "doctors": [], "conditions": []}
    conn.close()
    return data


@app.get("/api/dashboard")
def dashboard_get(
    clinic: Optional[str] = Query(None),
    doctor: Optional[str] = Query(None),
    condition: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    f = FilterRequest(clinic=_norm(clinic), doctor=_norm(doctor), condition=_norm(condition))
    return _dashboard_impl(f, page=page, page_size=page_size)


@app.post("/api/dashboard")
def dashboard_post(
    f: FilterRequest,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    f.clinic = _norm(f.clinic)
    f.doctor = _norm(f.doctor)
    f.condition = _norm(f.condition)
    return _dashboard_impl(f, page=page, page_size=page_size)


def _dashboard_impl(f: FilterRequest, page: int, page_size: int) -> Dict[str, Any]:
    """
    Returns:
      - kpis: computed on FULL filtered dataset (fast aggregates)
      - rows: only page_size rows
      - pagination: totals for UI
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    base_from = """
        FROM patients p
        JOIN clinics c ON p.clinic_id = c.clinic_id
        JOIN clinical_notes n ON p.patient_id = n.patient_id
        JOIN prescriptions r ON p.patient_id = r.patient_id
    """

    where_sql, params = _build_where_and_params(f)

    # KPI queries (FULL dataset)
    total_sql = f"SELECT COUNT(*) as cnt {base_from} {where_sql}"
    total_rows = int(cur.execute(total_sql, params).fetchone()["cnt"])

    uniq_pat_sql = f"SELECT COUNT(DISTINCT p.patient_id) as cnt {base_from} {where_sql}"
    unique_patients = int(cur.execute(uniq_pat_sql, params).fetchone()["cnt"])

    rx_status_sql = f"""
        SELECT r.status as status, COUNT(*) as cnt
        {base_from}
        {where_sql}
        GROUP BY r.status
    """
    rx_status_rows = cur.execute(rx_status_sql, params).fetchall()
    rx_status_map = {row["status"]: int(row["cnt"]) for row in rx_status_rows}
    active_rx = rx_status_map.get("Active", 0)
    expired_rx = rx_status_map.get("Expired", 0)

    # Paged table query
    offset = (page - 1) * page_size

    page_sql = f"""
        SELECT p.patient_id, p.full_name, c.name as clinic_name,
               n.doctor_name, n.condition_name, n.note_text, n.visit_date,
               r.medication_name, r.dosage, r.days_supply, r.refills_remaining,
               r.last_filled_date, r.status as rx_status
        {base_from}
        {where_sql}
        ORDER BY r.last_filled_date DESC
        LIMIT ? OFFSET ?
    """
    page_params = params + [page_size, offset]
    rows = cur.execute(page_sql, page_params).fetchall()
    conn.close()

    now = datetime.now()
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        try:
            last_filled = datetime.strptime(r["last_filled_date"], "%Y-%m-%d")
            days_supply = int(r["days_supply"] or 0)
            days_elapsed = (now - last_filled).days
            ratio = (days_elapsed / days_supply) if days_supply > 0 else 0.0

            status, action, action_type = "Good", "Monitor", "info"
            if ratio > 1.2:
                status, action, action_type = "Non-Adherent", "Call Patient", "danger"
            elif ratio > 0.9:
                if int(r["refills_remaining"] or 0) > 0:
                    status, action, action_type = "Refill Due", "Process Refill", "success"
                else:
                    status, action, action_type = "Renewal Needed", "Book Appointment", "warning"

            note_text = (r["note_text"] or "")
            note_snippet = note_text[:220] + ("..." if len(note_text) > 220 else "")

            out_rows.append({
                "patient_id": r["patient_id"],
                "name": r["full_name"],
                "clinic": r["clinic_name"],
                "doctor": r["doctor_name"],
                "condition": r["condition_name"],
                "medication": r["medication_name"],
                "dosage": r["dosage"],
                "note_snippet": note_snippet,
                "last_visit": r["visit_date"],
                "status": status,
                "action": action,
                "refills_left": r["refills_remaining"],
                "rx_status": r["rx_status"],
                "last_filled_date": r["last_filled_date"],
                "days_supply": r["days_supply"],
                "action_type": action_type,
            })
        except Exception:
            continue

    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    return {
        "kpis": {
            "total_rows": total_rows,
            "unique_patients": unique_patients,
            "active_rx": active_rx,
            "expired_rx": expired_rx,
        },
        "rows": out_rows,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_rows": total_rows,
            "total_pages": total_pages,
        },
        "applied_filters": {
            "clinic": f.clinic,
            "doctor": f.doctor,
            "condition": f.condition,
        },
    }


@app.post("/api/query")
def query_ai(req: QueryRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "Question required")

    try:
        response = _process_query(
            question=question,
            session_id=req.session_id,
        )

        if response.get("error") and not response.get("answer"):
            raise Exception(response["error"])

        # Log run for observability
        try:
            from backend.observability.logger import log_run
            run_id = response.get("metadata", {}).get("run_id", "")
            log_run(
                run_id=run_id,
                session_id=req.session_id,
                question=question,
                query_type=response.get("query_type", "unknown"),
                result=response,
            )
        except Exception as log_err:
            logger.warning(f"Failed to log run: {log_err}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(500, detail=str(e))


# ---------------------------
# FRONTEND (serve static files from /frontend)
# ---------------------------
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"\nServer running at: http://localhost:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)

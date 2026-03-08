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
from datetime import datetime, timedelta
from time import time as _time
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
            doctor_notes TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            rx_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            medication_name TEXT,
            dosage TEXT,
            form TEXT,
            drug_class TEXT,
            days_supply INTEGER,
            refills_remaining INTEGER,
            last_filled_date TEXT,
            status TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            message_count INTEGER DEFAULT 0
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
        
        # Check existing table columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_cols = [col[1] for col in cursor.fetchall()]
        
        # If table has different columns, add missing ones
        for col in headers:
            if col not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} TEXT")
                    logger.info(f"Added column {col} to {table_name}")
                except:
                    pass
        
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
_rag_init_attempts = 0
_MAX_RAG_INIT_ATTEMPTS = 3


def _ensure_rag_ready():
    """Initialize vector store and BM25 index once (retries up to 3 times on failure)."""
    global _rag_initialized, _rag_init_attempts
    if _rag_initialized:
        return
    if _rag_init_attempts >= _MAX_RAG_INIT_ATTEMPTS:
        return

    _rag_init_attempts += 1
    try:
        from backend.rag.vectorstore import populate_vectorstore
        from backend.rag.bm25 import get_bm25_index

        logger.info("Initializing RAG pipeline (vectorstore + BM25)...")
        populate_vectorstore()
        get_bm25_index()
        _rag_initialized = True
        logger.info("RAG pipeline ready.")
    except Exception as e:
        logger.error(
            f"RAG initialization failed (attempt {_rag_init_attempts}/{_MAX_RAG_INIT_ATTEMPTS}): {e}"
        )


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
    from backend.reliability.ollama_health import (
        check_ollama_health,
        OllamaStatus,
        get_user_friendly_error,
    )
    from backend.security.input_guard import check_input, ThreatLevel

    run_id = str(uuid.uuid4())

    # Security check - validate input before processing
    security_result = check_input(question, client_id=session_id)
    if security_result.should_block:
        logger.warning(f"[{run_id}] Blocked query: {security_result.threats_detected}")
        return {
            "query_type": "blocked",
            "answer": security_result.warning_message or "Your query was blocked for security reasons.",
            "result": security_result.warning_message,
            "sql_generated": None,
            "confidence": 0.0,
            "sources": [],
            "grounding": None,
            "clarification": None,
            "error": "security_blocked",
            "metadata": {
                "run_id": run_id,
                "security": security_result.to_dict(),
            },
        }

    # Log security warnings (but allow query to proceed)
    if security_result.threat_level == ThreatLevel.MEDIUM:
        logger.info(f"[{run_id}] Security warning: {security_result.threats_detected}")

    # Check Ollama health before processing (uses cached result, so fast)
    ollama_health = check_ollama_health()
    if ollama_health.status == OllamaStatus.UNAVAILABLE:
        error_msg = get_user_friendly_error(ollama_health.status, ollama_health.error_message)
        logger.error(f"[{run_id}] Ollama unavailable: {error_msg}")
        return {
            "query_type": "error",
            "answer": error_msg,
            "result": error_msg,
            "sql_generated": None,
            "confidence": 0.0,
            "sources": [],
            "grounding": None,
            "clarification": None,
            "error": "ollama_unavailable",
            "metadata": {
                "run_id": run_id,
                "ollama_status": ollama_health.to_dict(),
            },
        }


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
            "chart_data": result.get("chart_data"),
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

    elif query_type == "knowledge":
        import time as _time_module
        from langchain_core.messages import HumanMessage
        from langchain_ollama import ChatOllama
        from backend.config import LLM_MODEL
        from backend.agents.prompts import KNOWLEDGE_PROMPT

        t0 = _time_module.time()
        try:
            llm = ChatOllama(model=LLM_MODEL, temperature=0)
            prompt = KNOWLEDGE_PROMPT.format(question=question)
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = (response.content or "").strip()
            gen_time = int((_time_module.time() - t0) * 1000)
            error = None
        except Exception as e:
            answer = "Note: This answer is based on general knowledge, not our patient records.\n\nUnable to retrieve information at this time."
            gen_time = 0
            error = str(e)

        if session_id:
            get_conversation_memory().add_turn(
                session_id, question, answer, query_type="knowledge",
            )

        knowledge_response = {
            "query_type": "knowledge",
            "answer": answer,
            "result": answer,
            "sql_generated": None,
            "confidence": 0.7,
            "sources": [],
            "grounding": None,
            "clarification": None,
            "error": error,
            "metadata": {
                "run_id": run_id,
                "generation_time_ms": gen_time,
            },
        }
        result_cache.set(question, knowledge_response)
        return knowledge_response

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

        # Refusal policy (same as RAG)
        if should_refuse("hybrid", conf["score"], grounding):
            answer = REFUSAL
            if sources:
                snippets = "\n".join(f"- {s.get('text_snippet', '')[:100]}" for s in sources[:3])
                answer += f"\n\nClosest matches found:\n{snippets}"
        elif conf.get("disclaimer"):
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
            "chart_data": result.get("chart_data"),
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
    search: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None


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
    if f.search:
        where += " AND (p.full_name LIKE ? OR n.condition_name LIKE ? OR n.doctor_name LIKE ? OR n.note_text LIKE ? OR r.medication_name LIKE ? OR c.name LIKE ?)"
        term = f"%{f.search}%"
        params.extend([term] * 6)
    if f.from_date:
        where += " AND n.visit_date >= ?"
        params.append(f.from_date)
    if f.to_date:
        where += " AND n.visit_date <= ?"
        params.append(f.to_date)
    return where, params


@app.get("/api/health")
def health_check():
    """
    Health check endpoint with Ollama status.

    Returns:
        - status: "ok" if backend running, "degraded" if Ollama has issues
        - ollama: detailed Ollama health information
    """
    from backend.reliability.ollama_health import check_ollama_health, OllamaStatus

    ollama_health = check_ollama_health()

    # Determine overall status
    if ollama_health.status == OllamaStatus.HEALTHY:
        status = "ok"
        message = "Medical AI Backend Running"
    elif ollama_health.status == OllamaStatus.DEGRADED:
        status = "degraded"
        message = f"Backend running but Ollama missing models: {', '.join(ollama_health.missing_models)}"
    else:
        status = "degraded"
        message = ollama_health.error_message or "Ollama unavailable"

    return {
        "status": status,
        "message": message,
        "ollama": ollama_health.to_dict(),
    }


@app.get("/api/ollama/status")
def ollama_status():
    """
    Check Ollama service status independently.

    Returns detailed information about:
    - Whether Ollama is running
    - Which models are loaded
    - Which required models are missing
    - Latency to Ollama service
    """
    from backend.reliability.ollama_health import check_ollama_health
    health = check_ollama_health(force_refresh=True)
    return health.to_dict()


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


class SecurityTestRequest(BaseModel):
    text: str


@app.post("/api/security/check")
def security_check(req: SecurityTestRequest):
    """
    Test the security input guard without processing the query.

    Useful for testing and debugging security rules.
    """
    from backend.security.input_guard import check_input
    result = check_input(req.text)
    return result.to_dict()


@app.get("/api/rag/metrics")
def rag_metrics_summary():
    """
    Get aggregated RAG pipeline metrics.

    Returns statistics over recent requests including:
    - Quality metrics (relevance, utilization, faithfulness)
    - Precision metrics (P@K, MRR)
    - Latency breakdown by pipeline stage
    - Health status
    """
    from backend.rag.metrics import get_metrics_collector
    collector = get_metrics_collector()
    return collector.get_summary()


@app.get("/api/rag/metrics/{run_id}")
def rag_metrics_by_run(run_id: str):
    """
    Get metrics for a specific run.
    """
    from backend.rag.metrics import get_metrics_collector
    collector = get_metrics_collector()
    return collector.get_by_run_id(run_id)


@app.post("/api/chart/generate")
def generate_chart(req: dict):
    """
    Generate a chart on-demand based on user request.
    
    Request body:
        - chart_type: Type of chart (prescriptions_trend, gender_distribution, etc.)
        - custom_data: Optional custom data for dynamic charts
    
    Returns:
        - chart_url: URL to access the generated chart image
    """
    from backend.services.visualization import generate_chart, detect_chart_request
    
    chart_type = req.get('chart_type')
    user_query = req.get('user_query', '')
    custom_data = req.get('custom_data')
    
    # Auto-detect chart type from query if not provided
    if not chart_type and user_query:
        chart_type = detect_chart_request(user_query)
    
    if not chart_type:
        raise HTTPException(status_code=400, detail="Chart type not specified")
    
    try:
        # Generate the chart
        filepath = generate_chart(chart_type, custom_data)
        
        # Return the URL path (frontend will access via /static/charts/...)
        filename = os.path.basename(filepath)
        chart_url = f"/static/charts/{filename}"
        
        return {
            "success": True,
            "chart_url": chart_url,
            "chart_type": chart_type,
            "message": f"Generated {chart_type} chart"
        }
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chart: {str(e)}")


@app.get("/api/dashboard/charts")
def get_dashboard_charts():
    """
    Get aggregated chart data for the dashboard.
    Returns data for all dashboard charts (trends, medications, age, insurance, gender).
    """
    import pandas as pd
    
    try:
        # Load data
        patients = pd.read_csv(os.path.join(DATA_DIR, 'patients.csv'))
        prescriptions = pd.read_csv(os.path.join(DATA_DIR, 'prescriptions.csv'))
        
        # Convert dates
        prescriptions['last_filled_date'] = pd.to_datetime(prescriptions['last_filled_date'])
        patients['dob'] = pd.to_datetime(patients['dob'])
        
        # 1. Monthly prescription trends
        prescriptions['month'] = prescriptions['last_filled_date'].dt.to_period('M')
        monthly_trend = prescriptions.groupby('month').size().tail(12)
        trend_data = {
            'labels': [str(m) for m in monthly_trend.index],
            'values': monthly_trend.values.tolist()
        }
        
        # 2. Top medications
        top_meds = prescriptions['medication_name'].value_counts().head(8)
        meds_data = {
            'labels': top_meds.index.tolist(),
            'values': top_meds.values.tolist()
        }
        
        # 3. Age distribution
        patients['age'] = (datetime.now() - patients['dob']).dt.days // 365
        age_bins = [0, 18, 30, 45, 60, 75, 100]
        age_labels = ['0-18', '19-30', '31-45', '46-60', '61-75', '76+']
        patients['age_group'] = pd.cut(patients['age'], bins=age_bins, labels=age_labels)
        age_counts = patients['age_group'].value_counts().sort_index()
        age_data = {
            'labels': age_counts.index.tolist(),
            'values': age_counts.values.tolist()
        }
        
        # 4. Insurance providers
        insurance = patients['insurance_provider'].value_counts().head(6)
        insurance_data = {
            'labels': insurance.index.tolist(),
            'values': insurance.values.tolist()
        }
        
        # 5. Gender distribution
        gender = patients['gender'].value_counts()
        gender_data = {
            'labels': gender.index.tolist(),
            'values': gender.values.tolist()
        }
        
        return {
            'trend': trend_data,
            'medications': meds_data,
            'age': age_data,
            'insurance': insurance_data,
            'gender': gender_data
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard charts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# CHAT SESSIONS
# ---------------------------
class CreateSessionRequest(BaseModel):
    session_id: str
    title: Optional[str] = "New Chat"


class RenameSessionRequest(BaseModel):
    title: str


def _ensure_session(session_id: str, title: str = "New Chat"):
    """Create a session row if it doesn't already exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM chat_sessions WHERE session_id = ?", [session_id])
    if not cur.fetchone():
        now = _time()
        cur.execute(
            "INSERT INTO chat_sessions (session_id, title, created_at, updated_at, message_count) VALUES (?,?,?,?,0)",
            [session_id, title[:50], now, now],
        )
        conn.commit()
    conn.close()


def _bump_session(session_id: str):
    """Increment message_count and touch updated_at."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE chat_sessions SET message_count = message_count + 1, updated_at = ? WHERE session_id = ?",
        [_time(), session_id],
    )
    conn.commit()
    conn.close()


@app.get("/api/sessions")
def list_sessions():
    """List all chat sessions, newest first."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT session_id, title, created_at, updated_at, message_count FROM chat_sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/sessions")
def create_session(req: CreateSessionRequest):
    """Create a new chat session."""
    _ensure_session(req.session_id, req.title or "New Chat")
    return {"status": "ok", "session_id": req.session_id}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    """Reconstruct messages for a session from the runs table."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Check if runs table exists
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "runs" not in tables:
        conn.close()
        return []

    rows = conn.execute(
        """SELECT question, query_type, result_json, created_at
           FROM runs WHERE session_id = ? ORDER BY created_at ASC""",
        [session_id],
    ).fetchall()
    conn.close()

    import json
    messages = []
    for r in rows:
        messages.append({"role": "user", "content": r["question"], "timestamp": r["created_at"]})
        try:
            result = json.loads(r["result_json"]) if r["result_json"] else {}
        except (json.JSONDecodeError, TypeError):
            result = {}
        messages.append({
            "role": "assistant",
            "content": result.get("answer", ""),
            "query_type": r["query_type"],
            "sql_generated": result.get("sql_generated"),
            "confidence": result.get("confidence"),
            "sources": result.get("sources"),
            "decomposition": result.get("decomposition"),
            "hybrid_mode": result.get("hybrid_mode"),
            "chart_data": result.get("chart_data"),
            "timestamp": r["created_at"],
        })
    return messages


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a chat session and its runs."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", [session_id])
    # Also clean up runs for this session
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "runs" in tables:
        conn.execute("DELETE FROM runs WHERE session_id = ?", [session_id])
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, req: RenameSessionRequest):
    """Rename a chat session."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE session_id = ?",
        [req.title[:50], _time(), session_id],
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


# ---------------------------
# DASHBOARD CHARTS API
# ---------------------------
@app.get("/api/charts/dashboard")
def dashboard_charts():
    """Return pre-computed chart data for the dashboard from materialized views or raw tables."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    charts = {}

    # Check which tables exist
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    # Conditions doughnut (from mv_condition_stats or clinical_notes)
    try:
        if "mv_condition_stats" in tables:
            rows = conn.execute("SELECT condition_name, patient_count FROM mv_condition_stats ORDER BY patient_count DESC LIMIT 8").fetchall()
        else:
            rows = conn.execute("SELECT condition_name, COUNT(DISTINCT patient_id) as cnt FROM clinical_notes GROUP BY condition_name ORDER BY cnt DESC LIMIT 8").fetchall()
        charts["conditions"] = {
            "chart_type": "doughnut",
            "title": "Top Conditions",
            "labels": [r[0] for r in rows],
            "datasets": [{"data": [r[1] for r in rows], "backgroundColor": ['#0d9488','#3b82f6','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16'], "borderWidth": 0}],
        }
    except Exception:
        pass

    # Clinics bar (from mv_clinic_stats or clinics join)
    try:
        if "mv_clinic_stats" in tables:
            rows = conn.execute("SELECT clinic_name, patient_count FROM mv_clinic_stats ORDER BY patient_count DESC LIMIT 10").fetchall()
        else:
            rows = conn.execute("""
                SELECT c.name, COUNT(DISTINCT p.patient_id) as cnt
                FROM clinics c JOIN patients p ON c.clinic_id = p.clinic_id
                GROUP BY c.name ORDER BY cnt DESC LIMIT 10
            """).fetchall()
        charts["clinics"] = {
            "chart_type": "bar",
            "title": "Patients per Clinic",
            "labels": [r[0] for r in rows],
            "datasets": [{"label": "Patients", "data": [r[1] for r in rows], "backgroundColor": "#0d9488", "borderRadius": 4}],
        }
    except Exception:
        pass

    # Top medications bar
    try:
        if "mv_medication_stats" in tables:
            rows = conn.execute("SELECT medication_name, prescription_count FROM mv_medication_stats ORDER BY prescription_count DESC LIMIT 10").fetchall()
        else:
            rows = conn.execute("SELECT medication_name, COUNT(*) as cnt FROM prescriptions GROUP BY medication_name ORDER BY cnt DESC LIMIT 10").fetchall()
        charts["medications"] = {
            "chart_type": "bar",
            "title": "Top Medications",
            "labels": [r[0] for r in rows],
            "datasets": [{"label": "Prescriptions", "data": [r[1] for r in rows], "backgroundColor": "#3b82f6", "borderRadius": 4}],
        }
    except Exception:
        pass

    # Daily trend line (visits per month)
    try:
        rows = conn.execute("""
            SELECT strftime('%Y-%m', visit_date) as month, COUNT(*) as cnt
            FROM clinical_notes
            WHERE visit_date IS NOT NULL
            GROUP BY month ORDER BY month
        """).fetchall()
        if rows:
            charts["daily_trend"] = {
                "chart_type": "line",
                "title": "Visits per Month",
                "labels": [r[0] for r in rows],
                "datasets": [{"label": "Visits", "data": [r[1] for r in rows], "borderColor": "#0d9488", "backgroundColor": "rgba(13,148,136,0.1)", "fill": True, "tension": 0.3}],
            }
    except Exception:
        pass

    conn.close()
    return charts


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
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    f = FilterRequest(clinic=_norm(clinic), doctor=_norm(doctor), condition=_norm(condition), search=_norm(search))
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
    f.search = _norm(f.search)
    f.from_date = _norm(f.from_date)
    f.to_date = _norm(f.to_date)
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

    # KPI queries (FULL dataset) - use DISTINCT to avoid duplicate counts from joins
    total_sql = f"SELECT COUNT(DISTINCT r.rx_id) as cnt {base_from} {where_sql}"
    total_rows = int(cur.execute(total_sql, params).fetchone()["cnt"])

    uniq_pat_sql = f"SELECT COUNT(DISTINCT p.patient_id) as cnt {base_from} {where_sql}"
    unique_patients = int(cur.execute(uniq_pat_sql, params).fetchone()["cnt"])

    rx_status_sql = f"""
        SELECT r.status as status, COUNT(DISTINCT r.rx_id) as cnt
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

    # Check if doctor_notes column exists
    cur.execute("PRAGMA table_info(clinical_notes)")
    columns = [col[1] for col in cur.fetchall()]
    has_doctor_notes = "doctor_notes" in columns
    dn_col = ", n.doctor_notes" if has_doctor_notes else ""

    page_sql = f"""
        SELECT p.patient_id, r.rx_id, p.full_name, c.name as clinic_name,
               n.doctor_name, n.condition_name, n.note_text, n.visit_date,
               n.note_id,
               r.medication_name, r.dosage, r.days_supply, r.refills_remaining,
               r.last_filled_date, r.status as rx_status
               {dn_col}
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
            days_supply = int(r["days_supply"] or 0)

            # Handle "Not Purchased" (NULL last_filled_date)
            if not r["last_filled_date"]:
                status = "Not Purchased"
                next_steps = "Did Not Buy"
                action_type = "danger"
                refill_due = None
            else:
                last_filled = datetime.strptime(r["last_filled_date"], "%Y-%m-%d")
                days_elapsed = (now - last_filled).days
                ratio = (days_elapsed / days_supply) if days_supply > 0 else 0.0
                refill_due = (last_filled + timedelta(days=days_supply)).strftime("%Y-%m-%d")

                if ratio > 1.2:
                    status, next_steps, action_type = "Non-Adherent", "Call Patient", "danger"
                elif ratio > 0.9:
                    if int(r["refills_remaining"] or 0) > 0:
                        status, next_steps, action_type = "Refill Due", "Call for Refill", "success"
                    else:
                        status, next_steps, action_type = "Renewal Needed", "Book Appointment", "warning"
                else:
                    status, next_steps, action_type = "Good", "Monitor", "info"

            note_text = (r["note_text"] or "")
            note_snippet = note_text[:220] + ("..." if len(note_text) > 220 else "")

            doctor_notes_raw = (r["doctor_notes"] or "") if has_doctor_notes else ""
            dn_snippet = doctor_notes_raw[:300] + ("..." if len(doctor_notes_raw) > 300 else "")

            out_rows.append({
                "patient_id": r["patient_id"],
                "rx_id": r["rx_id"],
                "note_id": r["note_id"],
                "name": r["full_name"],
                "clinic": r["clinic_name"],
                "doctor": r["doctor_name"],
                "condition": r["condition_name"],
                "medication": r["medication_name"],
                "dosage": r["dosage"],
                "note_snippet": note_snippet,
                "doctor_notes_snippet": dn_snippet,
                "has_doctor_notes": bool(doctor_notes_raw.strip()),
                "last_visit": r["visit_date"],
                "status": status,
                "next_steps": next_steps,
                "refill_due_date": refill_due,
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

    # Auto-create session on first query
    if req.session_id:
        _ensure_session(req.session_id, question[:50])

    try:
        response = _process_query(
            question=question,
            session_id=req.session_id,
        )

        # Handle Ollama unavailable gracefully (return response, not error)
        if response.get("error") == "ollama_unavailable":
            return response

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

        # Bump session message count
        if req.session_id:
            try:
                _bump_session(req.session_id)
            except Exception:
                pass

        return response

    except HTTPException:
        raise
    except Exception as e:
        # Check if it's an Ollama connection error
        error_str = str(e).lower()
        if "connect" in error_str and ("11434" in error_str or "ollama" in error_str):
            from backend.reliability.ollama_health import (
                OllamaStatus,
                get_user_friendly_error,
            )
            return {
                "query_type": "error",
                "answer": get_user_friendly_error(OllamaStatus.UNAVAILABLE),
                "result": get_user_friendly_error(OllamaStatus.UNAVAILABLE),
                "sql_generated": None,
                "confidence": 0.0,
                "sources": [],
                "grounding": None,
                "error": "ollama_unavailable",
                "metadata": {},
            }

        logger.exception(f"Query failed: {e}")
        raise HTTPException(500, detail=str(e))


# ---------------------------
# CLINICAL NOTES BROWSING
# ---------------------------
@app.get("/api/clinical-notes")
def browse_clinical_notes(
    search: Optional[str] = Query(None),
    condition: Optional[str] = Query(None),
    doctor: Optional[str] = Query(None),
    has_doctor_notes: Optional[bool] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Browse and search clinical notes with filters."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Check if doctor_notes column exists
    cur.execute("PRAGMA table_info(clinical_notes)")
    columns = [col[1] for col in cur.fetchall()]
    dn_exists = "doctor_notes" in columns
    dn_col = ", n.doctor_notes" if dn_exists else ""

    where_clauses = ["1=1"]
    params: List[Any] = []

    if condition:
        where_clauses.append("n.condition_name = ?")
        params.append(condition)
    if doctor:
        where_clauses.append("n.doctor_name = ?")
        params.append(doctor)
    if has_doctor_notes and dn_exists:
        where_clauses.append("n.doctor_notes IS NOT NULL AND n.doctor_notes != ''")
    if search:
        search_clause = "(n.note_text LIKE ? OR n.condition_name LIKE ? OR p.full_name LIKE ?"
        if dn_exists:
            search_clause += " OR n.doctor_notes LIKE ?"
        search_clause += ")"
        where_clauses.append(search_clause)
        term = f"%{search}%"
        params.extend([term, term, term])
        if dn_exists:
            params.append(term)

    where_sql = " AND ".join(where_clauses)

    # Count
    count_sql = f"""
        SELECT COUNT(*) as cnt
        FROM clinical_notes n
        JOIN patients p ON n.patient_id = p.patient_id
        JOIN clinics c ON p.clinic_id = c.clinic_id
        WHERE {where_sql}
    """
    total = int(cur.execute(count_sql, params).fetchone()["cnt"])

    # Paged results
    offset = (page - 1) * page_size
    data_sql = f"""
        SELECT n.note_id, n.patient_id, n.visit_date, n.doctor_name,
               n.condition_name, n.note_text, n.diagnosis_code,
               p.full_name AS patient_name, c.name AS clinic_name
               {dn_col}
        FROM clinical_notes n
        JOIN patients p ON n.patient_id = p.patient_id
        JOIN clinics c ON p.clinic_id = c.clinic_id
        WHERE {where_sql}
        ORDER BY n.visit_date DESC
        LIMIT ? OFFSET ?
    """
    rows = cur.execute(data_sql, params + [page_size, offset]).fetchall()
    conn.close()

    notes = []
    for r in rows:
        dn = (r["doctor_notes"] or "") if dn_exists else ""
        notes.append({
            "note_id": r["note_id"],
            "patient_id": r["patient_id"],
            "patient_name": r["patient_name"],
            "visit_date": r["visit_date"],
            "doctor_name": r["doctor_name"],
            "condition_name": r["condition_name"],
            "diagnosis_code": r["diagnosis_code"],
            "clinic_name": r["clinic_name"],
            "note_snippet": (r["note_text"] or "")[:200],
            "has_doctor_notes": bool(dn.strip()),
            "doctor_notes_snippet": dn[:300] + ("..." if len(dn) > 300 else ""),
        })

    return {
        "notes": notes,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        },
    }


@app.get("/api/clinical-notes/{note_id}")
def get_clinical_note(note_id: int):
    """Get full clinical note detail."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(clinical_notes)")
    columns = [col[1] for col in cur.fetchall()]
    dn_exists = "doctor_notes" in columns
    dn_col = ", n.doctor_notes" if dn_exists else ""

    row = cur.execute(f"""
        SELECT n.note_id, n.patient_id, n.visit_date, n.doctor_name,
               n.condition_name, n.note_text, n.diagnosis_code,
               p.full_name AS patient_name, p.dob, p.gender,
               p.insurance_provider, c.name AS clinic_name, c.location AS clinic_location
               {dn_col}
        FROM clinical_notes n
        JOIN patients p ON n.patient_id = p.patient_id
        JOIN clinics c ON p.clinic_id = c.clinic_id
        WHERE n.note_id = ?
    """, [note_id]).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "Note not found")

    dn = (row["doctor_notes"] or "") if dn_exists else ""
    return {
        "note_id": row["note_id"],
        "patient_id": row["patient_id"],
        "patient_name": row["patient_name"],
        "dob": row["dob"],
        "gender": row["gender"],
        "insurance_provider": row["insurance_provider"],
        "visit_date": row["visit_date"],
        "doctor_name": row["doctor_name"],
        "condition_name": row["condition_name"],
        "diagnosis_code": row["diagnosis_code"],
        "clinic_name": row["clinic_name"],
        "clinic_location": row["clinic_location"],
        "note_text": row["note_text"],
        "doctor_notes": dn,
        "has_doctor_notes": bool(dn.strip()),
    }


# ---------------------------
# FRONTEND (serve static files from /frontend)
# ---------------------------
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")
app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"\nServer running at: http://localhost:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)

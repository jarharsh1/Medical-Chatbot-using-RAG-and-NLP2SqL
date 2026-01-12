# app.py
# FastAPI backend: SQLite + CSV bootstrap + LangGraph text-to-SQL via Ollama
# ✅ Fixes:
# 1) Quote-loss issue: execute RAW LLM SQL as-is (no cleaning).
# 2) "Top/Popular medicines by condition" issue: schema-aware prompt + auto-repair hint.
# 3) Safety guardrails: SELECT-only, one statement, no placeholders, balanced quotes, no write ops.
# 4) ✅ Dashboard pagination + KPIs computed on FULL filtered data instantly
#    - return only top 50 rows for table
#    - KPIs computed via aggregate queries on entire filtered dataset
#    - response shape: { kpis, rows, pagination }

import os
import csv
import sqlite3
import re
from datetime import datetime
from typing import Optional, TypedDict, Dict, Any, List, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from langgraph.graph import END, StateGraph


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
# PATH CONFIGURATION
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "medical_records.db")
DB_URI = f"sqlite:///{DB_PATH}"

POSSIBLE_DATA_DIRS = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(os.getcwd(), "data"),
    os.path.join(os.getcwd(), "backend", "data"),
]

DATA_DIR = None
for path in POSSIBLE_DATA_DIRS:
    if os.path.exists(path) and os.path.isdir(path):
        DATA_DIR = path
        break


# ---------------------------
# DATABASE INITIALIZATION
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS clinics (
            clinic_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            location TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY,
            full_name TEXT NOT NULL,
            dob TEXT,
            gender TEXT,
            insurance_provider TEXT,
            clinic_id INTEGER,
            FOREIGN KEY(clinic_id) REFERENCES clinics(clinic_id)
        )
        """
    )

    cursor.execute(
        """
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
        """
    )

    cursor.execute(
        """
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
        """
    )

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM clinics")
    count = cursor.fetchone()[0]

    if count == 0 and DATA_DIR:
        load_csv_to_table(conn, "clinics.csv", "clinics")
        load_csv_to_table(conn, "patients.csv", "patients")
        load_csv_to_table(conn, "clinical_notes.csv", "clinical_notes")
        load_csv_to_table(conn, "prescriptions.csv", "prescriptions")

    conn.close()


def load_csv_to_table(conn: sqlite3.Connection, filename: str, table_name: str):
    if not DATA_DIR:
        return

    file_path = os.path.join(DATA_DIR, filename)
    cursor = conn.cursor()

    if not os.path.exists(file_path):
        return

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


init_db()
db = SQLDatabase.from_uri(DB_URI, sample_rows_in_table_info=0)


# ---------------------------
# AI SETUP
# ---------------------------
llm = None
try:
    llm = ChatOllama(model="llama3.2", temperature=0)
except Exception:
    llm = None


# ---------------------------
# AGENT STATE
# ---------------------------
class AgentState(TypedDict):
    question: str
    schema: str
    sql_query: str
    query_result: Optional[str]
    error: Optional[str]
    iterations: int


def get_schema(state: AgentState):
    return {"schema": db.get_table_info(), "iterations": 0, "error": None, "sql_query": ""}


def validate_sql(sql: str) -> tuple[bool, str]:
    if sql is None:
        return False, "SQL is None."

    s = sql.strip()
    if not s:
        return False, "Empty SQL."

    low = s.lstrip().lower()

    if not (low.startswith("select") or low.startswith("with")):
        return False, "Only SELECT/WITH queries are allowed."

    if ";" in s.rstrip().rstrip(";"):
        return False, "Multiple statements detected. Return only one query."

    if "?" in s or re.search(r"[:$]\w+", s):
        return False, "Placeholders detected (?, :param, $1). Inline literals only."

    if s.count("'") % 2 != 0:
        return False, "Unbalanced single quotes detected in SQL."

    banned = ["insert", "update", "delete", "drop", "alter", "create", "pragma", "attach", "detach"]
    for b in banned:
        if re.search(rf"\b{b}\b", low):
            return False, "Non-SELECT operation detected."

    return True, ""


def generate_sql(state: AgentState):
    if llm is None:
        return {
            "sql_query": "",
            "error": "Ollama LLM is not available. Start Ollama and ensure llama3.2 is installed.",
            "iterations": state.get("iterations", 0) + 1,
        }

    schema = state["schema"]
    question = state["question"]
    prev_error = state.get("error")

    system_prompt = (
        "You are a senior SQLite expert.\n"
        "Return EXACTLY ONE SQLite SELECT query that answers the question.\n\n"
        "DATABASE FACTS (VERY IMPORTANT):\n"
        "- clinical_notes has: patient_id, visit_date, doctor_name, diagnosis_code, condition_name, note_text\n"
        "- prescriptions has: patient_id, medication_name, dosage, days_supply, refills_remaining, last_filled_date, status\n"
        "- patients has: patient_id, clinic_id\n"
        "- clinics has: clinic_id, name, location\n"
        "- To filter by condition/diagnosis, you MUST use clinical_notes.\n"
        "- To return medications, you MUST use prescriptions.\n"
        "- Link condition -> meds via patient_id (JOIN clinical_notes.patient_id = prescriptions.patient_id).\n\n"
        "HARD RULES:\n"
        "1) Output ONLY the SQL query. No explanations. No markdown.\n"
        "2) NEVER use parameter placeholders (?, :param, $1). Inline literals instead.\n"
        "3) Do not invent columns. Use only columns from schema.\n"
        "4) If question asks 'top', 'most popular', 'most prescribed':\n"
        "   - use COUNT(*) as cnt\n"
        "   - GROUP BY medication_name\n"
        "   - ORDER BY cnt DESC\n"
        "   - LIMIT N\n"
        "5) prescriptions.status values are exactly 'Active' or 'Expired' (case-sensitive).\n"
        "6) Use LIKE with wildcards for text filters (e.g., condition_name LIKE '%Hypertension%').\n"
        "7) One statement only.\n"
    )

    user_prompt = f"Schema:\n{schema}\n\nQuestion:\n{question}\n"
    if prev_error:
        user_prompt += f"\nPrevious SQL error:\n{prev_error}\nReturn corrected SQL only."

    res = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    raw = res.content or ""

    ok, err = validate_sql(raw)
    if not ok:
        return {
            "sql_query": "",
            "error": err,
            "iterations": state.get("iterations", 0) + 1,
        }

    return {"sql_query": raw, "error": None, "iterations": state.get("iterations", 0) + 1}


def execute_sql(state: AgentState):
    sql = state.get("sql_query")
    if not sql:
        return {"error": "Empty SQL query. Nothing to execute.", "query_result": None}

    ok, err = validate_sql(sql)
    if not ok:
        return {"error": f"SQL blocked: {err}", "query_result": None}

    try:
        res = db.run(sql)
        return {"query_result": str(res), "error": None}
    except Exception as e:
        msg = str(e)
        if "no such column: condition_name" in msg or "no such column: diagnosis_code" in msg:
            hint = (
                "You referenced a column that does not exist in that table. "
                "condition_name and diagnosis_code exist in clinical_notes, NOT prescriptions. "
                "To filter by condition and return medications, JOIN clinical_notes and prescriptions "
                "ON patient_id, then GROUP BY prescriptions.medication_name and ORDER BY COUNT(*) DESC."
            )
            return {"error": hint, "query_result": None}
        return {"error": msg, "query_result": None}


def should_continue(state: AgentState):
    return "retry" if state.get("error") and state.get("iterations", 0) < 3 else "end"


workflow = StateGraph(AgentState)
workflow.add_node("get_schema", get_schema)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)

workflow.set_entry_point("get_schema")
workflow.add_edge("get_schema", "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_conditional_edges("execute_sql", should_continue, {"retry": "generate_sql", "end": END})

agent_app = workflow.compile()


# ---------------------------
# API ROUTES
# ---------------------------
class FilterRequest(BaseModel):
    clinic: Optional[str] = None
    doctor: Optional[str] = None
    condition: Optional[str] = None


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


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Medical AI Backend Running"}


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


# ✅ GET dashboard with pagination (frontend can keep POST; GET is useful too)
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


# ✅ POST dashboard with pagination (fits your current frontend pattern)
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

    # Shared FROM/JOIN
    base_from = """
        FROM patients p
        JOIN clinics c ON p.clinic_id = c.clinic_id
        JOIN clinical_notes n ON p.patient_id = n.patient_id
        JOIN prescriptions r ON p.patient_id = r.patient_id
    """

    where_sql, params = _build_where_and_params(f)

    # --- KPI queries (FULL dataset, no LIMIT) ---
    # Total rows for pagination
    total_sql = f"SELECT COUNT(*) as cnt {base_from} {where_sql}"
    total_rows = int(cur.execute(total_sql, params).fetchone()["cnt"])

    # Unique patients in filtered set
    uniq_pat_sql = f"SELECT COUNT(DISTINCT p.patient_id) as cnt {base_from} {where_sql}"
    unique_patients = int(cur.execute(uniq_pat_sql, params).fetchone()["cnt"])

    # Rx status breakdown (Active/Expired) on FULL dataset
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

    # --- Paged table query (only page_size rows) ---
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

    # Convert to UI-friendly rows (status/action computed per row)
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

            out_rows.append(
                {
                    # table + card
                    "patient_id": r["patient_id"],
                    "name": r["full_name"],
                    "clinic": r["clinic_name"],
                    "doctor": r["doctor_name"],
                    "condition": r["condition_name"],
                    "medication": r["medication_name"],
                    "dosage": r["dosage"],
                    "note_snippet": note_snippet,
                    "last_visit": r["visit_date"],

                    # IMPORTANT: frontend expects these keys
                    "status": status,
                    "action": action,
                    "refills_left": r["refills_remaining"],

                    # optional debug
                    "rx_status": r["rx_status"],
                    "last_filled_date": r["last_filled_date"],
                    "days_supply": r["days_supply"],
                    "action_type": action_type,
                }
            )
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
def query_ai(req: dict):
    question = (req.get("question") or "")
    if question == "":
        raise HTTPException(400, "Question required")

    try:
        res = agent_app.invoke({"question": question})
        if res.get("error"):
            raise Exception(res["error"])

        result_str = res.get("query_result") or ""
        if result_str == "" or result_str == "[]":
            result_str = "No matching records found. Try simplifying your query."

        return {"sql_generated": res.get("sql_query"), "result": result_str}
    except Exception as e:
        raise HTTPException(500, detail=str(e))


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    # Keep fixed port (frontend uses localhost:8000)
    port = 8000
    print(f"\n🚀 Server running at: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

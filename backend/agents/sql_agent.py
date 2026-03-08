"""
SQL Agent: text-to-SQL pipeline extracted from the original app.py.

Features:
  - Schema retriever: only includes relevant tables/columns
  - SQL validation: SELECT-only, no placeholders, enforced LIMIT
  - Retry up to 3 times with error feedback
  - Few-shot examples from long-term query cache
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.agents.prompts import SQL_SYSTEM_PROMPT, SQL_USER_PROMPT, SQL_ANSWER_PROMPT
from backend.config import DB_URI, LLM_MODEL, MAX_SQL_RETRIES, SQL_BANNED_OPS

logger = logging.getLogger(__name__)

# Module-level DB and LLM
_db = None
_llm = None


def _get_db() -> SQLDatabase:
    global _db
    if _db is None:
        _db = SQLDatabase.from_uri(DB_URI, sample_rows_in_table_info=3)
    return _db


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, temperature=0)
    return _llm


def validate_sql(sql: str) -> Tuple[bool, str]:
    """Strict SQL validation: SELECT-only, single statement, no placeholders."""
    if sql is None:
        return False, "SQL is None."

    s = sql.strip().rstrip(";")
    if not s:
        return False, "Empty SQL."

    low = s.lstrip().lower()

    if not (low.startswith("select") or low.startswith("with")):
        return False, "Only SELECT/WITH queries are allowed."

    if ";" in s:
        return False, "Multiple statements detected. Return only one query."

    if "?" in s or re.search(r"[:$]\w+", s):
        return False, "Placeholders detected (?, :param, $1). Inline literals only."

    if s.count("'") % 2 != 0:
        return False, "Unbalanced single quotes detected in SQL."

    for b in SQL_BANNED_OPS:
        if re.search(rf"\b{b}\b", low):
            return False, f"Non-SELECT operation detected: {b}."

    # Enforce LIMIT
    if "limit" not in low:
        s += " LIMIT 100"

    return True, s


_MAIN_TABLES = ["clinics", "patients", "clinical_notes", "prescriptions"]


def get_relevant_schema(question: str) -> str:
    """
    Schema retriever: only includes the 4 domain tables.
    Excludes internal tables (indexed_notes, query_patterns, runs, etc.).
    """
    db = _get_db()
    return db.get_table_info(table_names=_MAIN_TABLES)


def generate_and_execute(
    question: str,
    conversation_context: str = "",
    few_shot_examples: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate SQL from a natural language question, validate, and execute.
    Retries up to MAX_SQL_RETRIES times on failure.

    Returns {sql_query, query_result, error, iterations, generation_time_ms}.
    """
    from backend.agents.query_rewriter import rewrite_for_sql

    start = time.time()
    llm = _get_llm()
    db = _get_db()

    # Rewrite vague questions into precise SQL-friendly queries
    rewritten_question = rewrite_for_sql(question)
    if rewritten_question != question:
        logger.info(f"Query rewritten for SQL: {question[:50]} → {rewritten_question[:50]}")

    schema = get_relevant_schema(rewritten_question)

    error = None
    sql_query = ""
    query_result = None
    iterations = 0

    few_shot_context = ""
    if few_shot_examples:
        few_shot_context = "\n\nSimilar successful queries (use as examples):\n" + "\n".join(
            f"  {i+1}. {ex}" for i, ex in enumerate(few_shot_examples[:3])
        )

    for attempt in range(MAX_SQL_RETRIES):
        iterations += 1

        error_context = ""
        if error:
            error_context = f"\nPrevious SQL error:\n{error}\nReturn corrected SQL only."

        user_prompt = SQL_USER_PROMPT.format(
            schema=schema,
            question=rewritten_question,
            error_context=error_context,
            few_shot_context=few_shot_context,
        )

        if conversation_context:
            user_prompt = f"Conversation context:\n{conversation_context}\n\n{user_prompt}"

        try:
            res = llm.invoke([
                SystemMessage(content=SQL_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            raw = (res.content or "").strip()

            ok, validated = validate_sql(raw)
            if not ok:
                error = validated
                continue

            sql_query = validated

            # Execute
            try:
                result = db.run(sql_query)
                query_result = str(result)
                error = None
                break
            except Exception as e:
                msg = str(e)
                if "no such column" in msg:
                    error = (
                        f"SQL error: {msg}\n"
                        "REMINDER — columns per table:\n"
                        "  clinical_notes: note_id, patient_id, visit_date, doctor_name, diagnosis_code, condition_name, note_text\n"
                        "  patients: patient_id, full_name, dob, gender, insurance_provider, clinic_id\n"
                        "  prescriptions: rx_id, patient_id, medication_name, dosage, days_supply, refills_remaining, last_filled_date, status\n"
                        "  clinics: clinic_id, name, location\n"
                        "doctor_name is in clinical_notes, NOT patients. "
                        "condition_name and diagnosis_code are in clinical_notes, NOT prescriptions.\n"
                        "CRITICAL: full_name is ONLY in patients table. To use p.full_name you MUST "
                        "define the alias: FROM patients p (or JOIN patients p ON ...). "
                        "Never reference an alias that is not explicitly defined in FROM/JOIN."
                    )
                else:
                    error = msg

        except Exception as e:
            error = f"LLM invocation failed: {e}"

    if query_result and (query_result == "" or query_result == "[]"):
        query_result = "No matching records found. Try simplifying your query."

    # Generate natural language answer from raw result
    nl_answer = query_result
    if query_result and not error and query_result != "No matching records found. Try simplifying your query.":
        try:
            prompt = SQL_ANSWER_PROMPT.format(
                question=question,
                sql_query=sql_query,
                result=query_result[:2000],  # truncate large results
            )
            res = llm.invoke([HumanMessage(content=prompt)])
            nl_answer = (res.content or "").strip() or query_result
        except Exception as e:
            logger.warning(f"SQL answer formatting failed: {e}")
            nl_answer = query_result

    # Detect chart-worthy data from raw result
    chart_data = _detect_chart_data(question, sql_query, query_result)

    elapsed = int((time.time() - start) * 1000)

    return {
        "sql_query": sql_query,
        "query_result": nl_answer,
        "raw_result": query_result,
        "chart_data": chart_data,
        "error": error,
        "iterations": iterations,
        "generation_time_ms": elapsed,
    }


def _detect_chart_data(
    question: str, sql_query: str, raw_result: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Parse raw SQL result and detect if it's chart-worthy.

    Returns a Chart.js-compatible spec or None.
    Heuristics:
      - 2-column results (label, numeric) → chart
      - <=6 categories → doughnut, >6 → bar
      - Date-like first column → line chart
    """
    if not raw_result or raw_result in ("[]", "", "No matching records found. Try simplifying your query."):
        return None

    try:
        import ast
        parsed = ast.literal_eval(raw_result)
    except Exception:
        return None

    if not isinstance(parsed, list) or len(parsed) < 2:
        return None

    # Check for 2-column tuples
    first = parsed[0]
    if not isinstance(first, (list, tuple)) or len(first) != 2:
        return None

    labels = []
    values = []
    for row in parsed:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return None
        label, val = row
        if label is None:
            label = "Unknown"
        try:
            numeric_val = float(val)
        except (TypeError, ValueError):
            return None
        labels.append(str(label))
        values.append(numeric_val)

    # Determine chart type
    date_pattern = re.compile(r"^\d{4}-\d{2}")
    is_date = all(date_pattern.match(str(l)) for l in labels)

    if is_date:
        chart_type = "line"
    elif len(labels) <= 6:
        chart_type = "doughnut"
    else:
        chart_type = "bar"

    # Color palette
    colors = ['#0d9488', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899',
              '#06b6d4', '#84cc16', '#f97316', '#6366f1', '#14b8a6', '#e11d48']

    title = question[:60]

    if chart_type == "doughnut":
        return {
            "chart_type": "doughnut",
            "title": title,
            "labels": labels,
            "datasets": [{
                "data": values,
                "backgroundColor": colors[:len(labels)],
                "borderWidth": 0,
            }],
        }
    elif chart_type == "line":
        return {
            "chart_type": "line",
            "title": title,
            "labels": labels,
            "datasets": [{
                "label": "Count",
                "data": values,
                "borderColor": "#0d9488",
                "backgroundColor": "rgba(13,148,136,0.1)",
                "fill": True,
                "tension": 0.3,
            }],
        }
    else:  # bar
        return {
            "chart_type": "bar",
            "title": title,
            "labels": labels,
            "datasets": [{
                "label": "Count",
                "data": values,
                "backgroundColor": colors[:len(labels)],
                "borderRadius": 4,
            }],
        }

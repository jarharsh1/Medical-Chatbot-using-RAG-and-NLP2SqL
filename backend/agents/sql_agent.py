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
    start = time.time()
    llm = _get_llm()
    db = _get_db()
    schema = get_relevant_schema(question)

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
            question=question,
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
                        "condition_name and diagnosis_code are in clinical_notes, NOT prescriptions."
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

    elapsed = int((time.time() - start) * 1000)

    return {
        "sql_query": sql_query,
        "query_result": nl_answer,
        "raw_result": query_result,
        "error": error,
        "iterations": iterations,
        "generation_time_ms": elapsed,
    }

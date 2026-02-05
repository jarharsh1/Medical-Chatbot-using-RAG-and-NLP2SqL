"""
Hybrid Agent: combines RAG retrieval with SQL execution.

Two modes:
  - Filter Mode (confidence >= 0.7): inject patient_ids as hard WHERE constraint
  - Assist Mode (lower confidence): suggest context but don't hard-constrain

Prevents retrieval errors from contaminating SQL answers.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from backend.agents.prompts import HYBRID_ASSIST_PROMPT
from backend.agents.rag_agent import retrieve_and_generate
from backend.agents.sql_agent import generate_and_execute
from backend.config import CONFIDENCE_THRESHOLDS

logger = logging.getLogger(__name__)


def _extract_patient_ids(docs: List[Dict]) -> List[int]:
    """Extract unique patient_ids from retrieved documents."""
    ids = set()
    for doc in docs:
        metadata = doc.get("metadata", {})
        pid = metadata.get("patient_id")
        if pid is not None:
            ids.add(int(pid))
    return sorted(ids)


def run_hybrid(
    question: str,
    conversation_context: str = "",
    few_shot_examples: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Hybrid pipeline: RAG retrieval → choose mode → SQL execution → combine.

    Returns {
        answer, sql_generated, sql_result, sources, confidence,
        hybrid_mode, grounding, retrieval_time_ms, generation_time_ms
    }
    """
    total_start = time.time()

    # ---- Step 1: RAG retrieval ----
    rag_result = retrieve_and_generate(
        question=question,
        conversation_context=conversation_context,
    )

    retrieved_docs = rag_result.get("retrieved_docs", [])
    rag_confidence = rag_result.get("confidence", 0.0)
    retrieval_time = rag_result.get("retrieval_time_ms", 0)

    # ---- Step 2: Choose mode ----
    patient_ids = _extract_patient_ids(retrieved_docs)
    high_threshold = CONFIDENCE_THRESHOLDS["hybrid"]["high"]

    if rag_confidence >= high_threshold and 0 < len(patient_ids) <= 100:
        mode = "filter"
        logger.info(f"Hybrid mode: FILTER (confidence={rag_confidence:.2f}, patients={len(patient_ids)})")
    else:
        mode = "assist"
        logger.info(f"Hybrid mode: ASSIST (confidence={rag_confidence:.2f}, patients={len(patient_ids)})")

    # ---- Step 3: SQL execution ----
    modified_question = question
    if mode == "filter" and patient_ids:
        ids_str = ", ".join(str(pid) for pid in patient_ids)
        # Summarize what RAG found so the SQL agent has context
        conditions = set()
        medications = set()
        for d in retrieved_docs[:10]:
            meta = d.get("metadata", {})
            if meta.get("condition"):
                conditions.add(meta["condition"])
            content = d.get("content", "")
            # Extract medication names from note text
            if "medication_name" in meta:
                medications.add(meta["medication_name"])
        context_hint = ""
        if conditions:
            context_hint += f"Conditions found: {', '.join(conditions)}. "
        if medications:
            context_hint += f"Medications found: {', '.join(medications)}. "
        modified_question = (
            f"{question}\n\n"
            f"IMPORTANT: Only consider patients with patient_id IN ({ids_str}). "
            f"These were identified from clinical notes matching the query. {context_hint}"
            f"Use prescriptions.medication_name to filter/count medications, NOT condition_name."
        )
    elif mode == "assist" and retrieved_docs:
        rag_context = "\n".join(
            f"- {d.get('metadata', {}).get('patient_name', 'Unknown')}: {d.get('content', '')[:150]}"
            for d in retrieved_docs[:5]
        )
        assist_context = HYBRID_ASSIST_PROMPT.format(rag_context=rag_context)
        modified_question = f"{question}\n\n{assist_context}"

    sql_result = generate_and_execute(
        question=modified_question,
        conversation_context=conversation_context,
        few_shot_examples=few_shot_examples,
    )

    generation_time = sql_result.get("generation_time_ms", 0)

    # ---- Step 4: Combine results ----
    sql_answer = sql_result.get("query_result", "")
    rag_answer = rag_result.get("answer", "")

    if sql_answer and rag_answer and sql_answer != "No matching records found. Try simplifying your query.":
        combined_answer = (
            f"Based on the clinical records and database:\n\n"
            f"**Database Results:**\n{sql_answer}\n\n"
            f"**Clinical Note Context:**\n{rag_answer}"
        )
    elif sql_answer:
        combined_answer = sql_answer
    else:
        combined_answer = rag_answer

    total_time = int((time.time() - total_start) * 1000)

    return {
        "answer": combined_answer,
        "sql_generated": sql_result.get("sql_query", ""),
        "sql_result": sql_answer,
        "rag_answer": rag_answer,
        "sources": rag_result.get("sources", []),
        "confidence": rag_confidence,
        "hybrid_mode": mode,
        "retrieved_docs": retrieved_docs,
        "error": sql_result.get("error"),
        "retrieval_time_ms": retrieval_time,
        "generation_time_ms": generation_time,
        "total_time_ms": total_time,
    }

"""
Query Orchestrator: coordinates execution of simple and multi-part questions.

For multi-part questions:
1. Decomposes into sub-questions
2. Routes each sub-question independently
3. Executes in dependency order
4. Combines answers coherently
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from backend.agents.decomposer import decompose_question, is_simple_question
from backend.agents.prompts import COMBINE_ANSWERS_PROMPT
from backend.config import get_active_model

logger = logging.getLogger(__name__)


def _execute_sub_question(
    sub_q: Dict[str, Any],
    conversation_context: str,
    few_shot_examples: Optional[List[str]],
    previous_answers: Dict[int, str],
) -> Dict[str, Any]:
    """
    Execute a single sub-question with its designated route.

    Returns result dict with answer, sources, confidence, etc.
    """
    from backend.agents.sql_agent import generate_and_execute
    from backend.agents.rag_agent import retrieve_and_generate
    from backend.agents.hybrid_agent import run_hybrid
    from backend.guardrails.grounding import check_grounding
    from backend.guardrails.confidence import compute_confidence
    from backend.guardrails.attribution import format_sources

    question = sub_q["sub_question"]
    route = sub_q["route"]
    depends_on = sub_q.get("depends_on")

    # If this depends on a previous answer, inject it as context
    if depends_on is not None and depends_on in previous_answers:
        prev_context = previous_answers[depends_on]
        question = f"Context from previous answer: {prev_context}\n\nQuestion: {question}"

    if route == "sql":
        result = generate_and_execute(
            question=question,
            conversation_context=conversation_context,
            few_shot_examples=few_shot_examples,
        )
        answer = result.get("query_result", "")
        if not answer or answer == "[]":
            answer = "No matching records found."

        return {
            "route": "sql",
            "answer": answer,
            "sql_generated": result.get("sql_query", ""),
            "confidence": 1.0 if not result.get("error") else 0.3,
            "sources": [],
            "error": result.get("error"),
            "generation_time_ms": result.get("generation_time_ms", 0),
        }

    elif route == "rag":
        result = retrieve_and_generate(
            question=question,
            conversation_context=conversation_context,
        )
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        retrieved_docs = result.get("retrieved_docs", [])

        grounding = check_grounding(answer, sources)
        conf = compute_confidence(
            query_type="rag",
            retrieved_docs=retrieved_docs,
            grounding_result=grounding,
            llm_self_confidence=result.get("llm_self_confidence", 0.5),
        )

        return {
            "route": "rag",
            "answer": answer,
            "sql_generated": None,
            "confidence": conf["score"],
            "sources": format_sources(sources),
            "grounding": grounding,
            "error": None,
            "retrieval_time_ms": result.get("retrieval_time_ms", 0),
            "generation_time_ms": result.get("generation_time_ms", 0),
        }

    elif route == "knowledge":
        # For general knowledge questions, use LLM directly without RAG
        from backend.agents.prompts import KNOWLEDGE_PROMPT
        llm = ChatOllama(model=get_active_model(), temperature=0)
        prompt = KNOWLEDGE_PROMPT.format(question=question)

        try:
            t0 = time.time()
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = (response.content or "").strip()
            gen_time = int((time.time() - t0) * 1000)

            return {
                "route": "knowledge",
                "answer": answer,
                "sql_generated": None,
                "confidence": 0.7,  # general knowledge has moderate confidence
                "sources": [],
                "error": None,
                "generation_time_ms": gen_time,
            }
        except Exception as e:
            return {
                "route": "knowledge",
                "answer": "Unable to retrieve medical knowledge at this time.",
                "sql_generated": None,
                "confidence": 0.0,
                "sources": [],
                "error": str(e),
                "generation_time_ms": 0,
            }

    else:  # hybrid or fallback
        result = run_hybrid(
            question=question,
            conversation_context=conversation_context,
            few_shot_examples=few_shot_examples,
        )
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        retrieved_docs = result.get("retrieved_docs", [])

        rag_answer = result.get("rag_answer", "")
        grounding = check_grounding(rag_answer, sources) if rag_answer else None

        conf = compute_confidence(
            query_type="hybrid",
            retrieved_docs=retrieved_docs,
            grounding_result=grounding,
            llm_self_confidence=result.get("confidence", 0.5),
        )

        return {
            "route": "hybrid",
            "answer": answer,
            "sql_generated": result.get("sql_generated", ""),
            "confidence": conf["score"],
            "sources": format_sources(sources),
            "grounding": grounding,
            "hybrid_mode": result.get("hybrid_mode"),
            "error": result.get("error"),
            "retrieval_time_ms": result.get("retrieval_time_ms", 0),
            "generation_time_ms": result.get("generation_time_ms", 0),
            "total_time_ms": result.get("total_time_ms", 0),
        }


def _combine_answers(
    original_question: str,
    sub_questions: List[Dict],
    sub_results: List[Dict],
) -> str:
    """Use LLM to combine sub-answers into a coherent response."""
    # Format sub-answers for the prompt
    parts = []
    for i, (sq, sr) in enumerate(zip(sub_questions, sub_results)):
        route_label = {
            "sql": "Database Query",
            "rag": "Clinical Notes",
            "knowledge": "Medical Knowledge",
            "hybrid": "Combined Analysis",
        }.get(sq["route"], sq["route"])

        parts.append(f"**Part {i+1} ({route_label})**: {sq['sub_question']}\n**Answer**: {sr['answer']}")

    sub_answers_text = "\n\n".join(parts)

    llm = ChatOllama(model=get_active_model(), temperature=0)
    prompt = COMBINE_ANSWERS_PROMPT.format(
        original_question=original_question,
        sub_answers=sub_answers_text,
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return (response.content or "").strip()
    except Exception as e:
        logger.error(f"Answer combination failed: {e}")
        # Fallback: just concatenate
        fallback_parts = []
        for sq, sr in zip(sub_questions, sub_results):
            fallback_parts.append(f"**{sq['sub_question']}**\n{sr['answer']}")
        return "\n\n".join(fallback_parts)


def orchestrate_query(
    question: str,
    session_id: Optional[str] = None,
    conversation_context: str = "",
    few_shot_examples: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Main orchestration function. Handles both simple and multi-part questions.

    For simple questions: direct routing
    For multi-part questions: decompose → execute each → combine

    Returns unified response dict.
    """
    run_id = str(uuid.uuid4())
    start_time = time.time()

    # Quick check if decomposition is needed
    if is_simple_question(question):
        # Use direct routing for simple questions
        logger.info(f"[{run_id}] Simple question detected, using direct routing")
        return None  # Signal to use existing _process_query

    # Decompose the question
    logger.info(f"[{run_id}] Analyzing question for decomposition...")
    sub_questions = decompose_question(question)

    # If decomposition returns single item, use direct routing
    if len(sub_questions) == 1 and sub_questions[0]["route"] != "knowledge":
        logger.info(f"[{run_id}] Single sub-question, using direct routing")
        return None

    logger.info(f"[{run_id}] Decomposed into {len(sub_questions)} parts: {[sq['route'] for sq in sub_questions]}")

    # Execute sub-questions in order (respecting dependencies)
    sub_results: List[Dict] = []
    previous_answers: Dict[int, str] = {}
    all_sources: List[Dict] = []
    all_sql: List[str] = []
    total_confidence = 0.0

    for i, sq in enumerate(sub_questions):
        logger.info(f"[{run_id}] Executing part {i+1}/{len(sub_questions)}: {sq['route']} - {sq['sub_question'][:50]}...")

        result = _execute_sub_question(
            sq,
            conversation_context,
            few_shot_examples,
            previous_answers,
        )
        sub_results.append(result)
        previous_answers[i] = result["answer"]

        # Aggregate sources and SQL
        if result.get("sources"):
            all_sources.extend(result["sources"])
        if result.get("sql_generated"):
            all_sql.append(result["sql_generated"])
        total_confidence += result.get("confidence", 0.5)

    # Combine answers
    combined_answer = _combine_answers(question, sub_questions, sub_results)

    # Deduplicate sources
    seen_doc_ids = set()
    unique_sources = []
    for src in all_sources:
        doc_id = src.get("doc_id", "")
        if doc_id and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            unique_sources.append(src)

    elapsed_ms = int((time.time() - start_time) * 1000)
    avg_confidence = total_confidence / len(sub_questions) if sub_questions else 0.5

    return {
        "query_type": "orchestrated",
        "answer": combined_answer,
        "result": combined_answer,  # backward compat
        "sql_generated": "; ".join(all_sql) if all_sql else None,
        "confidence": round(avg_confidence, 2),
        "sources": unique_sources,
        "grounding": None,  # TODO: aggregate grounding from sub-results
        "clarification": None,
        "error": None,
        "decomposition": {
            "sub_questions": [
                {
                    "question": sq["sub_question"],
                    "route": sq["route"],
                    "answer": sr["answer"][:200] + "..." if len(sr["answer"]) > 200 else sr["answer"],
                    "confidence": sr.get("confidence", 0.5),
                }
                for sq, sr in zip(sub_questions, sub_results)
            ],
            "parts_count": len(sub_questions),
        },
        "metadata": {
            "run_id": run_id,
            "total_time_ms": elapsed_ms,
            "parts_executed": len(sub_questions),
            "routes_used": list(set(sq["route"] for sq in sub_questions)),
        },
    }

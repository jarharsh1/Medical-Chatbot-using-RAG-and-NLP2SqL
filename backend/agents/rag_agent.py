"""
RAG Agent: retrieval-augmented generation over clinical notes.

Pipeline:
  4-stage retrieval → fit_context_window → generate with citations → grounding check
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from backend.agents.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT
from backend.config import get_active_model, MMR_TOP_K

logger = logging.getLogger(__name__)


def _format_context(docs: List[Dict]) -> str:
    """Format retrieved documents as context string with doc_ids."""
    parts = []
    for doc in docs:
        doc_id = doc.get("doc_id", "unknown")
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})
        patient = metadata.get("patient_name", "Unknown")
        condition = metadata.get("condition_name", "")
        visit = metadata.get("visit_date", "")

        header = f"[{doc_id}] Patient: {patient} | Condition: {condition} | Visit: {visit}"
        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


def _extract_confidence(response_text: str) -> float:
    """Extract the CONFIDENCE: X.X line from LLM response."""
    match = re.search(r"CONFIDENCE:\s*([\d.]+)", response_text, re.IGNORECASE)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass
    return 0.5  # default moderate confidence


def _clean_answer(response_text: str) -> str:
    """Remove the CONFIDENCE line from the answer text."""
    return re.sub(r"\n*CONFIDENCE:\s*[\d.]+\s*$", "", response_text, flags=re.IGNORECASE).strip()


def _extract_cited_doc_ids(answer: str) -> List[str]:
    """Extract all [Note doc_id] citations from the answer."""
    return re.findall(r"\[Note\s+(note:\d+)\]", answer)


def retrieve_and_generate(
    question: str,
    conversation_context: str = "",
    top_k: int = MMR_TOP_K,
    where_filter: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → fit context → generate → extract citations.

    Returns {
        answer, sources, confidence, llm_self_confidence,
        retrieved_docs, retrieval_time_ms, generation_time_ms
    }
    """
    from backend.rag.retriever import retrieve
    from backend.rag.context_window import get_context_manager

    # ---- Retrieval ----
    t0 = time.time()
    docs, _ = retrieve(question, top_k=top_k, where_filter=where_filter)
    retrieval_time = int((time.time() - t0) * 1000)

    if not docs:
        return {
            "answer": "I don't have enough information in the clinical records to answer this question.",
            "sources": [],
            "confidence": 0.0,
            "llm_self_confidence": 0.0,
            "retrieved_docs": [],
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": 0,
        }

    # ---- Fit context window ----
    ctx_mgr = get_context_manager()
    fitted_docs = ctx_mgr.fit_documents(docs)
    context = _format_context(fitted_docs)

    # ---- Generation ----
    t1 = time.time()
    llm = ChatOllama(model=get_active_model(), temperature=0)

    user_prompt = RAG_USER_PROMPT.format(context=context, question=question)
    if conversation_context:
        user_prompt = f"Conversation context:\n{conversation_context}\n\n{user_prompt}"

    response = llm.invoke([
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    raw_answer = (response.content or "").strip()
    generation_time = int((time.time() - t1) * 1000)

    llm_confidence = _extract_confidence(raw_answer)
    answer = _clean_answer(raw_answer)
    cited_ids = _extract_cited_doc_ids(answer)

    # Build sources list from cited doc_ids
    sources = []
    for doc in fitted_docs:
        doc_id = doc.get("doc_id", "")
        metadata = doc.get("metadata", {})
        relevance = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))

        sources.append({
            "doc_id": doc_id,
            "note_id": metadata.get("note_id"),
            "patient_name": metadata.get("patient_name", ""),
            "condition": metadata.get("condition_name", ""),
            "visit_date": metadata.get("visit_date", ""),
            "relevance_score": round(relevance, 3),
            "text_snippet": doc.get("content", "")[:200],
            "cited": doc_id in cited_ids,
        })

    return {
        "answer": answer,
        "sources": sources,
        "confidence": llm_confidence,
        "llm_self_confidence": llm_confidence,
        "retrieved_docs": docs,
        "retrieval_time_ms": retrieval_time,
        "generation_time_ms": generation_time,
    }

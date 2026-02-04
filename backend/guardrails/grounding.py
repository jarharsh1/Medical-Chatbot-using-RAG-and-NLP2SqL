"""
Grounding Validation: post-generation fact-checking against retrieved sources.

Separate LLM call checks if every claim in the answer is supported by sources.
Returns grounding_score that feeds into the coverage signal for confidence scoring.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from backend.agents.prompts import GROUNDING_PROMPT
from backend.config import LLM_MODEL

logger = logging.getLogger(__name__)


def check_grounding(answer: str, sources: List[Dict]) -> Dict[str, Any]:
    """
    Validate whether every claim in the answer is supported by the source documents.

    Args:
        answer: The generated answer text.
        sources: List of source dicts with at least 'doc_id' and 'text_snippet'.

    Returns:
        {
            is_grounded: bool,
            supported_sentences: int,
            total_sentences: int,
            grounding_score: float (0.0-1.0),
            unsupported_claims: [str],
            grounding_time_ms: int,
        }
    """
    if not answer or not sources:
        return {
            "is_grounded": not bool(answer),
            "supported_sentences": 0,
            "total_sentences": 0,
            "grounding_score": 0.0 if answer else 1.0,
            "unsupported_claims": [],
            "grounding_time_ms": 0,
        }

    # Format sources for the prompt
    sources_text = "\n\n".join(
        f"[{s.get('doc_id', 'unknown')}] {s.get('patient_name', '')} | "
        f"{s.get('condition', '')} | {s.get('visit_date', '')}\n"
        f"{s.get('text_snippet', '')}"
        for s in sources
    )

    prompt = GROUNDING_PROMPT.format(sources=sources_text, answer=answer)

    t0 = time.time()
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = (response.content or "").strip()
        elapsed = int((time.time() - t0) * 1000)

        result = _parse_grounding_response(raw)
        result["grounding_time_ms"] = elapsed
        return result

    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        logger.error(f"Grounding check failed: {e}")
        return {
            "is_grounded": True,  # fail-open: don't block answers on grounding errors
            "supported_sentences": 0,
            "total_sentences": 0,
            "grounding_score": 0.5,
            "unsupported_claims": [],
            "grounding_time_ms": elapsed,
        }


def _parse_grounding_response(raw: str) -> Dict[str, Any]:
    """Parse the JSON grounding response from the LLM."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

    try:
        data = json.loads(cleaned)
        is_grounded = bool(data.get("is_grounded", False))
        supported = int(data.get("supported_sentences", 0))
        total = int(data.get("total_sentences", 1))
        score = float(data.get("grounding_score", 0.0))
        unsupported = list(data.get("unsupported_claims", []))

        # Clamp score
        score = max(0.0, min(1.0, score))

        return {
            "is_grounded": is_grounded,
            "supported_sentences": supported,
            "total_sentences": total,
            "grounding_score": score,
            "unsupported_claims": unsupported,
        }

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse grounding response: {e}. Raw: {raw[:200]}")
        return {
            "is_grounded": True,
            "supported_sentences": 0,
            "total_sentences": 0,
            "grounding_score": 0.5,
            "unsupported_claims": [],
        }

"""
Confidence Scoring: margin-based + coverage + LLM self-assessment.

Three signals weighted per query type:
  - Retrieval margin (35%): score_1 - score_K separation
  - Coverage signal (35%): grounding_score from fact-checking
  - LLM self-assessment (30%): model's own confidence rating

Query-type-specific thresholds determine behavior:
  >= high:     normal answer
  low-high:    answer + disclaimer
  < low:       "I don't know" + closest matches
"""

import logging
from typing import Any, Dict, List, Optional

from backend.config import CONFIDENCE_THRESHOLDS, CONFIDENCE_WEIGHTS

logger = logging.getLogger(__name__)

DISCLAIMER = "Note: This answer has moderate confidence. Please verify with clinical records."
REFUSAL = "I don't have enough information in the clinical records to answer this question confidently."


def compute_confidence(
    query_type: str,
    retrieved_docs: Optional[List[Dict]] = None,
    grounding_result: Optional[Dict] = None,
    llm_self_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute a weighted confidence score from three signals.

    Args:
        query_type: "sql", "rag", or "hybrid"
        retrieved_docs: List of docs with 'rerank_score' or 'rrf_score' or 'score'
        grounding_result: Output from check_grounding()
        llm_self_confidence: The CONFIDENCE: X.X value from the LLM

    Returns:
        {
            score: float (0.0-1.0),
            level: "high" | "medium" | "low",
            signals: {retrieval_margin, coverage, llm_self_assessment},
            disclaimer: str or None,
        }
    """
    # SQL queries don't use retrieval — confidence based on execution success
    if query_type == "sql":
        return {
            "score": llm_self_confidence,
            "level": _get_level(query_type, llm_self_confidence),
            "signals": {
                "retrieval_margin": None,
                "coverage": None,
                "llm_self_assessment": llm_self_confidence,
            },
            "disclaimer": None,
        }

    # Compute retrieval margin
    margin = _compute_retrieval_margin(retrieved_docs or [])

    # Coverage from grounding
    coverage = 0.5
    if grounding_result:
        coverage = grounding_result.get("grounding_score", 0.5)

    # Weighted combination
    w = CONFIDENCE_WEIGHTS
    score = (
        w["retrieval_margin"] * margin
        + w["coverage"] * coverage
        + w["llm_self_assessment"] * llm_self_confidence
    )
    score = max(0.0, min(1.0, score))

    level = _get_level(query_type, score)
    disclaimer = DISCLAIMER if level == "medium" else None

    return {
        "score": round(score, 3),
        "level": level,
        "signals": {
            "retrieval_margin": round(margin, 3),
            "coverage": round(coverage, 3),
            "llm_self_assessment": round(llm_self_confidence, 3),
        },
        "disclaimer": disclaimer,
    }


def should_refuse(query_type: str, confidence_score: float, grounding_result: Optional[Dict] = None) -> bool:
    """Check if the system should refuse to answer (low confidence + grounding failure)."""
    thresholds = CONFIDENCE_THRESHOLDS.get(query_type, CONFIDENCE_THRESHOLDS["rag"])
    if confidence_score < thresholds["low"]:
        if grounding_result and not grounding_result.get("is_grounded", True):
            return True
        if confidence_score < thresholds["low"] * 0.5:
            return True
    return False


def _compute_retrieval_margin(docs: List[Dict]) -> float:
    """
    Compute retrieval margin: score_1 - score_K.

    High margin = clear signal (one doc stands out).
    Low margin = ambiguous retrieval (all docs equally relevant/irrelevant).
    """
    if not docs:
        return 0.0

    scores = []
    for doc in docs:
        s = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))
        scores.append(float(s))

    if not scores:
        return 0.0

    scores.sort(reverse=True)

    if len(scores) == 1:
        return scores[0]

    top_score = scores[0]
    bottom_score = scores[-1]

    # Normalize margin to 0-1 range
    margin = top_score - bottom_score
    return max(0.0, min(1.0, margin))


def _get_level(query_type: str, score: float) -> str:
    """Map confidence score to level using query-type-specific thresholds."""
    thresholds = CONFIDENCE_THRESHOLDS.get(query_type, CONFIDENCE_THRESHOLDS["rag"])
    if score >= thresholds["high"]:
        return "high"
    elif score >= thresholds["low"]:
        return "medium"
    else:
        return "low"

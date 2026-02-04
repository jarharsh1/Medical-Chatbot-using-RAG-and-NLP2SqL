"""
Source Attribution: format source citations with canonical doc_ids.

Ensures citations are traceable: doc_id -> ChromaDB -> SQLite -> original record.
Separates cited vs uncited sources in the response.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def format_sources(
    sources: List[Dict],
    cited_doc_ids: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Format source citations for API response.

    Args:
        sources: Raw source dicts from RAG agent.
        cited_doc_ids: List of doc_ids that were actually cited in the answer.

    Returns:
        List of formatted source dicts sorted by relevance, cited first.
    """
    if not sources:
        return []

    cited_set = set(cited_doc_ids or [])

    formatted = []
    for s in sources:
        doc_id = s.get("doc_id", "")
        formatted.append({
            "doc_id": doc_id,
            "note_id": s.get("note_id"),
            "patient_name": s.get("patient_name", ""),
            "condition": s.get("condition", ""),
            "visit_date": s.get("visit_date", ""),
            "relevance_score": s.get("relevance_score", 0.0),
            "text_snippet": s.get("text_snippet", ""),
            "cited": doc_id in cited_set if cited_set else s.get("cited", False),
        })

    # Sort: cited first, then by relevance score descending
    formatted.sort(key=lambda x: (-int(x["cited"]), -x["relevance_score"]))

    return formatted


def build_attribution_summary(sources: List[Dict]) -> str:
    """
    Build a human-readable attribution summary for the response.

    Example:
        "Sources: [Note note:1042] (John Smith, Hypertension), [Note note:1205] (Jane Doe, Diabetes)"
    """
    cited = [s for s in sources if s.get("cited")]
    if not cited:
        return ""

    parts = []
    for s in cited:
        doc_id = s.get("doc_id", "unknown")
        patient = s.get("patient_name", "Unknown")
        condition = s.get("condition", "")
        label = f"[Note {doc_id}] ({patient}"
        if condition:
            label += f", {condition}"
        label += ")"
        parts.append(label)

    return "Sources: " + ", ".join(parts)

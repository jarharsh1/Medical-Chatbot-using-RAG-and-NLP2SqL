"""
Context Relevance Metric

Measures how relevant each retrieved document is to the query.
Uses LLM to score relevance (0-1) for each document.

This is more accurate than embedding similarity because it understands
semantic nuances and query intent.
"""

import json
import logging
import re
from typing import Dict, List, Optional
from statistics import mean, stdev

from langchain_ollama import ChatOllama
from backend.config import LLM_MODEL

logger = logging.getLogger(__name__)


RELEVANCE_PROMPT = """You are a relevance scoring system. Rate how relevant the document is to answering the query.

Query: {query}

Document:
{document}

Rate the relevance from 0.0 to 1.0:
- 1.0 = Directly answers the query with specific information
- 0.7-0.9 = Highly relevant, contains useful related information
- 0.4-0.6 = Somewhat relevant, tangentially related
- 0.1-0.3 = Minimally relevant, only vague connection
- 0.0 = Completely irrelevant

Return ONLY a JSON object:
{{"score": 0.X, "reason": "brief explanation"}}"""


def score_single_document(
    query: str,
    document: str,
    llm: Optional[ChatOllama] = None,
) -> Dict:
    """
    Score a single document's relevance to the query.

    Args:
        query: User's question
        document: Document text (truncated if needed)
        llm: Optional LLM instance (creates new if not provided)

    Returns:
        {"score": 0.85, "reason": "..."}
    """
    if llm is None:
        llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # Truncate document to avoid context overflow
    doc_truncated = document[:1000] if len(document) > 1000 else document

    prompt = RELEVANCE_PROMPT.format(
        query=query,
        document=doc_truncated,
    )

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        # Parse JSON response
        # Handle cases where LLM wraps in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content)
        score = float(result.get("score", 0.5))
        reason = result.get("reason", "")

        # Clamp score to valid range
        score = max(0.0, min(1.0, score))

        return {"score": score, "reason": reason}

    except json.JSONDecodeError:
        # Try to extract score from text
        match = re.search(r'(\d+\.?\d*)', content)
        if match:
            score = float(match.group(1))
            if score > 1:
                score = score / 10  # Handle 8.5 -> 0.85
            return {"score": min(1.0, score), "reason": "parsed from text"}

        logger.warning(f"Failed to parse relevance score: {content[:100]}")
        return {"score": 0.5, "reason": "parse_error"}

    except Exception as e:
        logger.error(f"Relevance scoring error: {e}")
        return {"score": 0.5, "reason": f"error: {str(e)[:50]}"}


def compute_context_relevance(
    query: str,
    documents: List[Dict],
    max_docs: int = 5,
    doc_text_key: str = "text",
) -> Dict:
    """
    Compute relevance scores for retrieved documents.

    Args:
        query: User's question
        documents: List of retrieved documents
        max_docs: Max documents to score (LLM calls are expensive)
        doc_text_key: Key for document text in dict

    Returns:
        {
            "average_score": 0.75,
            "min_score": 0.45,
            "max_score": 0.95,
            "std_dev": 0.18,
            "scores": [0.95, 0.85, 0.72, 0.45],
            "score_reasons": ["directly answers...", ...],
            "interpretation": "high",
            "docs_scored": 4
        }
    """
    if not documents:
        return {
            "average_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "std_dev": 0.0,
            "scores": [],
            "score_reasons": [],
            "interpretation": "no_docs",
            "docs_scored": 0,
        }

    # Limit docs to score
    docs_to_score = documents[:max_docs]

    # Create LLM instance once
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    scores = []
    reasons = []

    for doc in docs_to_score:
        text = doc.get(doc_text_key, doc.get("content", ""))
        result = score_single_document(query, text, llm)
        scores.append(result["score"])
        reasons.append(result["reason"])

    # Compute statistics
    avg_score = mean(scores)
    min_score = min(scores)
    max_score = max(scores)
    std = stdev(scores) if len(scores) > 1 else 0.0

    # Interpretation
    if avg_score >= 0.7:
        interpretation = "high"
    elif avg_score >= 0.5:
        interpretation = "moderate"
    elif avg_score >= 0.3:
        interpretation = "low"
    else:
        interpretation = "very_low"

    return {
        "average_score": round(avg_score, 3),
        "min_score": round(min_score, 3),
        "max_score": round(max_score, 3),
        "std_dev": round(std, 3),
        "scores": [round(s, 3) for s in scores],
        "score_reasons": reasons,
        "interpretation": interpretation,
        "docs_scored": len(docs_to_score),
    }


def compute_context_relevance_fast(
    query: str,
    documents: List[Dict],
    rerank_scores: Optional[List[float]] = None,
) -> Dict:
    """
    Fast context relevance using existing reranker scores (no new LLM calls).

    Use this for production when you need speed over accuracy.

    Args:
        query: User's question
        documents: List of retrieved documents
        rerank_scores: Pre-computed reranker scores

    Returns:
        Same format as compute_context_relevance
    """
    if not documents:
        return {
            "average_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "scores": [],
            "interpretation": "no_docs",
            "docs_scored": 0,
            "method": "rerank_scores",
        }

    # Use reranker scores or extract from docs
    if rerank_scores:
        scores = rerank_scores
    else:
        scores = [
            doc.get("rerank_score", doc.get("score", 0.5))
            for doc in documents
        ]

    avg_score = mean(scores) if scores else 0.0
    min_score = min(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    if avg_score >= 0.7:
        interpretation = "high"
    elif avg_score >= 0.5:
        interpretation = "moderate"
    elif avg_score >= 0.3:
        interpretation = "low"
    else:
        interpretation = "very_low"

    return {
        "average_score": round(avg_score, 3),
        "min_score": round(min_score, 3),
        "max_score": round(max_score, 3),
        "scores": [round(s, 3) for s in scores],
        "interpretation": interpretation,
        "docs_scored": len(documents),
        "method": "rerank_scores",
    }

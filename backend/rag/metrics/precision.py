"""
Retrieval Precision Metrics

Standard IR metrics for measuring retrieval quality:
- Precision@K: Relevant docs in top K results
- MRR (Mean Reciprocal Rank): Position of first relevant doc
- NDCG: Normalized Discounted Cumulative Gain

Uses reranker scores as relevance proxy (no additional LLM calls).
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RankedDocument:
    """A retrieved document with its relevance score."""
    doc_id: str
    score: float  # Relevance score from reranker (0-1)
    text: Optional[str] = None


def compute_precision_at_k(
    documents: List[RankedDocument],
    k_values: List[int] = [3, 5, 10],
    relevance_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Precision@K for multiple K values.

    Args:
        documents: Ranked list of documents with scores
        k_values: List of K values to compute precision for
        relevance_threshold: Score threshold for "relevant" (default 0.5)

    Returns:
        {
            "P@3": 0.67,
            "P@5": 0.60,
            "P@10": 0.50,
            "relevant_count": 5,
            "total_count": 10
        }
    """
    results = {}

    # Count total relevant docs
    relevant_docs = [d for d in documents if d.score >= relevance_threshold]
    results["relevant_count"] = len(relevant_docs)
    results["total_count"] = len(documents)

    for k in k_values:
        top_k = documents[:k]
        relevant_in_top_k = sum(1 for d in top_k if d.score >= relevance_threshold)
        precision = relevant_in_top_k / k if k > 0 else 0.0
        results[f"P@{k}"] = round(precision, 3)

    return results


def compute_recall_at_k(
    documents: List[RankedDocument],
    k_values: List[int] = [3, 5, 10],
    relevance_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Recall@K for multiple K values.

    Recall@K = relevant docs in top K / total relevant docs

    Args:
        documents: Ranked list of documents with scores
        k_values: List of K values
        relevance_threshold: Score threshold for "relevant"

    Returns:
        {"R@3": 0.40, "R@5": 0.60, "R@10": 0.80}
    """
    total_relevant = sum(1 for d in documents if d.score >= relevance_threshold)

    if total_relevant == 0:
        return {f"R@{k}": 0.0 for k in k_values}

    results = {}
    for k in k_values:
        top_k = documents[:k]
        relevant_in_top_k = sum(1 for d in top_k if d.score >= relevance_threshold)
        recall = relevant_in_top_k / total_relevant
        results[f"R@{k}"] = round(recall, 3)

    return results


def compute_mrr(
    documents: List[RankedDocument],
    relevance_threshold: float = 0.5,
) -> float:
    """
    Compute Mean Reciprocal Rank.

    MRR = 1 / rank of first relevant document

    Args:
        documents: Ranked list of documents with scores
        relevance_threshold: Score threshold for "relevant"

    Returns:
        MRR score (0-1), 0 if no relevant docs found
    """
    for i, doc in enumerate(documents):
        if doc.score >= relevance_threshold:
            return round(1.0 / (i + 1), 3)
    return 0.0


def compute_ndcg_at_k(
    documents: List[RankedDocument],
    k: int = 10,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain @ K.

    NDCG accounts for the position of relevant docs (higher is better).

    Args:
        documents: Ranked list of documents with scores
        k: Cutoff position

    Returns:
        NDCG@K score (0-1)
    """
    top_k = documents[:k]

    if not top_k:
        return 0.0

    # DCG: sum of (relevance / log2(position + 1))
    dcg = sum(
        doc.score / math.log2(i + 2)  # +2 because log2(1) = 0
        for i, doc in enumerate(top_k)
    )

    # IDCG: DCG with perfect ranking (sorted by score desc)
    ideal_order = sorted(top_k, key=lambda d: d.score, reverse=True)
    idcg = sum(
        doc.score / math.log2(i + 2)
        for i, doc in enumerate(ideal_order)
    )

    if idcg == 0:
        return 0.0

    return round(dcg / idcg, 3)


def compute_all_precision_metrics(
    documents: List[RankedDocument],
    k_values: List[int] = [3, 5, 10],
    relevance_threshold: float = 0.5,
) -> Dict:
    """
    Compute all precision-related metrics in one call.

    Returns:
        {
            "precision": {"P@3": 0.67, "P@5": 0.60, "P@10": 0.50},
            "recall": {"R@3": 0.40, "R@5": 0.60, "R@10": 0.80},
            "mrr": 0.92,
            "ndcg@10": 0.85,
            "relevant_count": 5,
            "total_count": 10,
            "avg_score": 0.65,
            "score_distribution": {
                "high": 3,    # score >= 0.7
                "medium": 4,  # 0.4 <= score < 0.7
                "low": 3      # score < 0.4
            }
        }
    """
    if not documents:
        return {
            "precision": {},
            "recall": {},
            "mrr": 0.0,
            "ndcg@10": 0.0,
            "relevant_count": 0,
            "total_count": 0,
            "avg_score": 0.0,
            "score_distribution": {"high": 0, "medium": 0, "low": 0},
        }

    precision = compute_precision_at_k(documents, k_values, relevance_threshold)
    recall = compute_recall_at_k(documents, k_values, relevance_threshold)
    mrr = compute_mrr(documents, relevance_threshold)
    ndcg = compute_ndcg_at_k(documents, k=10)

    # Score distribution
    scores = [d.score for d in documents]
    distribution = {
        "high": sum(1 for s in scores if s >= 0.7),
        "medium": sum(1 for s in scores if 0.4 <= s < 0.7),
        "low": sum(1 for s in scores if s < 0.4),
    }

    return {
        "precision": {k: v for k, v in precision.items() if k.startswith("P@")},
        "recall": recall,
        "mrr": mrr,
        "ndcg@10": ndcg,
        "relevant_count": precision["relevant_count"],
        "total_count": precision["total_count"],
        "avg_score": round(sum(scores) / len(scores), 3),
        "score_distribution": distribution,
    }


def docs_to_ranked_documents(
    docs: List[Dict],
    score_key: str = "rerank_score",
) -> List[RankedDocument]:
    """
    Convert internal doc format to RankedDocument list.

    Args:
        docs: List of document dicts from retriever
        score_key: Key to use for relevance score

    Returns:
        List of RankedDocument objects
    """
    return [
        RankedDocument(
            doc_id=d.get("doc_id", d.get("id", "")),
            score=d.get(score_key, d.get("score", 0.5)),
            text=d.get("text", d.get("content", ""))[:200],
        )
        for d in docs
    ]

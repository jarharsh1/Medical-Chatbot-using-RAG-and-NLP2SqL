"""
RAG Metrics Module

Comprehensive metrics for measuring RAG pipeline quality:
1. Context Relevance - Are retrieved docs relevant to query?
2. Context Utilization - How much context is used in answer?
3. Answer Faithfulness - Is answer grounded in context?
4. Precision@K - Relevant docs in top K results
5. Latency Breakdown - Time per pipeline stage

Usage:
    from backend.rag.metrics import (
        LatencyTracker,
        compute_context_relevance,
        compute_context_utilization,
        compute_faithfulness,
        compute_all_precision_metrics,
        record_rag_metrics,
        get_metrics_collector,
    )

    # Track latency
    tracker = LatencyTracker()
    with tracker.track("bm25"):
        results = bm25_search(query)

    # Compute metrics after generation
    utilization = compute_context_utilization(context, answer)
    faithfulness = compute_faithfulness(answer, context)

    # Record for aggregation
    record_rag_metrics(
        run_id=run_id,
        query=question,
        context_utilization=utilization["score"],
        answer_faithfulness=faithfulness["score"],
        latency_breakdown=tracker.get_breakdown(),
    )

    # Get aggregated stats
    summary = get_metrics_collector().get_summary()
"""

# Latency tracking
from backend.rag.metrics.latency import (
    LatencyTracker,
    time_function,
)

# Context utilization (no LLM)
from backend.rag.metrics.context_utilization import (
    compute_context_utilization,
    compute_citation_coverage,
)

# Precision metrics (no LLM)
from backend.rag.metrics.precision import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_all_precision_metrics,
    RankedDocument,
    docs_to_ranked_documents,
)

# Context relevance (LLM-based)
from backend.rag.metrics.context_relevance import (
    compute_context_relevance,
    compute_context_relevance_fast,
    score_single_document,
)

# Faithfulness (LLM-based)
from backend.rag.metrics.faithfulness import (
    compute_faithfulness,
    compute_faithfulness_fast,
    extract_claims,
    verify_claim,
)

# Collector and aggregation
from backend.rag.metrics.collector import (
    RAGMetrics,
    RAGMetricsCollector,
    get_metrics_collector,
    record_rag_metrics,
)

__all__ = [
    # Latency
    "LatencyTracker",
    "time_function",
    # Utilization
    "compute_context_utilization",
    "compute_citation_coverage",
    # Precision
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_all_precision_metrics",
    "RankedDocument",
    "docs_to_ranked_documents",
    # Relevance
    "compute_context_relevance",
    "compute_context_relevance_fast",
    "score_single_document",
    # Faithfulness
    "compute_faithfulness",
    "compute_faithfulness_fast",
    "extract_claims",
    "verify_claim",
    # Collector
    "RAGMetrics",
    "RAGMetricsCollector",
    "get_metrics_collector",
    "record_rag_metrics",
]

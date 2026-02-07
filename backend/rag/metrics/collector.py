"""
RAG Metrics Collector

Central class for collecting, aggregating, and reporting RAG pipeline metrics.
Maintains a sliding window of recent requests for statistical analysis.
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import deque
from statistics import mean, median, stdev
import threading

logger = logging.getLogger(__name__)


@dataclass
class RAGMetrics:
    """Complete metrics for a single RAG request."""
    # Identifiers
    run_id: str
    query: str
    timestamp: float = field(default_factory=time.time)

    # Core metrics
    context_relevance: float = 0.0
    context_utilization: float = 0.0
    answer_faithfulness: float = 0.0

    # Precision metrics
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0

    # Latency breakdown (ms)
    latency_total: float = 0.0
    latency_bm25: float = 0.0
    latency_semantic: float = 0.0
    latency_rrf: float = 0.0
    latency_rerank: float = 0.0
    latency_mmr: float = 0.0
    latency_generate: float = 0.0

    # Counts
    docs_retrieved: int = 0
    docs_after_rerank: int = 0
    docs_after_mmr: int = 0

    # Query metadata
    query_type: str = ""  # sql, rag, hybrid
    from_cache: bool = False

    # Detailed breakdowns (optional)
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RAGMetricsCollector:
    """
    Collects and aggregates RAG metrics across requests.

    Thread-safe singleton with sliding window for recent metrics.
    """

    _instance: Optional['RAGMetricsCollector'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, window_size: int = 100):
        # Only initialize once
        if hasattr(self, '_initialized'):
            return

        self._metrics: deque[RAGMetrics] = deque(maxlen=window_size)
        self._window_size = window_size
        self._total_requests = 0
        self._lock = threading.Lock()
        self._initialized = True

    def record(self, metrics: RAGMetrics) -> None:
        """Record a new metrics entry."""
        with self._lock:
            self._metrics.append(metrics)
            self._total_requests += 1
            logger.debug(f"Recorded metrics for run_id={metrics.run_id}")

    def get_recent(self, n: int = 10) -> List[RAGMetrics]:
        """Get n most recent metrics entries."""
        with self._lock:
            return list(self._metrics)[-n:]

    def get_by_run_id(self, run_id: str) -> Optional[RAGMetrics]:
        """Get metrics for a specific run."""
        with self._lock:
            for m in reversed(self._metrics):
                if m.run_id == run_id:
                    return m
        return None

    def get_summary(self) -> Dict:
        """
        Get aggregated statistics over the sliding window.

        Returns comprehensive summary including averages, percentiles,
        and distributions for all metrics.
        """
        with self._lock:
            if not self._metrics:
                return self._empty_summary()

            metrics_list = list(self._metrics)

        # Helper for safe statistics
        def safe_stats(values: List[float]) -> Dict:
            if not values:
                return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
            sorted_vals = sorted(values)
            return {
                "avg": round(mean(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "p50": round(sorted_vals[len(sorted_vals) // 2], 3),
                "p95": round(sorted_vals[int(len(sorted_vals) * 0.95)], 3),
            }

        # Extract metric arrays
        relevance = [m.context_relevance for m in metrics_list if m.context_relevance > 0]
        utilization = [m.context_utilization for m in metrics_list if m.context_utilization > 0]
        faithfulness = [m.answer_faithfulness for m in metrics_list if m.answer_faithfulness > 0]
        latencies = [m.latency_total for m in metrics_list if m.latency_total > 0]

        # Precision metrics
        p_at_3 = [m.precision_at_3 for m in metrics_list if m.precision_at_3 > 0]
        p_at_5 = [m.precision_at_5 for m in metrics_list if m.precision_at_5 > 0]
        mrr_vals = [m.mrr for m in metrics_list if m.mrr > 0]

        # Latency breakdown
        latency_breakdown = {
            "bm25": safe_stats([m.latency_bm25 for m in metrics_list if m.latency_bm25 > 0]),
            "semantic": safe_stats([m.latency_semantic for m in metrics_list if m.latency_semantic > 0]),
            "rrf": safe_stats([m.latency_rrf for m in metrics_list if m.latency_rrf > 0]),
            "rerank": safe_stats([m.latency_rerank for m in metrics_list if m.latency_rerank > 0]),
            "mmr": safe_stats([m.latency_mmr for m in metrics_list if m.latency_mmr > 0]),
            "generate": safe_stats([m.latency_generate for m in metrics_list if m.latency_generate > 0]),
        }

        # Find bottleneck
        avg_by_stage = {k: v["avg"] for k, v in latency_breakdown.items()}
        bottleneck = max(avg_by_stage.keys(), key=lambda k: avg_by_stage[k]) if avg_by_stage else None

        # Query type distribution
        type_counts = {}
        for m in metrics_list:
            type_counts[m.query_type] = type_counts.get(m.query_type, 0) + 1

        # Cache hit rate
        cache_hits = sum(1 for m in metrics_list if m.from_cache)
        cache_rate = cache_hits / len(metrics_list) if metrics_list else 0

        return {
            "window_size": len(metrics_list),
            "total_requests": self._total_requests,
            "timestamp": time.time(),

            "quality_metrics": {
                "context_relevance": safe_stats(relevance),
                "context_utilization": safe_stats(utilization),
                "answer_faithfulness": safe_stats(faithfulness),
            },

            "precision_metrics": {
                "P@3": safe_stats(p_at_3),
                "P@5": safe_stats(p_at_5),
                "MRR": safe_stats(mrr_vals),
            },

            "latency": {
                "total": safe_stats(latencies),
                "breakdown": latency_breakdown,
                "bottleneck": bottleneck,
            },

            "query_types": type_counts,
            "cache_hit_rate": round(cache_rate, 3),

            "health": self._compute_health(
                relevance, utilization, faithfulness, latencies
            ),
        }

    def _compute_health(
        self,
        relevance: List[float],
        utilization: List[float],
        faithfulness: List[float],
        latencies: List[float],
    ) -> Dict:
        """Compute overall health status based on metrics."""
        issues = []

        # Check relevance
        if relevance and mean(relevance) < 0.5:
            issues.append("low_context_relevance")

        # Check faithfulness
        if faithfulness and mean(faithfulness) < 0.7:
            issues.append("potential_hallucinations")

        # Check latency
        if latencies and mean(latencies) > 5000:
            issues.append("high_latency")

        # Determine status
        if not issues:
            status = "healthy"
        elif len(issues) <= 1:
            status = "degraded"
        else:
            status = "unhealthy"

        return {
            "status": status,
            "issues": issues,
        }

    def _empty_summary(self) -> Dict:
        """Return empty summary when no metrics recorded."""
        return {
            "window_size": 0,
            "total_requests": 0,
            "timestamp": time.time(),
            "quality_metrics": {},
            "precision_metrics": {},
            "latency": {},
            "query_types": {},
            "cache_hit_rate": 0,
            "health": {"status": "no_data", "issues": []},
        }

    def clear(self) -> None:
        """Clear all metrics (useful for testing)."""
        with self._lock:
            self._metrics.clear()
            self._total_requests = 0


# Singleton accessor
_collector: Optional[RAGMetricsCollector] = None


def get_metrics_collector() -> RAGMetricsCollector:
    """Get or create the singleton metrics collector."""
    global _collector
    if _collector is None:
        _collector = RAGMetricsCollector()
    return _collector


def record_rag_metrics(
    run_id: str,
    query: str,
    query_type: str = "rag",
    context_relevance: float = 0.0,
    context_utilization: float = 0.0,
    answer_faithfulness: float = 0.0,
    precision_at_k: Optional[Dict] = None,
    latency_breakdown: Optional[Dict] = None,
    docs_retrieved: int = 0,
    docs_after_mmr: int = 0,
    from_cache: bool = False,
    details: Optional[Dict] = None,
) -> RAGMetrics:
    """
    Convenience function to record RAG metrics.

    Usage:
        record_rag_metrics(
            run_id=run_id,
            query=question,
            context_relevance=0.85,
            latency_breakdown=tracker.get_breakdown(),
        )
    """
    precision = precision_at_k or {}
    latency = latency_breakdown or {}

    metrics = RAGMetrics(
        run_id=run_id,
        query=query[:200],  # Truncate for storage
        query_type=query_type,
        context_relevance=context_relevance,
        context_utilization=context_utilization,
        answer_faithfulness=answer_faithfulness,
        precision_at_3=precision.get("P@3", 0.0),
        precision_at_5=precision.get("P@5", 0.0),
        precision_at_10=precision.get("P@10", 0.0),
        mrr=precision.get("mrr", precision.get("MRR", 0.0)),
        ndcg=precision.get("ndcg@10", 0.0),
        latency_total=latency.get("total_ms", 0.0),
        latency_bm25=latency.get("stages", {}).get("bm25", 0.0),
        latency_semantic=latency.get("stages", {}).get("semantic", 0.0),
        latency_rrf=latency.get("stages", {}).get("rrf", 0.0),
        latency_rerank=latency.get("stages", {}).get("rerank", 0.0),
        latency_mmr=latency.get("stages", {}).get("mmr", 0.0),
        latency_generate=latency.get("stages", {}).get("generate", 0.0),
        docs_retrieved=docs_retrieved,
        docs_after_mmr=docs_after_mmr,
        from_cache=from_cache,
        details=details or {},
    )

    get_metrics_collector().record(metrics)
    return metrics

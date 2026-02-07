"""
Latency Breakdown Tracking

Tracks time spent in each RAG pipeline stage:
- BM25 retrieval
- Semantic retrieval
- RRF fusion
- LLM reranking
- MMR diversity filter
- Answer generation

Usage:
    tracker = LatencyTracker()

    tracker.start("bm25")
    results = bm25_search(query)
    tracker.end("bm25")

    breakdown = tracker.get_breakdown()
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager


@dataclass
class StageLatency:
    """Latency info for a single stage."""
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0


class LatencyTracker:
    """
    Context-aware latency tracker for RAG pipeline stages.

    Supports both explicit start/end and context manager usage.
    """

    # Standard RAG pipeline stages
    STAGES = [
        "bm25",
        "semantic",
        "embedding",
        "rrf",
        "rerank",
        "mmr",
        "generate",
        "grounding",
        "total",
    ]

    def __init__(self):
        self._stages: Dict[str, StageLatency] = {}
        self._active_stage: Optional[str] = None
        self._total_start: Optional[float] = None

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        now = time.perf_counter()

        if self._total_start is None:
            self._total_start = now

        self._stages[stage] = StageLatency(
            name=stage,
            start_time=now,
        )
        self._active_stage = stage

    def end(self, stage: Optional[str] = None) -> float:
        """
        End timing a stage. Returns duration in ms.

        If stage is None, ends the currently active stage.
        """
        now = time.perf_counter()
        stage = stage or self._active_stage

        if stage and stage in self._stages:
            self._stages[stage].end_time = now
            self._stages[stage].duration_ms = (now - self._stages[stage].start_time) * 1000

            if self._active_stage == stage:
                self._active_stage = None

            return self._stages[stage].duration_ms

        return 0.0

    @contextmanager
    def track(self, stage: str):
        """
        Context manager for tracking a stage.

        Usage:
            with tracker.track("bm25"):
                results = bm25_search(query)
        """
        self.start(stage)
        try:
            yield
        finally:
            self.end(stage)

    def get_stage_duration(self, stage: str) -> float:
        """Get duration of a specific stage in ms."""
        if stage in self._stages:
            return self._stages[stage].duration_ms
        return 0.0

    def get_breakdown(self) -> Dict:
        """
        Get complete latency breakdown.

        Returns:
            {
                "total_ms": 1250.5,
                "stages": {
                    "bm25": 45.2,
                    "semantic": 120.8,
                    ...
                },
                "percentages": {
                    "bm25": 3.6,
                    "semantic": 9.7,
                    ...
                },
                "bottleneck": "rerank"
            }
        """
        # Calculate total
        total_ms = sum(s.duration_ms for s in self._stages.values())

        # Build stages dict
        stages = {name: s.duration_ms for name, s in self._stages.items()}

        # Calculate percentages
        percentages = {}
        if total_ms > 0:
            percentages = {
                name: round((ms / total_ms) * 100, 1)
                for name, ms in stages.items()
            }

        # Find bottleneck (slowest stage)
        bottleneck = max(stages.keys(), key=lambda k: stages[k]) if stages else None

        return {
            "total_ms": round(total_ms, 2),
            "stages": {k: round(v, 2) for k, v in stages.items()},
            "percentages": percentages,
            "bottleneck": bottleneck,
            "bottleneck_ms": round(stages.get(bottleneck, 0), 2) if bottleneck else 0,
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        breakdown = self.get_breakdown()
        lines = [f"Total: {breakdown['total_ms']:.1f}ms"]

        for stage, ms in sorted(breakdown['stages'].items(), key=lambda x: -x[1]):
            pct = breakdown['percentages'].get(stage, 0)
            lines.append(f"  {stage}: {ms:.1f}ms ({pct:.1f}%)")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all tracking data."""
        self._stages.clear()
        self._active_stage = None
        self._total_start = None


# Convenience function for one-off timing
def time_function(func, *args, **kwargs):
    """
    Time a function call and return (result, duration_ms).

    Usage:
        result, duration = time_function(expensive_operation, arg1, arg2)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms

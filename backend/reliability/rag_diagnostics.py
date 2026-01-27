"""
RAG Diagnostics Dashboard - LinkedIn Post Q1 Solution

Q: "How do you debug RAG systems?"
A: Comprehensive tracing and diagnostic tools

Diagnostic Capabilities:
1. Query Trace Viewer (full pipeline execution trace)
2. Chunk Attribution (which chunks contributed to response)
3. Context Utilization (how much retrieved context was used)
4. Retrieval Quality Metrics (relevance, coverage, redundancy)
5. Failure Analysis (identify why RAG failed)

Critical for debugging: Visibility into every step of RAG pipeline
"""

import time
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class RAGStage(Enum):
    """Stages in RAG pipeline"""
    QUERY_PREPROCESSING = "query_preprocessing"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION = "generation"
    POST_PROCESSING = "post_processing"


class TraceLevel(Enum):
    """Trace detail levels"""
    MINIMAL = "minimal"      # Just success/failure
    STANDARD = "standard"    # Key metrics only
    DETAILED = "detailed"    # Full details
    DEBUG = "debug"          # Everything including internals


@dataclass
class RetrievedChunk:
    """Information about a retrieved chunk"""
    chunk_id: str
    text: str
    source: str                      # Source document name
    similarity_score: float          # Vector similarity
    rank: int                        # Ranking position
    rerank_score: Optional[float] = None    # Cross-encoder score (if reranked)
    metadata: Dict = field(default_factory=dict)
    used_in_response: bool = False   # Was chunk cited in response?


@dataclass
class StageTrace:
    """Trace information for a single pipeline stage"""
    stage: RAGStage
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    input_data: Dict = field(default_factory=dict)
    output_data: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class RAGTrace:
    """Complete trace of RAG pipeline execution"""
    trace_id: str
    query: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    stages: List[StageTrace] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    final_response: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class ContextUtilization:
    """Metrics about how retrieved context was used"""
    total_chunks_retrieved: int
    chunks_cited: int                # Chunks explicitly cited
    chunks_inferred_used: int        # Chunks likely used (content overlap)
    total_tokens_retrieved: int
    tokens_in_response: int
    utilization_rate: float          # % of retrieved content used


@dataclass
class RetrievalQualityMetrics:
    """Metrics for retrieval quality"""
    avg_similarity_score: float
    min_similarity_score: float
    max_similarity_score: float
    relevance_score: float           # 0-1, how relevant were results
    coverage_score: float            # 0-1, did results cover query aspects
    redundancy_score: float          # 0-1, how redundant were results
    diversity_score: float           # 0-1, diversity of sources


@dataclass
class RAGDiagnosticReport:
    """Complete diagnostic report for a RAG execution"""
    trace: RAGTrace
    context_utilization: ContextUtilization
    retrieval_quality: RetrievalQualityMetrics
    bottlenecks: List[str]           # Identified performance bottlenecks
    recommendations: List[str]       # Recommendations for improvement
    health_score: float              # 0-1, overall health of RAG execution


class RAGTracer:
    """
    Tracer for RAG pipeline execution.

    Records detailed traces of each stage for debugging and analysis.

    Example:
        tracer = RAGTracer(trace_level=TraceLevel.DETAILED)

        with tracer.trace_query("What is diabetes?") as trace:
            # Query preprocessing
            with tracer.trace_stage(RAGStage.QUERY_PREPROCESSING) as stage:
                processed_query = preprocess(query)
                stage.output_data = {"processed": processed_query}

            # Retrieval
            with tracer.trace_stage(RAGStage.RETRIEVAL) as stage:
                chunks = retrieve(processed_query)
                tracer.record_chunks(chunks)

            # Generation
            with tracer.trace_stage(RAGStage.GENERATION) as stage:
                response = generate(chunks, query)
                tracer.record_response(response)

        report = tracer.generate_diagnostic_report(trace)
    """

    def __init__(
        self,
        trace_level: TraceLevel = TraceLevel.STANDARD,
        enable_persistence: bool = False,
        storage_path: Optional[str] = None
    ):
        """
        Initialize RAG tracer.

        Args:
            trace_level: Level of detail to capture
            enable_persistence: Save traces to disk
            storage_path: Path for trace storage
        """
        self.trace_level = trace_level
        self.enable_persistence = enable_persistence
        self.storage_path = storage_path or "./traces"

        self.current_trace: Optional[RAGTrace] = None
        self.current_stage: Optional[StageTrace] = None

        # Statistics
        self.stats = {
            "total_traces": 0,
            "successful_traces": 0,
            "failed_traces": 0,
            "avg_duration_ms": 0.0,
            "stage_performance": {stage.value: [] for stage in RAGStage}
        }

    def trace_query(self, query: str, trace_id: Optional[str] = None):
        """
        Start tracing a new RAG query.

        Returns context manager for the trace.

        Example:
            with tracer.trace_query("What is diabetes?") as trace:
                # RAG pipeline code here
                pass
        """
        return _QueryTraceContext(self, query, trace_id)

    def trace_stage(self, stage: RAGStage):
        """
        Start tracing a pipeline stage.

        Returns context manager for the stage.

        Example:
            with tracer.trace_stage(RAGStage.RETRIEVAL) as stage_trace:
                chunks = retrieve()
                stage_trace.output_data = {"chunks": chunks}
        """
        return _StageTraceContext(self, stage)

    def record_chunks(self, chunks: List[RetrievedChunk]):
        """Record retrieved chunks"""
        if self.current_trace:
            self.current_trace.retrieved_chunks.extend(chunks)

    def record_response(self, response: str):
        """Record final generated response"""
        if self.current_trace:
            self.current_trace.final_response = response

    def generate_diagnostic_report(self, trace: RAGTrace) -> RAGDiagnosticReport:
        """
        Generate comprehensive diagnostic report from trace.

        Analyzes:
        - Context utilization
        - Retrieval quality
        - Performance bottlenecks
        - Improvement recommendations

        Args:
            trace: RAG execution trace

        Returns:
            RAGDiagnosticReport with analysis and recommendations
        """
        # Calculate context utilization
        context_util = self._calculate_context_utilization(trace)

        # Calculate retrieval quality
        retrieval_quality = self._calculate_retrieval_quality(trace)

        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(trace)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            trace, context_util, retrieval_quality, bottlenecks
        )

        # Calculate health score
        health_score = self._calculate_health_score(
            trace, context_util, retrieval_quality
        )

        return RAGDiagnosticReport(
            trace=trace,
            context_utilization=context_util,
            retrieval_quality=retrieval_quality,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            health_score=health_score
        )

    def _calculate_context_utilization(self, trace: RAGTrace) -> ContextUtilization:
        """Calculate how much of retrieved context was actually used"""
        chunks = trace.retrieved_chunks
        response = trace.final_response or ""

        total_chunks = len(chunks)
        chunks_cited = sum(1 for c in chunks if c.used_in_response)

        # Estimate chunks used by text overlap
        chunks_inferred = 0
        for chunk in chunks:
            if self._text_appears_in_response(chunk.text, response):
                chunks_inferred += 1

        # Token estimates (rough)
        total_tokens_retrieved = sum(len(c.text.split()) for c in chunks)
        tokens_in_response = len(response.split())

        utilization_rate = (
            chunks_inferred / total_chunks if total_chunks > 0 else 0.0
        )

        return ContextUtilization(
            total_chunks_retrieved=total_chunks,
            chunks_cited=chunks_cited,
            chunks_inferred_used=chunks_inferred,
            total_tokens_retrieved=total_tokens_retrieved,
            tokens_in_response=tokens_in_response,
            utilization_rate=utilization_rate
        )

    def _text_appears_in_response(self, chunk_text: str, response: str) -> bool:
        """Check if chunk content appears in response (fuzzy match)"""
        # Extract key phrases from chunk (3+ word sequences)
        chunk_words = chunk_text.lower().split()
        response_lower = response.lower()

        # Check for 3-word overlaps
        for i in range(len(chunk_words) - 2):
            phrase = " ".join(chunk_words[i:i+3])
            if phrase in response_lower:
                return True

        return False

    def _calculate_retrieval_quality(self, trace: RAGTrace) -> RetrievalQualityMetrics:
        """Calculate metrics for retrieval quality"""
        chunks = trace.retrieved_chunks

        if not chunks:
            return RetrievalQualityMetrics(
                avg_similarity_score=0.0,
                min_similarity_score=0.0,
                max_similarity_score=0.0,
                relevance_score=0.0,
                coverage_score=0.0,
                redundancy_score=0.0,
                diversity_score=0.0
            )

        # Similarity scores
        scores = [c.similarity_score for c in chunks]
        avg_similarity = sum(scores) / len(scores)
        min_similarity = min(scores)
        max_similarity = max(scores)

        # Relevance: average of top-3 scores
        top_scores = sorted(scores, reverse=True)[:3]
        relevance_score = sum(top_scores) / len(top_scores)

        # Coverage: do chunks cover query comprehensively?
        # (Simplified: based on utilization)
        coverage_score = sum(1 for c in chunks if c.used_in_response) / len(chunks)

        # Redundancy: similarity between chunks
        redundancy_score = self._calculate_redundancy(chunks)

        # Diversity: variety of sources
        unique_sources = len(set(c.source for c in chunks))
        diversity_score = unique_sources / len(chunks) if chunks else 0.0

        return RetrievalQualityMetrics(
            avg_similarity_score=avg_similarity,
            min_similarity_score=min_similarity,
            max_similarity_score=max_similarity,
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            redundancy_score=redundancy_score,
            diversity_score=diversity_score
        )

    def _calculate_redundancy(self, chunks: List[RetrievedChunk]) -> float:
        """Calculate redundancy score (0=unique, 1=all similar)"""
        if len(chunks) < 2:
            return 0.0

        # Count overlapping 3-word phrases between chunks
        total_comparisons = 0
        overlap_count = 0

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                total_comparisons += 1
                if self._chunks_overlap(chunks[i].text, chunks[j].text):
                    overlap_count += 1

        return overlap_count / total_comparisons if total_comparisons > 0 else 0.0

    def _chunks_overlap(self, text1: str, text2: str) -> bool:
        """Check if two chunks have significant overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1 & words2)
        return overlap > min(len(words1), len(words2)) * 0.3

    def _identify_bottlenecks(self, trace: RAGTrace) -> List[str]:
        """Identify performance bottlenecks in pipeline"""
        bottlenecks = []

        # Analyze stage durations
        for stage_trace in trace.stages:
            if stage_trace.duration_ms > 1000:  # > 1 second
                bottlenecks.append(
                    f"{stage_trace.stage.value} took {stage_trace.duration_ms:.0f}ms (slow)"
                )

        # Check retrieval
        if len(trace.retrieved_chunks) > 20:
            bottlenecks.append(
                f"Retrieved {len(trace.retrieved_chunks)} chunks (consider reducing)"
            )

        # Check utilization
        util = self._calculate_context_utilization(trace)
        if util.utilization_rate < 0.3:
            bottlenecks.append(
                f"Low context utilization ({util.utilization_rate:.1%}) - "
                "retrieved chunks not used"
            )

        return bottlenecks

    def _generate_recommendations(
        self,
        trace: RAGTrace,
        context_util: ContextUtilization,
        retrieval_quality: RetrievalQualityMetrics,
        bottlenecks: List[str]
    ) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []

        # Low utilization
        if context_util.utilization_rate < 0.3:
            recommendations.append(
                "Improve retrieval relevance - only 30% of chunks used. "
                "Consider: (1) Better query preprocessing, (2) Hybrid search, "
                "(3) Cross-encoder reranking"
            )

        # Low relevance
        if retrieval_quality.relevance_score < 0.6:
            recommendations.append(
                f"Low relevance score ({retrieval_quality.relevance_score:.1%}). "
                "Consider: (1) Fine-tune embeddings, (2) Query expansion, "
                "(3) Better chunking strategy"
            )

        # High redundancy
        if retrieval_quality.redundancy_score > 0.5:
            recommendations.append(
                "High redundancy detected. Consider: (1) MMR (Maximal Marginal Relevance), "
                "(2) Diversity-aware retrieval"
            )

        # Low diversity
        if retrieval_quality.diversity_score < 0.3:
            recommendations.append(
                "Low source diversity. Consider retrieving from more varied sources"
            )

        # Performance issues
        if trace.total_duration_ms > 5000:
            recommendations.append(
                f"Total pipeline took {trace.total_duration_ms/1000:.1f}s. "
                "Consider: (1) Caching, (2) Async operations, (3) Smaller embedding models"
            )

        return recommendations

    def _calculate_health_score(
        self,
        trace: RAGTrace,
        context_util: ContextUtilization,
        retrieval_quality: RetrievalQualityMetrics
    ) -> float:
        """Calculate overall health score (0-1)"""
        if not trace.success:
            return 0.0

        # Component scores
        utilization_score = context_util.utilization_rate
        relevance_score = retrieval_quality.relevance_score
        diversity_score = retrieval_quality.diversity_score

        # Performance score (penalize if slow)
        performance_score = max(0, 1.0 - (trace.total_duration_ms / 10000))

        # Weighted average
        health = (
            0.3 * utilization_score +
            0.4 * relevance_score +
            0.2 * diversity_score +
            0.1 * performance_score
        )

        return health

    def get_statistics(self) -> Dict:
        """Get RAG tracing statistics"""
        return {
            **self.stats,
            "success_rate": (
                f"{self.stats['successful_traces'] / self.stats['total_traces']:.2%}"
                if self.stats['total_traces'] > 0 else "N/A"
            )
        }


class _QueryTraceContext:
    """Context manager for query tracing"""

    def __init__(self, tracer: RAGTracer, query: str, trace_id: Optional[str]):
        self.tracer = tracer
        self.query = query
        self.trace_id = trace_id or f"trace_{int(time.time() * 1000)}"
        self.trace: Optional[RAGTrace] = None

    def __enter__(self) -> RAGTrace:
        self.trace = RAGTrace(
            trace_id=self.trace_id,
            query=self.query,
            start_time=datetime.now()
        )
        self.tracer.current_trace = self.trace
        self.tracer.stats["total_traces"] += 1
        return self.trace

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace:
            self.trace.end_time = datetime.now()
            self.trace.total_duration_ms = (
                (self.trace.end_time - self.trace.start_time).total_seconds() * 1000
            )

            if exc_type:
                self.trace.success = False
                self.trace.error = str(exc_val)
                self.tracer.stats["failed_traces"] += 1
            else:
                self.tracer.stats["successful_traces"] += 1

        self.tracer.current_trace = None


class _StageTraceContext:
    """Context manager for stage tracing"""

    def __init__(self, tracer: RAGTracer, stage: RAGStage):
        self.tracer = tracer
        self.stage = stage
        self.stage_trace: Optional[StageTrace] = None

    def __enter__(self) -> StageTrace:
        self.stage_trace = StageTrace(
            stage=self.stage,
            start_time=datetime.now()
        )
        self.tracer.current_stage = self.stage_trace
        return self.stage_trace

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stage_trace:
            self.stage_trace.end_time = datetime.now()
            self.stage_trace.duration_ms = (
                (self.stage_trace.end_time - self.stage_trace.start_time).total_seconds() * 1000
            )

            if exc_type:
                self.stage_trace.success = False
                self.stage_trace.error = str(exc_val)

            # Add to current trace
            if self.tracer.current_trace:
                self.tracer.current_trace.stages.append(self.stage_trace)

            # Update statistics
            self.tracer.stats["stage_performance"][self.stage.value].append(
                self.stage_trace.duration_ms
            )

        self.tracer.current_stage = None


# Example usage
if __name__ == "__main__":
    tracer = RAGTracer(trace_level=TraceLevel.DETAILED)

    # Simulate RAG pipeline
    with tracer.trace_query("What is diabetes?") as trace:
        # Stage 1: Query preprocessing
        with tracer.trace_stage(RAGStage.QUERY_PREPROCESSING) as stage:
            time.sleep(0.01)  # Simulate work
            stage.output_data = {"processed": "diabetes"}

        # Stage 2: Retrieval
        with tracer.trace_stage(RAGStage.RETRIEVAL) as stage:
            time.sleep(0.05)  # Simulate retrieval
            chunks = [
                RetrievedChunk(
                    chunk_id="1",
                    text="Diabetes is a chronic condition affecting blood sugar.",
                    source="medical_textbook.pdf",
                    similarity_score=0.92,
                    rank=1,
                    used_in_response=True
                ),
                RetrievedChunk(
                    chunk_id="2",
                    text="Type 2 diabetes is managed with diet and medication.",
                    source="treatment_guide.pdf",
                    similarity_score=0.85,
                    rank=2,
                    used_in_response=True
                ),
                RetrievedChunk(
                    chunk_id="3",
                    text="Exercise helps control blood sugar levels.",
                    source="wellness_guide.pdf",
                    similarity_score=0.71,
                    rank=3,
                    used_in_response=False
                )
            ]
            tracer.record_chunks(chunks)

        # Stage 3: Generation
        with tracer.trace_stage(RAGStage.GENERATION) as stage:
            time.sleep(0.1)  # Simulate generation
            response = (
                "Diabetes is a chronic condition that affects how your body "
                "regulates blood sugar. Type 2 diabetes can be managed through "
                "diet, exercise, and medication."
            )
            tracer.record_response(response)

    # Generate diagnostic report
    report = tracer.generate_diagnostic_report(trace)

    print("=" * 80)
    print("RAG DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"Query: {trace.query}")
    print(f"Success: {trace.success}")
    print(f"Duration: {trace.total_duration_ms:.2f}ms")
    print(f"Health Score: {report.health_score:.2f}")

    print("\n" + "=" * 80)
    print("CONTEXT UTILIZATION")
    print("=" * 80)
    util = report.context_utilization
    print(f"Chunks Retrieved: {util.total_chunks_retrieved}")
    print(f"Chunks Used: {util.chunks_inferred_used}")
    print(f"Utilization Rate: {util.utilization_rate:.1%}")

    print("\n" + "=" * 80)
    print("RETRIEVAL QUALITY")
    print("=" * 80)
    qual = report.retrieval_quality
    print(f"Avg Similarity: {qual.avg_similarity_score:.2f}")
    print(f"Relevance Score: {qual.relevance_score:.2f}")
    print(f"Coverage Score: {qual.coverage_score:.2f}")
    print(f"Diversity Score: {qual.diversity_score:.2f}")

    print("\n" + "=" * 80)
    print("STAGE PERFORMANCE")
    print("=" * 80)
    for stage in trace.stages:
        print(f"{stage.stage.value}: {stage.duration_ms:.2f}ms")

    if report.bottlenecks:
        print("\n" + "=" * 80)
        print("BOTTLENECKS")
        print("=" * 80)
        for bottleneck in report.bottlenecks:
            print(f"⚠️  {bottleneck}")

    if report.recommendations:
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")

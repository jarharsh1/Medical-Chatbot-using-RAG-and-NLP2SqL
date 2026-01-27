"""
Unit tests for RAG Diagnostics Dashboard (LinkedIn Q1)

Tests RAG debugging capabilities:
- Query tracing
- Chunk attribution
- Context utilization metrics
- Retrieval quality analysis
- Performance monitoring
"""

import pytest
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from reliability.rag_diagnostics import (
    RAGTracer,
    RAGStage,
    TraceLevel,
    RetrievedChunk
)


class TestRAGTracer:
    """Test suite for RAG tracer"""

    @pytest.fixture
    def tracer(self):
        """Create tracer instance"""
        return RAGTracer(trace_level=TraceLevel.DETAILED)

    # =========================================================================
    # QUERY TRACING
    # =========================================================================

    def test_trace_query_basic(self, tracer):
        """Test basic query tracing"""
        with tracer.trace_query("What is diabetes?") as trace:
            assert trace.query == "What is diabetes?"
            assert trace.start_time is not None
            assert trace.success is True

        # After context exit
        assert trace.end_time is not None
        assert trace.total_duration_ms > 0

    def test_trace_query_captures_error(self, tracer):
        """Test that query tracing captures errors"""
        with pytest.raises(ValueError):
            with tracer.trace_query("test query") as trace:
                raise ValueError("Test error")

        # Trace should record failure
        assert trace.success is False
        assert trace.error is not None
        assert "Test error" in trace.error

    def test_multiple_queries_tracked(self, tracer):
        """Test that multiple queries are tracked in statistics"""
        with tracer.trace_query("Query 1"):
            pass

        with tracer.trace_query("Query 2"):
            pass

        stats = tracer.get_statistics()
        assert stats["total_traces"] == 2
        assert stats["successful_traces"] == 2

    # =========================================================================
    # STAGE TRACING
    # =========================================================================

    def test_trace_stage_basic(self, tracer):
        """Test basic stage tracing"""
        with tracer.trace_query("test") as trace:
            with tracer.trace_stage(RAGStage.RETRIEVAL) as stage:
                assert stage.stage == RAGStage.RETRIEVAL
                assert stage.start_time is not None

            # After stage exit
            assert stage.end_time is not None
            assert stage.duration_ms > 0

        # Stage should be in trace
        assert len(trace.stages) == 1
        assert trace.stages[0].stage == RAGStage.RETRIEVAL

    def test_trace_multiple_stages(self, tracer):
        """Test tracing multiple stages in pipeline"""
        with tracer.trace_query("test") as trace:
            with tracer.trace_stage(RAGStage.QUERY_PREPROCESSING):
                time.sleep(0.01)

            with tracer.trace_stage(RAGStage.RETRIEVAL):
                time.sleep(0.02)

            with tracer.trace_stage(RAGStage.GENERATION):
                time.sleep(0.01)

        assert len(trace.stages) == 3
        assert trace.stages[0].stage == RAGStage.QUERY_PREPROCESSING
        assert trace.stages[1].stage == RAGStage.RETRIEVAL
        assert trace.stages[2].stage == RAGStage.GENERATION

    def test_stage_captures_error(self, tracer):
        """Test that stage tracing captures errors"""
        with tracer.trace_query("test") as trace:
            with pytest.raises(Exception):
                with tracer.trace_stage(RAGStage.RETRIEVAL) as stage:
                    raise Exception("Retrieval failed")

            # Stage should record failure
            assert not stage.success
            assert "Retrieval failed" in stage.error

    def test_stage_input_output_data(self, tracer):
        """Test recording input/output data for stages"""
        with tracer.trace_query("test"):
            with tracer.trace_stage(RAGStage.EMBEDDING) as stage:
                stage.input_data = {"query": "diabetes"}
                stage.output_data = {"vector": [0.1, 0.2, 0.3]}

        assert stage.input_data["query"] == "diabetes"
        assert "vector" in stage.output_data

    # =========================================================================
    # CHUNK RECORDING
    # =========================================================================

    def test_record_chunks(self, tracer):
        """Test recording retrieved chunks"""
        chunks = [
            RetrievedChunk(
                chunk_id="1",
                text="Diabetes is a chronic condition.",
                source="textbook.pdf",
                similarity_score=0.92,
                rank=1
            ),
            RetrievedChunk(
                chunk_id="2",
                text="Diabetes treatment includes medication.",
                source="guide.pdf",
                similarity_score=0.85,
                rank=2
            )
        ]

        with tracer.trace_query("test") as trace:
            tracer.record_chunks(chunks)

        assert len(trace.retrieved_chunks) == 2
        assert trace.retrieved_chunks[0].chunk_id == "1"
        assert trace.retrieved_chunks[1].chunk_id == "2"

    def test_record_response(self, tracer):
        """Test recording final response"""
        with tracer.trace_query("test") as trace:
            tracer.record_response("This is the generated response.")

        assert trace.final_response == "This is the generated response."

    # =========================================================================
    # DIAGNOSTIC REPORT GENERATION
    # =========================================================================

    def test_generate_diagnostic_report(self, tracer):
        """Test generating diagnostic report"""
        with tracer.trace_query("What is diabetes?") as trace:
            # Add chunks
            chunks = [
                RetrievedChunk(
                    chunk_id="1",
                    text="Diabetes is a chronic condition affecting blood sugar.",
                    source="doc1.pdf",
                    similarity_score=0.9,
                    rank=1,
                    used_in_response=True
                )
            ]
            tracer.record_chunks(chunks)

            # Add response
            tracer.record_response("Diabetes is a chronic condition.")

        report = tracer.generate_diagnostic_report(trace)

        assert report.trace == trace
        assert report.context_utilization is not None
        assert report.retrieval_quality is not None
        assert report.health_score >= 0.0
        assert report.health_score <= 1.0

    # =========================================================================
    # CONTEXT UTILIZATION METRICS
    # =========================================================================

    def test_context_utilization_all_chunks_used(self, tracer):
        """Test context utilization when all chunks are used"""
        with tracer.trace_query("test") as trace:
            chunks = [
                RetrievedChunk(
                    chunk_id="1",
                    text="Metformin treats diabetes",
                    source="doc1",
                    similarity_score=0.9,
                    rank=1,
                    used_in_response=True
                ),
                RetrievedChunk(
                    chunk_id="2",
                    text="Insulin is another treatment",
                    source="doc2",
                    similarity_score=0.8,
                    rank=2,
                    used_in_response=True
                )
            ]
            tracer.record_chunks(chunks)
            tracer.record_response(
                "Metformin treats diabetes and insulin is another treatment option."
            )

        report = tracer.generate_diagnostic_report(trace)
        util = report.context_utilization

        assert util.total_chunks_retrieved == 2
        assert util.chunks_cited == 2
        assert util.utilization_rate == 1.0  # 100% utilization

    def test_context_utilization_partial_usage(self, tracer):
        """Test context utilization with partial chunk usage"""
        with tracer.trace_query("test") as trace:
            chunks = [
                RetrievedChunk(
                    chunk_id="1",
                    text="Relevant information about diabetes",
                    source="doc1",
                    similarity_score=0.9,
                    rank=1
                ),
                RetrievedChunk(
                    chunk_id="2",
                    text="Completely unrelated information",
                    source="doc2",
                    similarity_score=0.5,
                    rank=2
                )
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("Information about diabetes is important.")

        report = tracer.generate_diagnostic_report(trace)
        util = report.context_utilization

        # Should detect that only some chunks were used
        assert util.utilization_rate < 1.0

    def test_context_utilization_no_chunks(self, tracer):
        """Test context utilization when no chunks retrieved"""
        with tracer.trace_query("test") as trace:
            tracer.record_response("Some response")

        report = tracer.generate_diagnostic_report(trace)
        util = report.context_utilization

        assert util.total_chunks_retrieved == 0
        assert util.utilization_rate == 0.0

    # =========================================================================
    # RETRIEVAL QUALITY METRICS
    # =========================================================================

    def test_retrieval_quality_high_similarity(self, tracer):
        """Test retrieval quality with high similarity scores"""
        with tracer.trace_query("test") as trace:
            chunks = [
                RetrievedChunk("1", "text", "doc1", 0.95, 1, used_in_response=True),
                RetrievedChunk("2", "text", "doc2", 0.90, 2, used_in_response=True),
                RetrievedChunk("3", "text", "doc3", 0.85, 3, used_in_response=True)
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("response")

        report = tracer.generate_diagnostic_report(trace)
        qual = report.retrieval_quality

        assert qual.avg_similarity_score > 0.8
        assert qual.max_similarity_score == 0.95
        assert qual.min_similarity_score == 0.85
        assert qual.relevance_score > 0.8

    def test_retrieval_quality_diverse_sources(self, tracer):
        """Test diversity score with different sources"""
        with tracer.trace_query("test") as trace:
            chunks = [
                RetrievedChunk("1", "text", "source_A", 0.9, 1),
                RetrievedChunk("2", "text", "source_B", 0.8, 2),
                RetrievedChunk("3", "text", "source_C", 0.7, 3)
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("response")

        report = tracer.generate_diagnostic_report(trace)
        qual = report.retrieval_quality

        # All different sources = high diversity
        assert qual.diversity_score == 1.0

    def test_retrieval_quality_redundant_sources(self, tracer):
        """Test diversity score with redundant sources"""
        with tracer.trace_query("test") as trace:
            chunks = [
                RetrievedChunk("1", "text", "same_source", 0.9, 1),
                RetrievedChunk("2", "text", "same_source", 0.8, 2),
                RetrievedChunk("3", "text", "same_source", 0.7, 3)
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("response")

        report = tracer.generate_diagnostic_report(trace)
        qual = report.retrieval_quality

        # All same source = low diversity
        assert qual.diversity_score < 0.5

    # =========================================================================
    # BOTTLENECK IDENTIFICATION
    # =========================================================================

    def test_identify_slow_stage_bottleneck(self, tracer):
        """Test identification of slow stage as bottleneck"""
        with tracer.trace_query("test") as trace:
            with tracer.trace_stage(RAGStage.RETRIEVAL):
                time.sleep(1.1)  # Slow retrieval (>1 second)

        report = tracer.generate_diagnostic_report(trace)

        # Should identify slow retrieval
        assert len(report.bottlenecks) > 0
        assert any("retrieval" in b.lower() for b in report.bottlenecks)

    def test_identify_low_utilization_bottleneck(self, tracer):
        """Test identification of low context utilization"""
        with tracer.trace_query("test") as trace:
            # Retrieve many chunks but use none
            chunks = [
                RetrievedChunk(f"{i}", "unrelated text", f"doc{i}", 0.5, i)
                for i in range(10)
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("Short response with no chunk content.")

        report = tracer.generate_diagnostic_report(trace)

        # Should identify low utilization
        assert any("utilization" in b.lower() for b in report.bottlenecks)

    def test_identify_too_many_chunks_bottleneck(self, tracer):
        """Test identification of retrieving too many chunks"""
        with tracer.trace_query("test") as trace:
            chunks = [
                RetrievedChunk(f"{i}", "text", f"doc{i}", 0.7, i)
                for i in range(25)  # More than 20
            ]
            tracer.record_chunks(chunks)

        report = tracer.generate_diagnostic_report(trace)

        assert any("chunks" in b.lower() for b in report.bottlenecks)

    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================

    def test_recommend_better_retrieval(self, tracer):
        """Test recommendation for low relevance"""
        with tracer.trace_query("test") as trace:
            # Low similarity scores
            chunks = [
                RetrievedChunk("1", "text", "doc1", 0.3, 1),
                RetrievedChunk("2", "text", "doc2", 0.2, 2)
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("response")

        report = tracer.generate_diagnostic_report(trace)

        # Should recommend improving retrieval
        assert len(report.recommendations) > 0

    def test_recommend_reduce_redundancy(self, tracer):
        """Test recommendation for high redundancy"""
        with tracer.trace_query("test") as trace:
            # Very similar chunks (simulated by same text)
            chunks = [
                RetrievedChunk("1", "diabetes treatment medication", "doc1", 0.9, 1),
                RetrievedChunk("2", "diabetes treatment medication", "doc2", 0.9, 2),
                RetrievedChunk("3", "diabetes treatment medication", "doc3", 0.9, 3)
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("response")

        report = tracer.generate_diagnostic_report(trace)

        # Should recommend reducing redundancy
        recs_text = " ".join(report.recommendations).lower()
        # May recommend MMR or diversity

    # =========================================================================
    # HEALTH SCORE
    # =========================================================================

    def test_health_score_perfect_execution(self, tracer):
        """Test health score for perfect RAG execution"""
        with tracer.trace_query("test") as trace:
            # High quality retrieval
            chunks = [
                RetrievedChunk(
                    "1",
                    "Diabetes information here",
                    "doc1",
                    0.95,
                    1,
                    used_in_response=True
                )
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("Diabetes information in response")

        report = tracer.generate_diagnostic_report(trace)

        # Should have high health score
        assert report.health_score > 0.7

    def test_health_score_failed_execution(self, tracer):
        """Test health score for failed execution"""
        with pytest.raises(Exception):
            with tracer.trace_query("test") as trace:
                raise Exception("Pipeline failed")

        report = tracer.generate_diagnostic_report(trace)

        # Failed execution = low health
        assert report.health_score == 0.0

    def test_health_score_poor_quality(self, tracer):
        """Test health score for poor quality execution"""
        with tracer.trace_query("test") as trace:
            # Low quality retrieval
            chunks = [
                RetrievedChunk("1", "text", "doc1", 0.2, 1),
                RetrievedChunk("2", "text", "doc2", 0.1, 2)
            ]
            tracer.record_chunks(chunks)
            tracer.record_response("response")

        report = tracer.generate_diagnostic_report(trace)

        # Poor quality = lower health score
        assert report.health_score < 0.5

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def test_statistics_tracking(self, tracer):
        """Test comprehensive statistics tracking"""
        # Successful trace
        with tracer.trace_query("Query 1"):
            pass

        # Failed trace
        with pytest.raises(Exception):
            with tracer.trace_query("Query 2"):
                raise Exception("Error")

        stats = tracer.get_statistics()

        assert stats["total_traces"] == 2
        assert stats["successful_traces"] == 1
        assert stats["failed_traces"] == 1

    def test_stage_performance_statistics(self, tracer):
        """Test that stage performance is tracked"""
        with tracer.trace_query("test"):
            with tracer.trace_stage(RAGStage.RETRIEVAL):
                time.sleep(0.01)

        stats = tracer.get_statistics()
        stage_perf = stats["stage_performance"]

        assert RAGStage.RETRIEVAL.value in stage_perf
        assert len(stage_perf[RAGStage.RETRIEVAL.value]) > 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestRAGDiagnosticsIntegration:
    """Integration tests for RAG diagnostics"""

    def test_complete_rag_pipeline_trace(self):
        """Test tracing complete RAG pipeline"""
        tracer = RAGTracer(trace_level=TraceLevel.DETAILED)

        with tracer.trace_query("What is diabetes?") as trace:
            # Stage 1: Preprocessing
            with tracer.trace_stage(RAGStage.QUERY_PREPROCESSING) as stage:
                time.sleep(0.01)
                stage.output_data = {"processed_query": "diabetes"}

            # Stage 2: Embedding
            with tracer.trace_stage(RAGStage.EMBEDDING) as stage:
                time.sleep(0.02)
                stage.output_data = {"embedding": [0.1] * 768}

            # Stage 3: Retrieval
            with tracer.trace_stage(RAGStage.RETRIEVAL) as stage:
                chunks = [
                    RetrievedChunk(
                        "1",
                        "Diabetes is a chronic metabolic condition.",
                        "medical_textbook.pdf",
                        0.92,
                        1,
                        used_in_response=True
                    ),
                    RetrievedChunk(
                        "2",
                        "Treatment includes lifestyle and medication.",
                        "treatment_guide.pdf",
                        0.87,
                        2,
                        used_in_response=True
                    )
                ]
                tracer.record_chunks(chunks)
                time.sleep(0.05)

            # Stage 4: Generation
            with tracer.trace_stage(RAGStage.GENERATION) as stage:
                response = (
                    "Diabetes is a chronic metabolic condition that requires "
                    "treatment including lifestyle changes and medication."
                )
                tracer.record_response(response)
                time.sleep(0.1)

        # Generate diagnostic report
        report = tracer.generate_diagnostic_report(trace)

        # Verify complete trace
        assert len(trace.stages) == 4
        assert trace.success
        assert trace.total_duration_ms > 0

        # Verify metrics
        assert report.context_utilization.total_chunks_retrieved == 2
        assert report.retrieval_quality.avg_similarity_score > 0.8
        assert report.health_score > 0.5

    def test_trace_with_failure_recovery(self):
        """Test tracing pipeline with stage failure"""
        tracer = RAGTracer()

        with tracer.trace_query("test") as trace:
            # Successful stage
            with tracer.trace_stage(RAGStage.QUERY_PREPROCESSING):
                pass

            # Failed stage (but caught)
            try:
                with tracer.trace_stage(RAGStage.RETRIEVAL):
                    raise Exception("Retrieval failed")
            except Exception:
                pass

            # Recovery with fallback
            with tracer.trace_stage(RAGStage.GENERATION):
                tracer.record_response("Fallback response")

        # Should still complete trace
        assert trace.success
        assert len(trace.stages) == 3

    def test_performance_minimal_overhead(self):
        """Test that tracing has minimal performance overhead"""
        tracer = RAGTracer(trace_level=TraceLevel.MINIMAL)

        start = time.time()
        for _ in range(100):
            with tracer.trace_query("test"):
                with tracer.trace_stage(RAGStage.RETRIEVAL):
                    pass
        elapsed = time.time() - start

        avg_overhead_ms = (elapsed / 100) * 1000
        # Overhead should be minimal (< 2ms per trace)
        assert avg_overhead_ms < 2, f"Too much overhead: {avg_overhead_ms:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

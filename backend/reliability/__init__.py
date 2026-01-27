"""
Production Reliability Package

Components for production-grade system reliability:
- Runtime Hallucination Detection (detect when LLM makes things up)
- API Resilience (circuit breaker, retry, fallback)
- RAG Diagnostics (trace and debug RAG pipeline)

These components address LinkedIn interview questions Q1, Q4, Q5.
"""

from .hallucination_detection import (
    RuntimeHallucinationDetector,
    HallucinationDetectionResult,
    Claim,
    ClaimType,
    ConfidenceLevel
)

from .api_resilience import (
    APIResilientCaller,
    CircuitBreaker,
    APICallResult,
    CircuitState,
    CircuitBreakerConfig,
    RetryConfig,
    resilient_api_call
)

from .rag_diagnostics import (
    RAGTracer,
    RAGDiagnosticReport,
    RAGTrace,
    RetrievedChunk,
    ContextUtilization,
    RetrievalQualityMetrics,
    RAGStage,
    TraceLevel
)

__all__ = [
    # Hallucination Detection
    "RuntimeHallucinationDetector",
    "HallucinationDetectionResult",
    "Claim",
    "ClaimType",
    "ConfidenceLevel",

    # API Resilience
    "APIResilientCaller",
    "CircuitBreaker",
    "APICallResult",
    "CircuitState",
    "CircuitBreakerConfig",
    "RetryConfig",
    "resilient_api_call",

    # RAG Diagnostics
    "RAGTracer",
    "RAGDiagnosticReport",
    "RAGTrace",
    "RetrievedChunk",
    "ContextUtilization",
    "RetrievalQualityMetrics",
    "RAGStage",
    "TraceLevel",
]

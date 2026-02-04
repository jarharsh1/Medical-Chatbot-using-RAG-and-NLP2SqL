"""Unified agent state shared across all pipelines (SQL, RAG, Hybrid)."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Input
    question: str
    session_id: Optional[str]
    conversation_history: str

    # Routing
    query_type: str  # "sql", "rag", "hybrid", "clarification"

    # SQL path
    schema: str
    sql_query: str
    query_result: Optional[str]

    # RAG path
    retrieved_docs: List[Dict[str, Any]]
    rag_answer: Optional[str]

    # Hybrid mode
    hybrid_mode: str  # "filter" or "assist"
    patient_ids: List[int]

    # Quality signals
    sources: List[Dict[str, Any]]
    confidence: float
    grounding: Dict[str, Any]

    # Control
    error: Optional[str]
    iterations: int
    final_answer: str
    clarification: Optional[str]

    # Timing (ms)
    retrieval_time_ms: int
    rerank_time_ms: int
    generation_time_ms: int
    grounding_time_ms: int
    total_time_ms: int

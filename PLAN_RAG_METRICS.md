# RAG-Specific Metrics Implementation Plan

## Objective
Add comprehensive metrics to measure and monitor RAG pipeline quality in real-time.

---

## Metrics to Implement

| # | Metric | Formula/Logic | Why It Matters |
|---|--------|---------------|----------------|
| 1 | **Context Relevance** | LLM scores each retrieved doc (0-1) for query relevance | Measures retrieval quality |
| 2 | **Context Utilization** | % of retrieved tokens that appear in final answer | Detects wasted retrieval |
| 3 | **Answer Faithfulness** | % of answer claims supported by retrieved docs | Measures hallucination |
| 4 | **Retrieval Precision@K** | Relevant docs / Total docs at K | Standard IR metric |
| 5 | **Latency Breakdown** | Time per stage: BM25, Semantic, RRF, Rerank, MMR, Generate | Identifies bottlenecks |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  BM25    │──▶│ Semantic │──▶│   RRF    │──▶│ Reranker │     │
│  │ t1=50ms  │   │ t2=100ms │   │ t3=5ms   │   │ t4=500ms │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│                                                    │            │
│                                                    ▼            │
│                                              ┌──────────┐       │
│                                              │   MMR    │       │
│                                              │ t5=50ms  │       │
│                                              └────┬─────┘       │
│                                                   │             │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
                                                    ▼
                                    ┌───────────────────────────┐
                                    │      RAG Metrics          │
                                    │  Collector (per request)  │
                                    └───────────────┬───────────┘
                                                    │
                    ┌───────────────┬───────────────┼───────────────┬───────────────┐
                    ▼               ▼               ▼               ▼               ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │  Context    │ │  Context    │ │   Answer    │ │ Precision   │ │  Latency    │
            │  Relevance  │ │ Utilization │ │ Faithfulness│ │    @K       │ │  Breakdown  │
            └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

---

## File Structure

```
backend/
├── rag/
│   └── metrics/
│       ├── __init__.py           # Exports
│       ├── collector.py          # RAGMetricsCollector class
│       ├── context_relevance.py  # Metric 1
│       ├── context_utilization.py # Metric 2
│       ├── faithfulness.py       # Metric 3
│       ├── precision.py          # Metric 4
│       └── latency.py            # Metric 5
├── app.py                        # Add /api/rag/metrics endpoint
```

---

## Detailed Implementation

### 1. Context Relevance Score

**What**: Measures how relevant each retrieved document is to the query.

**How**:
```python
def compute_context_relevance(query: str, documents: List[Document]) -> float:
    """
    Uses LLM to score each doc's relevance to query (0-1).
    Returns: average relevance score across all docs.
    """
    scores = []
    for doc in documents:
        prompt = f"""
        Query: {query}
        Document: {doc.text[:500]}

        Rate relevance (0.0 = irrelevant, 1.0 = highly relevant):
        Score:
        """
        score = llm.invoke(prompt)  # Returns float
        scores.append(score)

    return {
        "average": mean(scores),
        "min": min(scores),
        "max": max(scores),
        "per_doc": scores
    }
```

**Integration Point**: After reranking, before generation.

---

### 2. Context Utilization Score

**What**: Measures how much of the retrieved context was actually used in the answer.

**How**:
```python
def compute_context_utilization(context: str, answer: str) -> float:
    """
    Calculates overlap between context and answer.
    Uses n-gram overlap or semantic similarity.
    """
    # Tokenize
    context_tokens = set(tokenize(context.lower()))
    answer_tokens = set(tokenize(answer.lower()))

    # Remove stopwords
    context_tokens -= STOPWORDS
    answer_tokens -= STOPWORDS

    # Calculate utilization
    used_tokens = context_tokens & answer_tokens
    utilization = len(used_tokens) / len(context_tokens) if context_tokens else 0

    return {
        "score": utilization,
        "context_tokens": len(context_tokens),
        "answer_tokens": len(answer_tokens),
        "overlapping_tokens": len(used_tokens),
        "key_terms_used": list(used_tokens)[:10]
    }
```

**Integration Point**: After answer generation.

---

### 3. Answer Faithfulness Score

**What**: Measures if the answer is grounded in retrieved documents (no hallucination).

**How**:
```python
def compute_faithfulness(answer: str, context: str) -> float:
    """
    Decomposes answer into claims, checks each against context.
    """
    # Step 1: Extract claims from answer
    claims = extract_claims(answer)  # LLM call

    # Step 2: Verify each claim against context
    verified = 0
    claim_results = []
    for claim in claims:
        is_supported = verify_claim(claim, context)  # LLM call
        if is_supported:
            verified += 1
        claim_results.append({"claim": claim, "supported": is_supported})

    return {
        "score": verified / len(claims) if claims else 1.0,
        "total_claims": len(claims),
        "verified_claims": verified,
        "unverified_claims": len(claims) - verified,
        "claim_details": claim_results
    }
```

**Note**: We already have `guardrails/grounding.py` - this metric extends it with claim-level detail.

**Integration Point**: After answer generation.

---

### 4. Retrieval Precision@K

**What**: Standard IR metric - what fraction of top-K docs are relevant.

**How**:
```python
def compute_precision_at_k(
    query: str,
    retrieved_docs: List[Document],
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, float]:
    """
    Computes Precision@K for multiple K values.
    Uses LLM or embedding similarity to judge relevance.
    """
    results = {}

    for k in k_values:
        top_k = retrieved_docs[:k]
        relevant_count = 0

        for doc in top_k:
            # Binary relevance judgment
            if is_relevant(query, doc):  # threshold-based or LLM
                relevant_count += 1

        results[f"P@{k}"] = relevant_count / k

    # Also compute Mean Reciprocal Rank (MRR)
    for i, doc in enumerate(retrieved_docs):
        if is_relevant(query, doc):
            results["MRR"] = 1 / (i + 1)
            break
    else:
        results["MRR"] = 0.0

    return results
```

**Integration Point**: After retrieval, using reranker scores as relevance proxy.

---

### 5. Latency Breakdown

**What**: Tracks time spent in each pipeline stage.

**How**:
```python
class LatencyTracker:
    """Context manager for tracking stage latencies."""

    def __init__(self):
        self.stages = {}
        self._start_times = {}

    def start(self, stage: str):
        self._start_times[stage] = time.perf_counter()

    def end(self, stage: str):
        if stage in self._start_times:
            elapsed = (time.perf_counter() - self._start_times[stage]) * 1000
            self.stages[stage] = elapsed

    def get_breakdown(self) -> Dict:
        total = sum(self.stages.values())
        return {
            "total_ms": total,
            "stages": self.stages,
            "percentages": {k: v/total*100 for k, v in self.stages.items()}
        }

# Usage in retriever.py:
tracker = LatencyTracker()

tracker.start("bm25")
bm25_results = bm25_search(query)
tracker.end("bm25")

tracker.start("semantic")
semantic_results = semantic_search(query)
tracker.end("semantic")

# ... etc
```

**Integration Point**: Wrap each stage in retriever.py.

---

## RAGMetricsCollector Class

```python
@dataclass
class RAGMetrics:
    """Complete metrics for a single RAG request."""
    run_id: str
    query: str
    timestamp: float

    # Metric scores
    context_relevance: float
    context_utilization: float
    answer_faithfulness: float
    precision_at_k: Dict[str, float]

    # Latency
    latency_breakdown: Dict[str, float]
    total_latency_ms: float

    # Details
    num_docs_retrieved: int
    num_docs_after_rerank: int
    num_docs_after_mmr: int

    def to_dict(self) -> Dict:
        ...

class RAGMetricsCollector:
    """Collects and aggregates RAG metrics across requests."""

    def __init__(self, window_size: int = 100):
        self._metrics: deque[RAGMetrics] = deque(maxlen=window_size)

    def record(self, metrics: RAGMetrics):
        self._metrics.append(metrics)

    def get_summary(self) -> Dict:
        """Returns aggregated stats over recent requests."""
        return {
            "total_requests": len(self._metrics),
            "avg_context_relevance": mean([m.context_relevance for m in self._metrics]),
            "avg_context_utilization": mean([m.context_utilization for m in self._metrics]),
            "avg_faithfulness": mean([m.answer_faithfulness for m in self._metrics]),
            "avg_precision_at_5": mean([m.precision_at_k.get("P@5", 0) for m in self._metrics]),
            "avg_latency_ms": mean([m.total_latency_ms for m in self._metrics]),
            "latency_p50": percentile([m.total_latency_ms for m in self._metrics], 50),
            "latency_p95": percentile([m.total_latency_ms for m in self._metrics], 95),
        }
```

---

## API Endpoints

### GET /api/rag/metrics
Returns aggregated RAG metrics summary.

```json
{
  "total_requests": 150,
  "metrics": {
    "context_relevance": {"avg": 0.82, "min": 0.45, "max": 0.98},
    "context_utilization": {"avg": 0.35, "min": 0.10, "max": 0.65},
    "answer_faithfulness": {"avg": 0.91, "min": 0.70, "max": 1.0},
    "precision": {"P@3": 0.85, "P@5": 0.78, "P@10": 0.65, "MRR": 0.92}
  },
  "latency": {
    "avg_ms": 1250,
    "p50_ms": 1100,
    "p95_ms": 2500,
    "breakdown": {
      "bm25": 45,
      "semantic": 120,
      "rrf": 5,
      "rerank": 650,
      "mmr": 80,
      "generate": 350
    }
  }
}
```

### GET /api/rag/metrics/{run_id}
Returns detailed metrics for a specific request.

---

## Integration into Existing Code

### retriever.py changes:
```python
def retrieve(query: str) -> RetrievalResult:
    tracker = LatencyTracker()

    # Stage 1: BM25
    tracker.start("bm25")
    bm25_docs = self._bm25_search(query)
    tracker.end("bm25")

    # ... other stages ...

    # Compute metrics
    metrics = RAGMetrics(
        run_id=run_id,
        query=query,
        latency_breakdown=tracker.get_breakdown(),
        # ... other fields computed here
    )

    get_metrics_collector().record(metrics)

    return result
```

### rag_agent.py changes:
```python
def retrieve_and_generate(question: str) -> Dict:
    # ... existing retrieval ...

    # Compute post-generation metrics
    faithfulness = compute_faithfulness(answer, context)
    utilization = compute_context_utilization(context, answer)

    # Include in response
    result["metrics"] = {
        "faithfulness": faithfulness["score"],
        "utilization": utilization["score"],
        # ...
    }
```

---

## Implementation Order

1. **Latency Breakdown** (simplest, no LLM calls)
2. **Context Utilization** (token overlap, no LLM calls)
3. **Precision@K** (uses existing reranker scores)
4. **Context Relevance** (LLM scoring)
5. **Answer Faithfulness** (extends existing grounding.py)

---

## Testing Plan

1. Unit tests for each metric function
2. Integration test: run 10 queries, verify all metrics collected
3. Add to golden_set.json: expected metric ranges for test queries
4. Benchmark: ensure metrics computation adds <100ms overhead

---

## Success Criteria

- [ ] All 5 metrics implemented and tested
- [ ] Metrics visible in API response under `metadata.rag_metrics`
- [ ] Aggregated metrics available at `/api/rag/metrics`
- [ ] Latency overhead < 100ms for non-LLM metrics
- [ ] Documentation updated

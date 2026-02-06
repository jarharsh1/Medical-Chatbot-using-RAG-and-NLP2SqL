"""
4-Stage Hybrid Retriever:

  Stage 1: Dual retrieval (BM25 keyword + ChromaDB semantic) → top 20 each
  Stage 2: Reciprocal Rank Fusion (RRF) → merged top 30
  Stage 3: LLM Re-Ranking → top 15
  Stage 4: MMR Diversity Filter → final top K (default 8)

Mirrors production systems (Google, Azure Cognitive Search).
Each stage trades compute for quality. Pipeline is configurable.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.config import (
    BM25_TOP_K,
    MMR_LAMBDA,
    MMR_TOP_K,
    RERANK_ENABLED,
    RERANK_TOP_K,
    RRF_K,
    SEMANTIC_TOP_K,
)
from backend.rag.metrics import LatencyTracker

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    k: int = RRF_K,
) -> List[Dict]:
    """
    Combine multiple ranked lists using RRF.
    RRF_score(d) = sum over all lists: 1 / (k + rank(d))

    Score-agnostic: works even when BM25 and cosine scores are on different scales.
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc["doc_id"]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0.0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 1.0 / (k + rank)

    # Sort by fused score descending
    sorted_ids = sorted(fused_scores.keys(), key=lambda d: fused_scores[d], reverse=True)

    results = []
    for doc_id in sorted_ids:
        doc = dict(doc_map[doc_id])
        doc["rrf_score"] = fused_scores[doc_id]
        results.append(doc)

    return results


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity with numerical stability."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def maximal_marginal_relevance(
    docs: List[Dict],
    query_embedding: List[float],
    doc_embeddings: Dict[str, List[float]],
    lambda_param: float = MMR_LAMBDA,
    top_k: int = MMR_TOP_K,
) -> List[Dict]:
    """
    MMR: balance relevance (lambda) vs diversity (1 - lambda).

    MMR(d) = λ * Relevance(d, query) - (1-λ) * max(Similarity(d, d_selected))

    λ=0.7 prioritizes relevance while ensuring diversity.

    Improvements:
    - Pre-compute all document vectors as numpy arrays
    - Cache similarity computations
    - Better numerical stability
    """
    if not docs:
        return []

    if len(docs) <= top_k:
        return docs

    query_vec = np.array(query_embedding, dtype=np.float32)

    # Pre-compute document vectors and relevance scores
    doc_vectors: Dict[str, np.ndarray] = {}
    relevance: Dict[str, float] = {}

    for doc in docs:
        doc_id = doc["doc_id"]
        if doc_id in doc_embeddings:
            doc_vec = np.array(doc_embeddings[doc_id], dtype=np.float32)
            doc_vectors[doc_id] = doc_vec
            relevance[doc_id] = _cosine_similarity(query_vec, doc_vec)
        else:
            # Use existing score as fallback (normalize to 0-1 range)
            score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))
            relevance[doc_id] = min(1.0, max(0.0, score))

    # Pre-compute similarity matrix for efficiency (only for docs with embeddings)
    doc_ids_with_vectors = list(doc_vectors.keys())
    n = len(doc_ids_with_vectors)
    sim_matrix: Dict[tuple, float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            id1, id2 = doc_ids_with_vectors[i], doc_ids_with_vectors[j]
            sim = _cosine_similarity(doc_vectors[id1], doc_vectors[id2])
            sim_matrix[(id1, id2)] = sim
            sim_matrix[(id2, id1)] = sim

    def get_doc_similarity(id1: str, id2: str) -> float:
        """Get cached similarity between two documents."""
        if id1 == id2:
            return 1.0
        return sim_matrix.get((id1, id2), 0.0)

    selected: List[Dict] = []
    selected_ids: set = set()
    remaining = list(docs)

    for _ in range(min(top_k, len(docs))):
        best_score = -float("inf")
        best_idx = 0

        for i, doc in enumerate(remaining):
            doc_id = doc["doc_id"]
            rel = relevance.get(doc_id, 0.0)

            # Max similarity to already selected docs
            max_sim = 0.0
            if selected_ids:
                for sel_id in selected_ids:
                    sim = get_doc_similarity(doc_id, sel_id)
                    max_sim = max(max_sim, sim)

            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = round(best_score, 4)
        chosen["mmr_relevance"] = round(relevance.get(chosen["doc_id"], 0.0), 4)
        selected.append(chosen)
        selected_ids.add(chosen["doc_id"])

    return selected


def retrieve(
    query: str,
    top_k: int = MMR_TOP_K,
    where_filter: Optional[Dict] = None,
    skip_rerank: bool = False,
    skip_mmr: bool = False,
    return_metrics: bool = False,
) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Main retrieval entry point: 4-stage pipeline.

    Args:
        query: Search query
        top_k: Final number of documents to return
        where_filter: Optional ChromaDB filter
        skip_rerank: Skip LLM reranking stage
        skip_mmr: Skip MMR diversity stage
        return_metrics: If True, returns (docs, latency_breakdown)

    Returns:
        If return_metrics=False: List of docs
        If return_metrics=True: (List of docs, latency_breakdown dict)
    """
    from backend.rag.bm25 import get_bm25_index
    from backend.rag.vectorstore import semantic_search
    from backend.rag.embeddings import embed_query

    # Initialize latency tracker
    tracker = LatencyTracker()

    # ---- Stage 1: Dual Retrieval ----
    logger.info(f"Stage 1: Dual retrieval for query: {query[:80]}...")

    tracker.start("bm25")
    bm25_results = get_bm25_index().search(query, top_k=BM25_TOP_K, where_filter=where_filter)
    tracker.end("bm25")

    tracker.start("semantic")
    semantic_results = semantic_search(query, top_k=SEMANTIC_TOP_K, where_filter=where_filter)
    tracker.end("semantic")

    logger.info(f"  BM25: {len(bm25_results)} results, Semantic: {len(semantic_results)} results")

    if not bm25_results and not semantic_results:
        logger.warning("No results from either retrieval method.")
        return []

    # ---- Stage 2: RRF Fusion ----
    logger.info("Stage 2: Reciprocal Rank Fusion")
    tracker.start("rrf")
    fused = reciprocal_rank_fusion([bm25_results, semantic_results])
    tracker.end("rrf")
    logger.info(f"  RRF merged: {len(fused)} unique documents")

    # ---- Stage 3: LLM Re-Ranking ----
    if not skip_rerank and RERANK_ENABLED and len(fused) > top_k:
        logger.info("Stage 3: LLM Re-Ranking")
        tracker.start("rerank")
        from backend.rag.reranker import rerank
        reranked = rerank(query, fused, top_k=RERANK_TOP_K)
        tracker.end("rerank")
    else:
        reranked = fused[:RERANK_TOP_K]
        if skip_rerank or not RERANK_ENABLED:
            logger.info("Stage 3: Skipped (reranking disabled)")

    # ---- Stage 4: MMR Diversity Filter ----
    if not skip_mmr and len(reranked) > top_k:
        logger.info("Stage 4: MMR Diversity Filter")
        tracker.start("mmr")

        tracker.start("embedding")
        query_embedding = embed_query(query)
        tracker.end("embedding")

        # Get embeddings for MMR candidates
        doc_embeddings = {}
        try:
            from backend.rag.vectorstore import get_collection
            collection = get_collection()
            candidate_ids = [d["doc_id"] for d in reranked]
            result = collection.get(ids=candidate_ids, include=["embeddings"])
            # Fix: Check embeddings exists and has content (avoiding numpy truthiness issue)
            if result and result.get("embeddings") is not None and len(result["embeddings"]) > 0:
                for i, doc_id in enumerate(result["ids"]):
                    if i < len(result["embeddings"]) and result["embeddings"][i] is not None:
                        doc_embeddings[doc_id] = result["embeddings"][i]
                logger.info(f"  Fetched {len(doc_embeddings)} embeddings for MMR")
        except Exception as e:
            logger.warning(f"Could not fetch embeddings for MMR: {e}")

        if doc_embeddings:
            final = maximal_marginal_relevance(
                reranked, query_embedding, doc_embeddings,
                lambda_param=MMR_LAMBDA, top_k=top_k,
            )
            logger.info(f"  MMR selected {len(final)} diverse documents")
        else:
            logger.warning("  No embeddings available, falling back to top-k")
            final = reranked[:top_k]

        tracker.end("mmr")
    else:
        final = reranked[:top_k]
        if skip_mmr:
            logger.info("Stage 4: Skipped (MMR disabled)")

    logger.info(f"Retrieval complete: {len(final)} documents returned")

    # Collect all stage scores for observability
    for doc in final:
        bm25_match = next((d for d in bm25_results if d["doc_id"] == doc["doc_id"]), None)
        semantic_match = next((d for d in semantic_results if d["doc_id"] == doc["doc_id"]), None)
        doc["bm25_score"] = bm25_match["score"] if bm25_match else 0.0
        doc["semantic_score"] = semantic_match["score"] if semantic_match else 0.0

    # Return with or without metrics
    if return_metrics:
        return final, tracker.get_breakdown()
    return final, None

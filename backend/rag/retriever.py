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
from typing import Dict, List, Optional

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
    """
    if not docs:
        return []

    if len(docs) <= top_k:
        return docs

    query_vec = np.array(query_embedding)

    # Compute relevance scores (cosine sim with query)
    relevance = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        if doc_id in doc_embeddings:
            doc_vec = np.array(doc_embeddings[doc_id])
            sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-10
            )
            relevance[doc_id] = float(sim)
        else:
            # Use existing score as fallback
            relevance[doc_id] = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0.0)))

    selected: List[Dict] = []
    remaining = list(docs)

    for _ in range(min(top_k, len(docs))):
        best_score = -float("inf")
        best_idx = 0

        for i, doc in enumerate(remaining):
            doc_id = doc["doc_id"]
            rel = relevance.get(doc_id, 0.0)

            # Max similarity to already selected docs
            max_sim = 0.0
            if selected and doc_id in doc_embeddings:
                doc_vec = np.array(doc_embeddings[doc_id])
                for sel in selected:
                    sel_id = sel["doc_id"]
                    if sel_id in doc_embeddings:
                        sel_vec = np.array(doc_embeddings[sel_id])
                        sim = np.dot(doc_vec, sel_vec) / (
                            np.linalg.norm(doc_vec) * np.linalg.norm(sel_vec) + 1e-10
                        )
                        max_sim = max(max_sim, float(sim))

            mmr_score = lambda_param * rel - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = best_score
        selected.append(chosen)

    return selected


def retrieve(
    query: str,
    top_k: int = MMR_TOP_K,
    where_filter: Optional[Dict] = None,
    skip_rerank: bool = False,
    skip_mmr: bool = False,
) -> List[Dict]:
    """
    Main retrieval entry point: 4-stage pipeline.

    Returns list of {doc_id, content, metadata, score, rrf_score, rerank_score, mmr_score}.
    """
    from backend.rag.bm25 import get_bm25_index
    from backend.rag.vectorstore import semantic_search
    from backend.rag.embeddings import embed_query

    # ---- Stage 1: Dual Retrieval ----
    logger.info(f"Stage 1: Dual retrieval for query: {query[:80]}...")

    bm25_results = get_bm25_index().search(query, top_k=BM25_TOP_K, where_filter=where_filter)
    semantic_results = semantic_search(query, top_k=SEMANTIC_TOP_K, where_filter=where_filter)

    logger.info(f"  BM25: {len(bm25_results)} results, Semantic: {len(semantic_results)} results")

    if not bm25_results and not semantic_results:
        logger.warning("No results from either retrieval method.")
        return []

    # ---- Stage 2: RRF Fusion ----
    logger.info("Stage 2: Reciprocal Rank Fusion")
    fused = reciprocal_rank_fusion([bm25_results, semantic_results])
    logger.info(f"  RRF merged: {len(fused)} unique documents")

    # ---- Stage 3: LLM Re-Ranking ----
    if not skip_rerank and RERANK_ENABLED and len(fused) > top_k:
        logger.info("Stage 3: LLM Re-Ranking")
        from backend.rag.reranker import rerank
        reranked = rerank(query, fused, top_k=RERANK_TOP_K)
    else:
        reranked = fused[:RERANK_TOP_K]
        if skip_rerank or not RERANK_ENABLED:
            logger.info("Stage 3: Skipped (reranking disabled)")

    # ---- Stage 4: MMR Diversity Filter ----
    if not skip_mmr and len(reranked) > top_k:
        logger.info("Stage 4: MMR Diversity Filter")
        query_embedding = embed_query(query)

        # Get embeddings for MMR candidates
        doc_embeddings = {}
        try:
            from backend.rag.vectorstore import get_collection
            collection = get_collection()
            candidate_ids = [d["doc_id"] for d in reranked]
            result = collection.get(ids=candidate_ids, include=["embeddings"])
            if result and result["embeddings"]:
                for i, doc_id in enumerate(result["ids"]):
                    doc_embeddings[doc_id] = result["embeddings"][i]
        except Exception as e:
            logger.warning(f"Could not fetch embeddings for MMR: {e}")

        if doc_embeddings:
            final = maximal_marginal_relevance(
                reranked, query_embedding, doc_embeddings,
                lambda_param=MMR_LAMBDA, top_k=top_k,
            )
        else:
            final = reranked[:top_k]
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

    return final

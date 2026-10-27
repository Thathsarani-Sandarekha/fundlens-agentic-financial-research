"""
Reciprocal Rank Fusion (RRF) — combines multiple ranked lists.

The RRF formula:
    score(chunk) = sum over each retriever of:
                       1 / (k + rank_in_that_retriever)
    
    where k is a constant (typically 60) that dampens the effect of rank.

Why RRF instead of weighted sum?
  - Doesn't require score normalization (scores from BM25 and vector are 
    on different scales — BM25 is unbounded, vector is 0-1).
  - Robust to outliers.
  - Original paper (Cormack et al. 2009) showed it consistently wins.

When you have 3+ retrievers, just add more terms to the sum.
"""

import logging
from collections import defaultdict
from dataclasses import replace
from typing import Sequence

from fundlens.agent.retrieval.bm25 import ScoredChunk

logger = logging.getLogger(__name__)


# RRF dampening constant. Higher k = less aggressive top-rank weighting.
# k=60 is the value from the original RRF paper. Works well in practice.
RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: Sequence[list[ScoredChunk]],
    top_k: int = 20,
    k_constant: int = RRF_K,
) -> list[ScoredChunk]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.
    
    Args:
        result_lists: One list per retriever, each pre-sorted best-first.
        top_k: How many results to return.
        k_constant: RRF dampening constant (default 60).
    
    Returns:
        Top-K chunks ranked by RRF score, with source tagging:
          - 'both' if found by multiple retrievers
          - Original source ('bm25' or 'vector') if found by only one
    """
    # Map chunk_id → (accumulated RRF score, sources found in, original ScoredChunk)
    chunk_scores: dict[str, dict] = defaultdict(lambda: {
        "rrf_score": 0.0,
        "sources": set(),
        "chunk": None,
    })
    
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            entry = chunk_scores[chunk.chunk_id]
            entry["rrf_score"] += 1.0 / (k_constant + rank)
            entry["sources"].add(chunk.source)
            # Never compare BM25 and cosine similarity raw score scales.
            if entry["chunk"] is None:
                entry["chunk"] = chunk
    
    # Sort by RRF score descending
    sorted_entries = sorted(
        chunk_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )
    
    # Build final list with updated source tagging
    fused = []
    for entry in sorted_entries[:top_k]:
        chunk = replace(entry["chunk"])
        sources = entry["sources"]
        
        # Update source tag: 'both' if multiple retrievers found it
        if len(sources) > 1:
            chunk.source = "both"
        # else keep original source ('bm25' or 'vector')
        
        # Update the score to be the RRF score
        chunk.score = entry["rrf_score"]
        fused.append(chunk)
    
    # Logging
    if fused:
        both_count = sum(1 for c in fused if c.source == "both")
        logger.info(
            f"🔀 RRF fusion: {len(fused)} merged results "
            f"({both_count} found by multiple retrievers, top RRF: {fused[0].score:.4f})"
        )
    
    return fused
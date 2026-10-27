"""
HybridRetriever: combines BM25 + Vector with RRF fusion + optional reranking.

Pipeline:
    Query → BM25(top_20) + Vector(top_20)
          → RRF fusion (top_20)
          → Reranker (top_k, default 5)
          → Return

Use rerank=False to skip the reranker (for ablation/speed).
"""

import logging
from typing import Optional

from fundlens.agent.retrieval.bm25 import get_bm25_retriever, ScoredChunk
from fundlens.agent.retrieval.vector import get_vector_retriever
from fundlens.agent.retrieval.fusion import reciprocal_rank_fusion
from fundlens.agent.retrieval.reranker import get_reranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines BM25 keyword search + vector semantic search via RRF,
    with optional cross-encoder reranking.
    """
    
    def __init__(
        self,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        fusion_top_k: int = 20,
    ):
        """
        Args:
            bm25_top_k: How many results from BM25 before fusion.
            vector_top_k: How many results from vector search.
            fusion_top_k: How many candidates to pass to reranker.
        """
        self.bm25 = get_bm25_retriever()
        self.vector = get_vector_retriever()
        self.reranker = get_reranker()
        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.fusion_top_k = fusion_top_k
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
        company_ticker: Optional[str] = None,
        section: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[str] = None,
    ) -> list[ScoredChunk]:
        """
        Full hybrid pipeline: BM25 + Vector → RRF → (optional Rerank) → top_k.
        
        Args:
            query: User's question.
            top_k: Final number of results to return.
            rerank: Whether to apply the cross-encoder reranker (default True).
            company_ticker, section, fiscal_year, fiscal_quarter: Metadata filters.
        """
        logger.info(f"\n🔍 HYBRID SEARCH: '{query}' (rerank={rerank})")
        
        filters = {
            "company_ticker": company_ticker,
            "section": section,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": fiscal_quarter,
        }
        
        # Stage 1 + 2: BM25 and vector retrieval
        bm25_results = self.bm25.search(query, top_k=self.bm25_top_k, **filters)
        vector_results = self.vector.search(query, top_k=self.vector_top_k, **filters)
        
        # Stage 3: RRF fusion
        fused = reciprocal_rank_fusion(
            [bm25_results, vector_results],
            top_k=self.fusion_top_k if rerank else top_k,
        )
        
        # Stage 4: Rerank (optional)
        if rerank and fused:
            final = self.reranker.rerank(query, fused, top_k=top_k)
        else:
            final = fused[:top_k]
        
        return final


# ============================================================
# SINGLETON
# ============================================================

_hybrid_instance: Optional[HybridRetriever] = None

def get_hybrid_retriever() -> HybridRetriever:
    """Get the singleton HybridRetriever instance."""
    global _hybrid_instance
    if _hybrid_instance is None:
        _hybrid_instance = HybridRetriever()
    return _hybrid_instance
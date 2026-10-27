"""
Cross-encoder reranker — re-scores retrieved chunks against the query.

Why rerank?
  BM25 and vector retrieval are "bi-encoders" — they compute embeddings
  independently for the query and each chunk, then compare.
  
  Cross-encoders take the query AND the chunk TOGETHER, then produce a
  relevance score. Much more accurate but slower (can't precompute).
  
  Strategy: use cheap bi-encoder (vector/BM25) to get top-20 candidates,
  then use expensive cross-encoder to re-rank those 20 into a final top-5.

Model: BAAI/bge-reranker-v2-m3
  - 568MB, CPU-friendly
  - ~50-100ms per chunk on CPU
  - Trained specifically for retrieval reranking
"""

import logging
import time
from functools import lru_cache
from typing import Optional

from fundlens.agent.retrieval.bm25 import ScoredChunk

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# MODEL — lazy load, cached
# ============================================================

@lru_cache(maxsize=1)
def _load_model():
    """Load the cross-encoder model. Downloads ~568MB on first call."""
    from sentence_transformers import CrossEncoder
    logger.info(f"📥 Loading reranker: {RERANKER_MODEL}")
    logger.info("   First load downloads ~90MB.")
    start = time.time()
    model = CrossEncoder(RERANKER_MODEL, device="cpu", max_length=512)
    logger.info(f"✅ Reranker loaded in {time.time() - start:.1f}s")
    return model


# ============================================================
# RERANKER
# ============================================================

class Reranker:
    """
    Cross-encoder reranker over ScoredChunks.
    
    Takes a query + candidate chunks, returns them re-ordered by relevance.
    """
    
    def __init__(self):
        self._model = None
    
    def _ensure_loaded(self):
        if self._model is None:
            self._model = _load_model()
    
    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """
        Re-score chunks against the query and return top-K sorted by relevance.
        
        Args:
            query: The user's question.
            chunks: Candidate chunks (from BM25/Vector/Hybrid).
            top_k: How many top-scored chunks to return.
        
        Returns:
            Top-K chunks sorted by rerank score (descending).
            The chunk.score field is REPLACED with the rerank score.
        """
        if not chunks:
            return []
        
        self._ensure_loaded()
        
        start = time.time()
        
        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, c.content) for c in chunks]
        
        # Score all pairs at once (batched internally by the model)
        scores = self._model.predict(pairs, show_progress_bar=False)
        
        # Attach scores back to chunks
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
            # Note: not modifying chunk.source — keep whatever it was
            # ('bm25', 'vector', or 'both' from fusion stage)
        
        # Sort by rerank score descending
        chunks.sort(key=lambda c: c.score, reverse=True)
        top = chunks[:top_k]
        
        duration = time.time() - start
        top_score = top[0].score if top else 0.0
        logger.info(
            f"🎯 Reranked {len(chunks)} chunks → top {len(top)} "
            f"(top score: {top_score:.3f}, {duration:.2f}s)"
        )
        
        return top


# ============================================================
# SINGLETON
# ============================================================

_reranker_instance: Optional[Reranker] = None

def get_reranker() -> Reranker:
    """Get the singleton Reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker()
    return _reranker_instance
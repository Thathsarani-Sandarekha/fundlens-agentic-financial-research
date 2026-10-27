"""
Vector similarity retriever — clean wrapper over your existing pgvector search.

Wraps embedder.search_similar_chunks() in the same ScoredChunk interface as BM25,
so they can be combined by the RRF fusion layer.
"""

import logging
from typing import Optional

from fundlens.ingestion.embedder import search_similar_chunks
from fundlens.agent.retrieval.bm25 import ScoredChunk

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Vector similarity retriever over document_chunks.
    
    Uses bge-base embeddings + pgvector cosine similarity.
    Already populated by your ingestion pipeline.
    """
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        company_ticker: Optional[str] = None,
        section: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[str] = None,
    ) -> list[ScoredChunk]:
        """Search for chunks similar to the query (cosine similarity)."""
        
        # Note: fiscal_quarter filter isn't supported by search_similar_chunks 
        # — we apply it client-side. Easy fix later.
        raw_results = search_similar_chunks(
            query=query,
            top_k=top_k,
            company_ticker=company_ticker,
            section=section,
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
        )
        
        # Convert to ScoredChunk
        results = []
        for r in raw_results:
            results.append(ScoredChunk(
                chunk_id=r["chunk_id"],
                content=r["raw_content"],
                company_ticker=r["company_ticker"],
                filing_type=r["filing_type"],
                fiscal_year=r["fiscal_year"],
                fiscal_quarter=r["fiscal_quarter"],
                section=r["section"],
                page_number=r["page_number"],
                score=float(r["similarity"]),
                source="vector",
            ))
            
            if len(results) >= top_k:
                break
        
        top_score = results[0].score if results else 0.0
        logger.info(
            f"📐 Vector search for '{query[:50]}...': "
            f"{len(results)} results (top similarity: {top_score:.3f})"
        )
        return results


# ============================================================
# SINGLETON
# ============================================================

_vector_instance: Optional[VectorRetriever] = None

def get_vector_retriever() -> VectorRetriever:
    """Get the singleton VectorRetriever instance."""
    global _vector_instance
    if _vector_instance is None:
        _vector_instance = VectorRetriever()
    return _vector_instance
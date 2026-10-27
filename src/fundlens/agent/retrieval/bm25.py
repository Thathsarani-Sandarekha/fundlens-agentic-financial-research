"""
BM25 keyword-based retriever over document_chunks.

BM25 catches what vector search misses:
  - Exact keyword matches (ticker symbols, named entities)
  - Acronyms (AI, EPS, GAAP)
  - Numbers in narrative ("Q1 2025", "Section 232")
  - Rare terms that dominate IDF

Uses rank_bm25 (in-memory, Python). For 261 chunks this is fine.
At 100k+ chunks, switch to Postgres FTS or Elasticsearch.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from rank_bm25 import BM25Okapi
from sqlalchemy import select

from fundlens.storage.db import get_session
from fundlens.storage.models import DocumentChunk

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class ScoredChunk:
    """A chunk returned from a retriever with its score."""
    chunk_id: str
    content: str  # raw_content
    company_ticker: str
    filing_type: str
    fiscal_year: int
    fiscal_quarter: str
    section: str
    page_number: int
    score: float
    source: str = "bm25"  # 'bm25', 'vector', or 'both' (set later by fusion)


# ============================================================
# TOKENIZATION
# ============================================================

# Simple tokenizer: lowercase, split on non-alphanumerics, keep tokens >= 2 chars
# Good enough for English financial text. Production systems use spaCy/NLTK.
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25. Lowercases, splits on word boundaries."""
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2]


# ============================================================
# BM25 RETRIEVER
# ============================================================

class BM25Retriever:
    """
    BM25 index over document_chunks.
    
    Loads all chunks into memory on first use, builds the index, then
    serves queries fast. Re-load (call .reload()) if chunks change.
    """
    
    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Load all chunks from DB and build the BM25 index."""
        if self._initialized:
            return
        
        logger.info("📚 Initializing BM25 retriever...")
        start = time.time()
        
        with get_session() as s:
            chunks = list(s.execute(select(DocumentChunk)).scalars().all())
        
        if not chunks:
            logger.warning("⚠️  No chunks in document_chunks table!")
            self._chunks = []
            self._initialized = True
            return
        
        self._chunks = chunks
        self._tokenized_corpus = [tokenize(c.raw_content) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._initialized = True
        
        logger.info(
            f"✅ BM25 ready: {len(chunks)} chunks indexed in {time.time() - start:.1f}s"
        )
    
    def reload(self) -> None:
        """Force re-load chunks from DB. Use after ingestion runs."""
        self._initialized = False
        self.initialize()
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        company_ticker: Optional[str] = None,
        section: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_quarter: Optional[str] = None,
    ) -> list[ScoredChunk]:
        """
        Search for chunks matching the query, with optional metadata filters.
        
        Returns top_k highest-scoring chunks.
        Filters are applied AFTER scoring (so top_k is the post-filter count).
        """
        self.initialize()
        
        if not self._chunks:
            return []
        
        # Tokenize query
        query_tokens = tokenize(query)
        if not query_tokens:
            logger.warning(f"Empty query after tokenization: '{query}'")
            return []
        
        # Score all chunks
        scores = self._bm25.get_scores(query_tokens)
        
        # Pair scores with chunks, apply filters
        scored = []
        for chunk, score in zip(self._chunks, scores):
            # Apply metadata filters
            if company_ticker and chunk.company_ticker != company_ticker:
                continue
            if section and chunk.section != section:
                continue
            if fiscal_year and chunk.fiscal_year != fiscal_year:
                continue
            if fiscal_quarter and chunk.fiscal_quarter != fiscal_quarter:
                continue
            
            scored.append((chunk, float(score)))
        
        # Sort by score desc, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        
        # Convert to ScoredChunk
        results = [
            ScoredChunk(
                chunk_id=c.chunk_id,
                content=c.raw_content,
                company_ticker=c.company_ticker,
                filing_type=c.filing_type,
                fiscal_year=c.fiscal_year,
                fiscal_quarter=c.fiscal_quarter,
                section=c.section,
                page_number=c.page_number,
                score=s,
                source="bm25",
            )
            for c, s in top
            if s > 0  # filter out zero-score chunks (no token overlap)
        ]
        
        top_score = results[0].score if results else 0.0
        logger.info(
            f"🔤 BM25 search for '{query[:50]}...': "
            f"{len(results)} results (top score: {top_score:.2f})"
        )
        return results


# ============================================================
# SINGLETON
# ============================================================

_bm25_instance: Optional[BM25Retriever] = None

def get_bm25_retriever() -> BM25Retriever:
    """Get the singleton BM25Retriever instance."""
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Retriever()
    return _bm25_instance
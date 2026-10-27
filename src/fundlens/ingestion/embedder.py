"""
Embedding generation + pgvector storage for narrative chunks.

Pipeline:
  1. Load model (lazy, singleton)
  2. Batch embed text chunks (CPU-friendly)
  3. Insert/upsert into document_chunks table

Usage:
    from ingestion.embedder import embed_and_store_chunks
    inserted = embed_and_store_chunks(chunks)

Model:
    BAAI/bge-small-en-v1.5 — 384 dims, ~130MB, runs on CPU
    Top of MTEB leaderboard for English in its size class.
"""

import logging
import time
from typing import Iterable, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from fundlens.models.chunk import TextChunk
from fundlens.storage.db import get_session
from fundlens.storage.models import DocumentChunk

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768 
BATCH_SIZE = 32

# BGE recommends adding this prefix when embedding queries (not docs)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ============================================================
# MODEL — singleton, lazy load
# ============================================================

_model = None


def get_model():
    """Lazy-load the embedding model. Downloads ~130MB on first call."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"📥 Loading embedding model: {EMBEDDING_MODEL_NAME}")
        logger.info(f"   First load downloads ~130MB. Subsequent loads are instant.")
        start = time.time()
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        logger.info(f"✅ Model loaded in {time.time() - start:.1f}s")
    return _model


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def embed_texts(
    texts: list[str],
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts. Returns array of shape (N, 384).
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # IMPORTANT for cosine similarity
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a search query (with BGE's recommended prefix).
    Returns a list (not array) for easier passing to pgvector.
    """
    model = get_model()
    full_query = QUERY_PREFIX + query
    embedding = model.encode(
        [full_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return embedding.tolist()


# ============================================================
# DB INSERT / UPSERT
# ============================================================

def embed_and_store_chunks(
    chunks: list[TextChunk],
    skip_existing: bool = True,
) -> dict:
    """
    Embed chunks and upsert into document_chunks table.

    Args:
        chunks: List of TextChunk from text_chunker
        skip_existing: If True, skip chunks already in the DB (by chunk_id)

    Returns:
        Summary dict: {inserted, skipped, total, duration_seconds}
    """
    start = time.time()
    total = len(chunks)

    if total == 0:
        logger.warning("  No chunks to embed")
        return {"inserted": 0, "skipped": 0, "total": 0, "duration_seconds": 0}

    logger.info(f"\n🔢 Embedding {total} chunks...")

    # ── 1. Check which chunks already exist (idempotency) ────
    skipped = 0
    chunks_to_embed = chunks

    if skip_existing:
        chunk_ids = [c.chunk_id for c in chunks]
        with get_session() as s:
            existing = set(
                row[0] for row in s.execute(
                    select(DocumentChunk.chunk_id).where(
                        DocumentChunk.chunk_id.in_(chunk_ids)
                    )
                ).all()
            )
        chunks_to_embed = [c for c in chunks if c.chunk_id not in existing]
        skipped = len(existing)
        if skipped > 0:
            logger.info(f"   ⏭️  Skipping {skipped} chunks already in DB")

    if not chunks_to_embed:
        logger.info(f"   ✅ All chunks already embedded")
        return {"inserted": 0, "skipped": skipped, "total": total, "duration_seconds": time.time() - start}

    # ── 2. Embed in batches ──────────────────────────────────
    # Embed raw_content (without prefix) so vector search isn't dominated
    # by metadata boilerplate. The prefix-enhanced 'content' is preserved
    # for use as context when sending chunks to a synthesis LLM.
    texts = [c.raw_content for c in chunks_to_embed]
    embeddings = embed_texts(texts, batch_size=BATCH_SIZE, show_progress=True)
    
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Model returned {embeddings.shape[1]}-dim embeddings, "
            f"but schema expects {EMBEDDING_DIM}"
        )

    # ── 3. Bulk insert ───────────────────────────────────────
    logger.info(f"   💾 Inserting {len(chunks_to_embed)} chunks into Postgres...")
    inserted = 0

    with get_session() as s:
        for chunk, embedding in zip(chunks_to_embed, embeddings):
            doc_chunk = DocumentChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                raw_content=chunk.raw_content,
                embedding=embedding.tolist(),
                filing_id=chunk.filing_id,
                company_ticker=chunk.company_ticker,
                filing_type=chunk.filing_type,
                fiscal_year=chunk.fiscal_year,
                fiscal_quarter=chunk.fiscal_quarter,
                section=chunk.section,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                overlap_with_prev=chunk.overlap_with_prev,
            )
            s.merge(doc_chunk)  # merge = upsert by primary key
            inserted += 1

    duration = time.time() - start
    logger.info(f"   ✅ Stored {inserted} chunks in {duration:.1f}s")

    return {
        "inserted": inserted,
        "skipped": skipped,
        "total": total,
        "duration_seconds": duration,
    }


# ============================================================
# SIMILARITY SEARCH
# ============================================================

def search_similar_chunks(
    query: str,
    top_k: int = 5,
    company_ticker: Optional[str] = None,
    section: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    fiscal_quarter: Optional[str] = None,
) -> list[dict]:
    """
    Run cosine similarity search over the embedded chunks.

    Args:
        query: Natural language query
        top_k: Number of results to return
        company_ticker, section, fiscal_year: Optional metadata filters

    Returns:
        List of dicts with chunk metadata + similarity score
    """
    logger.info(f"\n🔍 Searching: '{query}' (top {top_k})")

    # Embed query
    query_emb = embed_query(query)

    # Build SQL with optional filters
    from sqlalchemy import text as sql_text

    where_clauses = []
    params = {"query_emb": str(query_emb), "top_k": top_k}

    if company_ticker:
        where_clauses.append("company_ticker = :ticker")
        params["ticker"] = company_ticker
    if section:
        where_clauses.append("section = :section")
        params["section"] = section
    if fiscal_year:
        where_clauses.append("fiscal_year = :year")
        params["year"] = fiscal_year

    if fiscal_quarter:
        where_clauses.append("fiscal_quarter = :quarter")
        params["quarter"] = fiscal_quarter
    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT
            chunk_id,
            raw_content,
            company_ticker,
            filing_type,
            fiscal_year,
            fiscal_quarter,
            section,
            page_number,
            1 - (embedding <=> CAST(:query_emb AS vector)) AS similarity
        FROM fundlens.document_chunks
        {where_sql}
        ORDER BY embedding <=> CAST(:query_emb AS vector)
        LIMIT :top_k
    """

    with get_session() as s:
        result = s.execute(sql_text(sql), params)
        rows = [dict(row._mapping) for row in result.fetchall()]

    logger.info(f"   ✅ Found {len(rows)} results")
    return rows
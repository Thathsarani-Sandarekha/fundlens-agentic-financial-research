"""
Build the Neo4j knowledge graph from Postgres data + LLM entity extraction.

Two ingestion sources:
  1. Sync structured data already in Postgres (Company, Filing, Segment, Executive)
  2. LLM-extract topics + competitors from narrative chunks (per filing)

Usage:
    from ingestion.graph_builder import build_graph_for_filing
    build_graph_for_filing(
        ticker="GOOGL",
        fiscal_year=2025,
        fiscal_quarter="Q1",
        filing_type="10-Q",
    )
"""

import hashlib
import logging
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import select

from fundlens.config import settings
from fundlens.storage.graph_client import neo4j_session
from fundlens.storage.graph_schema import CANONICAL_TOPICS, bootstrap_schema
from fundlens.storage.db import get_session
from fundlens.storage.models import (
    Company, FilingPeriod, Filing,
    SegmentMetric, ExecutiveCommentary, DocumentChunk,
)
from fundlens.llm.groq_client import get_instructor_client, call_with_backoff

logger = logging.getLogger(__name__)


# ============================================================
# LLM CONFIG
# ============================================================

GROQ_MODEL = settings.groq_model_default


# ============================================================
# PYDANTIC SCHEMA FOR LLM EXTRACTION
# ============================================================

class ExtractedTopic(BaseModel):
    """A topic mentioned in a filing, from the controlled vocabulary."""
    topic_name: str = Field(
        ...,
        description=f"Must be one of: {', '.join(CANONICAL_TOPICS)}",
    )
    relevance: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="How prominently this topic features (0=mention, 1=core theme)",
    )
    context_snippet: Optional[str] = Field(
        None,
        description="Brief 1-sentence context where this topic appears (~150 chars)",
    )


class ExtractedCompetitor(BaseModel):
    """A competing company explicitly named in the filing."""
    company_name: str = Field(..., description="Name as mentioned in filing")
    standardized_ticker: Optional[str] = Field(
        None,
        description="Stock ticker if obvious (e.g., 'Microsoft' → 'MSFT'). Null if unknown.",
    )


class ExtractedEntities(BaseModel):
    """Output of LLM entity extraction from a filing's narrative."""
    topics: list[ExtractedTopic] = Field(default_factory=list)
    competitors: list[ExtractedCompetitor] = Field(default_factory=list)


# ============================================================
# HELPERS
# ============================================================

def deterministic_id(*parts: str) -> str:
    """Build a stable ID from parts. Same inputs → same ID."""
    return hashlib.md5("__".join(str(p) for p in parts).encode()).hexdigest()[:16]


# ============================================================
# SYNC FROM POSTGRES → NEO4J
# ============================================================

def sync_company(ticker: str) -> str:
    """Upsert a Company node from Postgres. Returns the ticker (Neo4j key)."""
    with get_session() as s:
        company = s.execute(
            select(Company).where(Company.ticker == ticker)
        ).scalar_one()

        with neo4j_session() as nsess:
            nsess.run(
                """
                MERGE (c:Company {ticker: $ticker})
                SET c.name = $name,
                    c.sector = $sector,
                    c.industry = $industry,
                    c.market_cap_bucket = $bucket,
                    c.updated_at = datetime()
                """,
                ticker=company.ticker,
                name=company.name,
                sector=company.sector,
                industry=company.industry,
                bucket=company.market_cap_bucket,
            )
    logger.info(f"  + Synced Company: {ticker}")
    return ticker


def sync_filing(ticker: str, fiscal_year: int, fiscal_quarter: str, filing_type: str) -> str:
    """Upsert a Filing node + FILED edge. Returns filing_id."""
    with get_session() as s:
        company = s.execute(
            select(Company).where(Company.ticker == ticker)
        ).scalar_one()
        period = s.execute(
            select(FilingPeriod).join(Filing, Filing.period_id == FilingPeriod.id).where(
                Filing.company_id == company.id,
                FilingPeriod.fiscal_year == fiscal_year,
                FilingPeriod.fiscal_quarter == fiscal_quarter,
                FilingPeriod.filing_type == filing_type,
            )
        ).scalar_one()
        filing = s.execute(
            select(Filing).where(
                Filing.company_id == company.id,
                Filing.period_id == period.id,
            )
        ).scalar_one()

        filing_id = deterministic_id(ticker, fiscal_year, fiscal_quarter, filing_type)

        with neo4j_session() as nsess:
            nsess.run(
                """
                MATCH (c:Company {ticker: $ticker})
                MERGE (f:Filing {id: $id})
                SET f.filing_type = $filing_type,
                    f.fiscal_year = $year,
                    f.fiscal_quarter = $quarter,
                    f.period_label = $label,
                    f.filing_date = date($filing_date),
                    f.s3_key = $s3_key
                MERGE (c)-[:FILED]->(f)
                """,
                ticker=ticker,
                id=filing_id,
                filing_type=filing_type,
                year=fiscal_year,
                quarter=fiscal_quarter,
                label=period.period_label,
                filing_date=filing.filing_date.isoformat(),
                s3_key=filing.s3_key,
            )
    logger.info(f"  + Synced Filing: {filing_id[:8]}... ({ticker} {fiscal_quarter} {fiscal_year})")
    return filing_id


def sync_segments(ticker: str, fiscal_year: int, fiscal_quarter: str) -> int:
    """Sync all SegmentMetrics for this (company, period) as Segment nodes + edges."""
    with get_session() as s:
        company = s.execute(
            select(Company).where(Company.ticker == ticker)
        ).scalar_one()
        period = s.execute(
            select(FilingPeriod).join(Filing, Filing.period_id == FilingPeriod.id).where(
                Filing.company_id == company.id,
                FilingPeriod.fiscal_year == fiscal_year,
                FilingPeriod.fiscal_quarter == fiscal_quarter,
            )
        ).scalar_one()
        segments = s.execute(
            select(SegmentMetric).where(
                SegmentMetric.company_id == company.id,
                SegmentMetric.period_id == period.id,
            )
        ).scalars().all()

        if not segments:
            return 0

        count = 0
        with neo4j_session() as nsess:
            for seg in segments:
                seg_id = deterministic_id(ticker, seg.segment_name)
                nsess.run(
                    """
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (s:Segment {id: $id})
                    SET s.name = $name,
                        s.name_raw = $name_raw,
                        s.segment_type = $seg_type
                    MERGE (c)-[r:HAS_SEGMENT]->(s)
                    SET r.latest_revenue = $revenue,
                        r.latest_period = $period_label
                    """,
                    ticker=ticker,
                    id=seg_id,
                    name=seg.segment_name,
                    name_raw=seg.segment_name_raw,
                    seg_type=seg.segment_type,
                    revenue=float(seg.segment_revenue) if seg.segment_revenue is not None else None,
                    period_label=period.period_label,
                )
                count += 1

    logger.info(f"  + Synced {count} Segment nodes for {ticker}")
    return count


def sync_executives(ticker: str) -> int:
    """Sync executives from ExecutiveCommentary table for this company."""
    with get_session() as s:
        company = s.execute(
            select(Company).where(Company.ticker == ticker)
        ).scalar_one()
        commentary = s.execute(
            select(ExecutiveCommentary).where(
                ExecutiveCommentary.company_id == company.id,
            )
        ).scalars().all()

        if not commentary:
            return 0

        # Dedupe by (name, title) — multiple quotes from same exec
        unique_execs: dict[tuple, dict] = {}
        for ec in commentary:
            if not ec.executive_name:
                continue
            key = (ec.executive_name, ec.executive_title or "")
            if key not in unique_execs:
                unique_execs[key] = {
                    "name": ec.executive_name,
                    "title": ec.executive_title,
                    "topics_discussed": set(),
                }
            if ec.topic:
                unique_execs[key]["topics_discussed"].add(ec.topic)

        count = 0
        with neo4j_session() as nsess:
            for (name, title), data in unique_execs.items():
                exec_id = deterministic_id(ticker, name, title)
                nsess.run(
                    """
                    MATCH (c:Company {ticker: $ticker})
                    MERGE (e:Executive {id: $id})
                    SET e.name = $name,
                        e.title = $title,
                        e.company_ticker = $ticker
                    MERGE (c)-[:HAS_EXECUTIVE]->(e)
                    """,
                    ticker=ticker,
                    id=exec_id,
                    name=name,
                    title=title,
                )
                count += 1

    logger.info(f"  + Synced {count} Executive nodes for {ticker}")
    return count


# ============================================================
# LLM ENTITY EXTRACTION FROM CHUNKS
# ============================================================

EXTRACTION_SYSTEM_PROMPT = f"""You analyze SEC filing excerpts and extract:
1. TOPICS being discussed — MUST be from this controlled vocabulary:
   {', '.join(CANONICAL_TOPICS)}
2. COMPETITORS — explicitly named competing companies

RULES:
- Only return topics from the list above. Don't invent new ones.
- For competitors, give the company name as written + standardized ticker if obvious.
  Examples: "Microsoft" → MSFT, "Amazon" → AMZN, "Apple" → AAPL, "Meta" → META.
  If ticker is unclear, leave null.
- Be selective. Only flag topics that are actually substantive in the text.
- Set relevance high (0.7+) only if topic is a major theme, not just a brief mention.
"""


def extract_entities_from_chunks(
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: str,
) -> Optional[ExtractedEntities]:
    """
    Pull relevant chunks for this filing, send to LLM, return extracted entities.
    """
    # Pull narrative chunks (focus on MD&A, Risk Factors, Forward-Looking)
    INTERESTING_SECTIONS = ("MD&A", "Risk Factors", "Forward-Looking Statements", "Business")

    with get_session() as s:
        chunks = s.execute(
            select(DocumentChunk).where(
                DocumentChunk.company_ticker == ticker,
                DocumentChunk.fiscal_year == fiscal_year,
                DocumentChunk.fiscal_quarter == fiscal_quarter,
                DocumentChunk.section.in_(INTERESTING_SECTIONS),
            ).limit(8)  # cap to keep prompt size manageable
        ).scalars().all()

    if not chunks:
        logger.warning(f"  ⚠️  No interesting chunks found for {ticker} {fiscal_quarter} {fiscal_year}")
        return None

    # Build prompt with chunk content
    text_corpus = "\n\n---\n\n".join(
        f"[Section: {c.section} | Page {c.page_number}]\n{c.raw_content[:600]}"
        for c in chunks
    )

    user_prompt = f"""Analyze these excerpts from {ticker}'s {fiscal_quarter} {fiscal_year} filing.

{text_corpus}

Extract topics (from the controlled list) and explicitly named competitors.
"""

    try:
        client = get_instructor_client()
        result: ExtractedEntities = call_with_backoff(
        client,
        model=GROQ_MODEL,
        response_model=ExtractedEntities,
         messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        max_retries=2,
        temperature=0.0,
        max_tokens=2048,
    )
        return result
    except Exception as e:
        logger.error(f"  ❌ Entity extraction failed: {e}")
        return None


def merge_topics_to_graph(
    ticker: str,
    filing_id: str,
    topics: list[ExtractedTopic],
):
    """Create MENTIONS edges from Filing to Topic, filter to canonical only."""
    if not topics:
        return 0

    count = 0
    with neo4j_session() as nsess:
        for topic in topics:
            if topic.topic_name not in CANONICAL_TOPICS:
                logger.debug(f"    ⏭️  Skipping non-canonical topic: {topic.topic_name}")
                continue

            nsess.run(
                """
                MATCH (f:Filing {id: $filing_id})
                MATCH (t:Topic {name: $topic_name})
                MERGE (f)-[r:MENTIONS]->(t)
                SET r.relevance = $relevance,
                    r.context = $context
                """,
                filing_id=filing_id,
                topic_name=topic.topic_name,
                relevance=topic.relevance,
                context=topic.context_snippet,
            )

            # Also link executives discussing this topic
            nsess.run(
                """
                MATCH (c:Company {ticker: $ticker})-[:HAS_EXECUTIVE]->(e:Executive)
                MATCH (t:Topic {name: $topic_name})
                MERGE (e)-[:DISCUSSED]->(t)
                """,
                ticker=ticker,
                topic_name=topic.topic_name,
            )
            count += 1

    logger.info(f"  + Linked {count} Topic edges from filing")
    return count


def merge_competitors_to_graph(
    ticker: str,
    competitors: list[ExtractedCompetitor],
):
    """Create COMPETES_WITH edges between Companies."""
    if not competitors:
        return 0

    count = 0
    with neo4j_session() as nsess:
        for comp in competitors:
            # Create competitor company node (minimal, may not be in Postgres yet)
            if not comp.standardized_ticker:
                continue  # Do not invent ticker symbols from a company name
            comp_ticker = comp.standardized_ticker
            nsess.run(
                """
                MERGE (c2:Company {ticker: $comp_ticker})
                ON CREATE SET c2.name = $name, c2.created_via = 'competitor_mention'
                """,
                comp_ticker=comp_ticker,
                name=comp.company_name,
            )

            # Create bidirectional COMPETES_WITH (use MERGE so no dupes)
            nsess.run(
                """
                MATCH (c1:Company {ticker: $t1})
                MATCH (c2:Company {ticker: $t2})
                WHERE c1.ticker <> c2.ticker
                MERGE (c1)-[r:COMPETES_WITH]->(c2)
                SET r.last_mentioned = datetime()
                """,
                t1=ticker,
                t2=comp_ticker,
            )
            count += 1

    logger.info(f"  + Created {count} COMPETES_WITH edges")
    return count


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def build_graph_for_filing(
    *,
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: str,
    filing_type: str,
    extract_with_llm: bool = True,
) -> dict:
    """
    Full graph build for one filing:
      1. Bootstrap schema (idempotent)
      2. Sync Company, Filing, Segments, Executives from Postgres
      3. Extract Topics + Competitors via LLM (optional)
      4. Wire up all edges

    Returns a summary dict.
    """
    logger.info(f"\n🕸️  Building graph for {ticker} {fiscal_quarter} {fiscal_year} {filing_type}")

    summary = {
        "company": ticker,
        "filing_id": None,
        "segments_synced": 0,
        "executives_synced": 0,
        "topics_linked": 0,
        "competitors_linked": 0,
        "llm_extraction_done": False,
    }

    # 1. Schema bootstrap (idempotent)
    bootstrap_schema()

    # 2. Sync structured entities
    sync_company(ticker)
    filing_id = sync_filing(ticker, fiscal_year, fiscal_quarter, filing_type)
    summary["filing_id"] = filing_id
    summary["segments_synced"] = sync_segments(ticker, fiscal_year, fiscal_quarter)
    summary["executives_synced"] = sync_executives(ticker)

    # 3. LLM-extract topics + competitors
    if extract_with_llm:
        entities = extract_entities_from_chunks(ticker, fiscal_year, fiscal_quarter)
        if entities:
            summary["llm_extraction_done"] = True
            summary["topics_linked"] = merge_topics_to_graph(
                ticker, filing_id, entities.topics
            )
            summary["competitors_linked"] = merge_competitors_to_graph(
                ticker, entities.competitors
            )

    logger.info(f"\n✅ Graph build complete for {ticker}")
    return summary
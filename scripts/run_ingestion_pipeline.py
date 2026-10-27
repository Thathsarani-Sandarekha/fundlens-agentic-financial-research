"""
End-to-end ingestion pipeline for a single SEC filing.

Pipeline stages:
  1. Parse PDF              → ParsedDocument
  2. Verify tables (LLM)    → list[VerifiedTable]
  3. Map to financial_metrics → Postgres rows
  4. Chunk narrative text   → list[TextChunk]
  5. Embed + store chunks   → pgvector rows
  6. Save full output JSON  → data/processed/{ticker}/{filing_id}.json

Usage:
    python scripts/run_ingestion_pipeline.py AAPL 2025 Q1 10-Q
    python scripts/run_ingestion_pipeline.py AAPL 2024 FY 10-K
"""

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ingestion_pipeline")

# Reduce noise from libraries
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Project imports
from fundlens.config import settings
from fundlens.ingestion.parser import parse_document  # adjust if your function name differs
from fundlens.ingestion.table_extractor import verify_all_tables_in_document
from fundlens.ingestion.text_chunker import chunk_document
from fundlens.ingestion.embedder import embed_and_store_chunks
from fundlens.ingestion.financial_mapper import extract_and_store
from fundlens.storage.s3_client import upload_file, ensure_bucket


# ============================================================
# COMPANY METADATA (extend as needed)
# ============================================================

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla Inc.",
    "NFLX": "Netflix Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corp.",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "GS": "Goldman Sachs Group",
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale",
    "HD": "Home Depot Inc.",
    "NKE": "Nike Inc.",
    "XOM": "Exxon Mobil Corp.",
    "CVX": "Chevron Corporation",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer Inc.",
    "UNH": "UnitedHealth Group",
    "KO": "Coca-Cola Company",
    "PEP": "PepsiCo Inc.",
    "DIS": "Walt Disney Company",
    "MCD": "McDonald's Corporation",
    "BA": "Boeing Company",
}

# ============================================================
# PATH HELPERS
# ============================================================

def build_pdf_path(ticker: str, fiscal_year: int, fiscal_quarter: str, filing_type: str) -> Path:
    """Construct path to the PDF based on your data/raw/ naming convention."""
    if filing_type == "10-K":
        filename = f"{ticker}_{fiscal_year}_10K.pdf"
    else:
        filename = f"{ticker}_{fiscal_year}_{fiscal_quarter}_10Q.pdf"
    
    return settings.data_raw_dir / "sec_filings" / ticker / filename


def build_filing_id(ticker: str, fiscal_year: int, fiscal_quarter: str, filing_type: str) -> str:
    """Standard filing_id used throughout the system."""
    return f"{ticker}__{filing_type}__{fiscal_quarter}_{fiscal_year}"


def build_output_path(ticker: str, filing_id: str) -> Path:
    """Where to save the full pipeline output JSON."""
    output_dir = settings.data_processed_dir / ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{filing_id}.json"


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: str,
    filing_type: str,
    skip_embedding: bool = False,
    skip_graph: bool = False,
    period_end_date: date | None = None,
    filing_date: date | None = None,
) -> dict:
    """
    Run the full ingestion pipeline on one filing.
    Returns a summary dict of what happened.
    """
    
    # ─── Setup ────────────────────────────────────────────
    pdf_path = build_pdf_path(ticker, fiscal_year, fiscal_quarter, filing_type)
    filing_id = build_filing_id(ticker, fiscal_year, fiscal_quarter, filing_type)
    output_path = build_output_path(ticker, filing_id)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found locally: {pdf_path}")
    
    company_name = COMPANY_NAMES.get(ticker, ticker)
    
    # Default period_end_date
    if period_end_date is None or filing_date is None:
        raise ValueError("Pass actual SEC --period-end and --filing-date; fiscal calendars vary")
    
    # Build S3 key matching MinIO upload structure
    # Local: data/raw/sec_filings/AAPL/AAPL_2025_Q1_10Q.pdf
    # S3:    s3://fundlens-filings/filings/sec_filings/AAPL/AAPL_2025_Q1_10Q.pdf
    s3_key = f"filings/sec_filings/{ticker}/{pdf_path.name}"
    
    logger.info("=" * 70)
    logger.info(f"INGESTION PIPELINE: {filing_id}")
    logger.info(f"Local PDF: {pdf_path}")
    logger.info(f"S3 key:    s3://{settings.s3_bucket}/{s3_key}")
    logger.info(f"Output:    {output_path}")
    logger.info("=" * 70)
    
    pipeline_start = time.time()
    summary = {
        "filing_id": filing_id,
        "ticker": ticker,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "filing_type": filing_type,
        "pdf_path": str(pdf_path),
        "s3_bucket": settings.s3_bucket,
        "s3_key": s3_key,
        "stages": {},
    }
    
    # ─── Stage 1: Parse PDF ───────────────────────────────
    logger.info("\n[STAGE 1/5] Parsing PDF...")
    stage_start = time.time()
    
    ensure_bucket(settings.s3_bucket)
    upload_file(pdf_path, settings.s3_bucket, s3_key, content_type="application/pdf")
    parsed = parse_document(
        bucket=settings.s3_bucket,
        key=s3_key,
        ticker=ticker,
        filing_type=filing_type,
        filing_id=filing_id,
    )
    
    summary["stages"]["parse"] = {
        "duration_sec": round(time.time() - stage_start, 2),
        "total_pages": parsed.total_pages_in_pdf,
        "pages_parsed": parsed.pages_parsed_successfully,
        "total_tables": parsed.total_tables,
        "total_text_blocks": parsed.total_text_blocks,
        "success_rate": round(parsed.parse_success_rate, 3),
    }
    logger.info(f"  ✅ {parsed.summary()}")
    
    # ─── Stage 2: Verify Tables ───────────────────────────
    logger.info("\n[STAGE 2/5] Verifying tables with LLM...")
    stage_start = time.time()
    
    verified_tables = verify_all_tables_in_document(parsed)
    
    total_verified = sum(len(tables) for tables in verified_tables.values())
    high_conf = sum(
        1 for tables in verified_tables.values()
        for t in tables if t.is_high_confidence
    )
    
    summary["stages"]["verify_tables"] = {
        "duration_sec": round(time.time() - stage_start, 2),
        "total_verified": total_verified,
        "high_confidence": high_conf,
        "by_type": _count_table_types(verified_tables),
    }
    logger.info(f"  ✅ Verified {total_verified} tables ({high_conf} high-confidence)")
    
    # ─── Stage 3: Map to Postgres ─────────────────────────
    logger.info("\n[STAGE 3/5] Mapping tables to Postgres...")
    stage_start = time.time()
    
    mapping_summary = extract_and_store(
        verified_tables_by_page=verified_tables,
        ticker=ticker,
        company_name=company_name,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        filing_type=filing_type,
        s3_bucket=settings.s3_bucket,
        s3_key=s3_key,
        period_end_date=period_end_date,
        filing_date=filing_date,
    )
    
    summary["stages"]["map_to_db"] = {
        "duration_sec": round(time.time() - stage_start, 2),
        **mapping_summary,
    }
    logger.info(f"  ✅ Stored financial data + {mapping_summary['segments_inserted']} segments")
    
    # ─── Stage 4: Chunk Narrative ─────────────────────────
    logger.info("\n[STAGE 4/5] Chunking narrative text...")
    stage_start = time.time()
    
    chunks = chunk_document(
        parsed_doc=parsed,
        ticker=ticker,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        filing_type=filing_type,
        filing_id=filing_id,
    )
    
    summary["stages"]["chunk"] = {
        "duration_sec": round(time.time() - stage_start, 2),
        "total_chunks": len(chunks),
        "sections": _count_chunk_sections(chunks),
    }
    logger.info(f"  ✅ Created {len(chunks)} chunks across {len(summary['stages']['chunk']['sections'])} sections")
    
    # ─── Stage 5: Embed + Store ───────────────────────────
    if not skip_embedding:
        logger.info("\n[STAGE 5/5] Embedding + storing chunks...")
        stage_start = time.time()
        
        embed_summary = embed_and_store_chunks(chunks)
        
        summary["stages"]["embed"] = {
            "duration_sec": round(time.time() - stage_start, 2),
            **embed_summary,
        }
        logger.info(f"  ✅ Stored {embed_summary['inserted']} chunks in pgvector")
    else:
        logger.info("\n[STAGE 5/5] Skipping embedding (--skip-embedding flag)")
        summary["stages"]["embed"] = {"skipped": True}
    
    # ─── Optional Neo4j graph stage ───────────────────────
    if not skip_graph:
        from fundlens.ingestion.graph_builder import build_graph_for_filing
        summary["stages"]["graph"] = build_graph_for_filing(
            ticker=ticker, fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter, filing_type=filing_type,
        )
    else:
        summary["stages"]["graph"] = {"skipped": True}

    # ─── Save full summary JSON ───────────────────────────
    summary["total_duration_sec"] = round(time.time() - pipeline_start, 2)
    summary["status"] = "completed"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ PIPELINE COMPLETE in {summary['total_duration_sec']}s")
    logger.info(f"   Summary saved: {output_path}")
    logger.info("=" * 70)
    
    return summary

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _count_table_types(verified_tables: dict) -> dict:
    """Count tables by table_type."""
    counts = {}
    for tables in verified_tables.values():
        for t in tables:
            counts[t.table_type] = counts.get(t.table_type, 0) + 1
    return counts


def _count_chunk_sections(chunks) -> dict:
    """Count chunks by section."""
    counts = {}
    for c in chunks:
        counts[c.section] = counts.get(c.section, 0) + 1
    return counts


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run full ingestion pipeline on one SEC filing")
    parser.add_argument("ticker", help="Stock ticker (e.g., AAPL)")
    parser.add_argument("fiscal_year", type=int, help="Fiscal year (e.g., 2025)")
    parser.add_argument("fiscal_quarter", choices=["Q1", "Q2", "Q3", "Q4", "FY"])
    parser.add_argument("filing_type", choices=["10-K", "10-Q"])
    parser.add_argument("--period-end", type=date.fromisoformat, required=True, help="Actual fiscal period end YYYY-MM-DD")
    parser.add_argument("--filing-date", type=date.fromisoformat, required=True, help="Actual SEC filing date YYYY-MM-DD")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip Stage 5 (useful for fast iteration)")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip Neo4j graph build")
    
    args = parser.parse_args()
    
    try:
        summary = run_pipeline(
            ticker=args.ticker,
            fiscal_year=args.fiscal_year,
            fiscal_quarter=args.fiscal_quarter,
            filing_type=args.filing_type,
            skip_embedding=args.skip_embedding,
            skip_graph=args.skip_graph,
            period_end_date=args.period_end,
            filing_date=args.filing_date,
        )
        print("\n📊 FINAL SUMMARY:")
        print(json.dumps(summary, indent=2, default=str))
        sys.exit(0)
    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
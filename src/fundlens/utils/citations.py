"""
Citation resolution service.

Given any chunk_id, filing_id+page, or financial_metric_id, return a
Citation object with:
  - Source PDF location (S3 key + presigned URL)
  - Page number
  - Thumbnail image (S3 key + presigned URL)
  - Full-resolution page image (S3 key + presigned URL)

Used by:
  - Frontend (display source images alongside answers)
  - API response formatting
  - Audit / debugging
"""

import logging
from datetime import timedelta
from pathlib import PurePosixPath
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import select

from fundlens.storage.s3_client import get_s3_client
from fundlens.config import settings

THUMBNAIL_BUCKET = "fundlens-thumbnails"

def s3_key_for_page(filing_id: str, page_number: int, image_type: str) -> str:
    if page_number < 1 or image_type not in ("thumb", "full"):
        raise ValueError("Invalid page number or image type")
    return f"{filing_id}/{image_type}/page_{page_number:03d}.png"
from fundlens.storage.db import get_session
from fundlens.storage.models import (
    DocumentChunk, FinancialMetric, Filing, FilingPeriod, Company,
)

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

PRESIGNED_URL_TTL = timedelta(hours=1)
FILINGS_BUCKET = "fundlens-filings"


# ============================================================
# CITATION MODEL
# ============================================================

class Citation(BaseModel):
    """A fully resolved citation pointing to a specific page in a filing."""

    # Identity
    filing_id: str
    company_ticker: str
    filing_type: str
    fiscal_year: int
    fiscal_quarter: str
    page_number: int

    # Source PDF
    pdf_s3_bucket: str
    pdf_s3_key: str
    pdf_url: Optional[str] = Field(None, description="Presigned URL (1h TTL)")

    # Page images
    thumbnail_s3_key: str
    thumbnail_url: Optional[str] = Field(None, description="Presigned URL (1h TTL)")
    full_image_s3_key: str
    full_image_url: Optional[str] = Field(None, description="Presigned URL (1h TTL)")

    # Optional: the actual content that led to this citation
    snippet: Optional[str] = None
    source_type: Optional[str] = Field(None, description="'chunk', 'financial_metric', 'segment'")


# ============================================================
# PRESIGNED URL HELPERS
# ============================================================

def presign_url(bucket: str, key: str, expires_seconds: int = 3600) -> Optional[str]:
    """Generate a presigned GET URL for an S3 object. Returns None on failure."""
    try:
        s3 = get_s3_client()
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )
    except Exception as e:
        logger.warning(f"  Could not presign {bucket}/{key}: {e}")
        return None


def build_filing_id_from_db_filing(filing: Filing, period: FilingPeriod, company: Company) -> str:
    """Build the standard filing_id from Postgres entities."""
    return f"{company.ticker}__{period.filing_type}__{period.fiscal_quarter}_{period.fiscal_year}"


# ============================================================
# CORE RESOLVER
# ============================================================

def build_citation(
    *,
    filing_id: str,
    page_number: int,
    company_ticker: str,
    filing_type: str,
    fiscal_year: int,
    fiscal_quarter: str,
    pdf_s3_bucket: str,
    pdf_s3_key: str,
    snippet: Optional[str] = None,
    source_type: Optional[str] = None,
    include_presigned: bool = True,
) -> Citation:
    """Low-level builder that takes all required pieces and assembles a Citation."""

    pdf_stem = PurePosixPath(pdf_s3_key).stem
    thumb_key = f"thumbnails/{company_ticker}/{pdf_stem}/page_{page_number:03d}.png"
    full_key = ""  # Parser creates 72-DPI thumbnails, not full-resolution page images

    citation = Citation(
        filing_id=filing_id,
        company_ticker=company_ticker,
        filing_type=filing_type,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        page_number=page_number,
        pdf_s3_bucket=pdf_s3_bucket,
        pdf_s3_key=pdf_s3_key,
        thumbnail_s3_key=thumb_key,
        full_image_s3_key=full_key,
        snippet=snippet,
        source_type=source_type,
    )

    if include_presigned:
        citation.pdf_url = presign_url(pdf_s3_bucket, pdf_s3_key)
        citation.thumbnail_url = presign_url(THUMBNAIL_BUCKET, thumb_key)
        citation.full_image_url = None

    return citation


# ============================================================
# LOOKUP BY CHUNK
# ============================================================

def get_citation_for_chunk(chunk_id: str, include_presigned: bool = True) -> Optional[Citation]:
    """Resolve a citation from a chunk_id."""
    with get_session() as s:
        chunk = s.get(DocumentChunk, chunk_id)
        if chunk is None:
            logger.warning(f"No chunk found: {chunk_id}")
            return None

        # We need the PDF's S3 location from the filings table
        company = s.execute(
            select(Company).where(Company.ticker == chunk.company_ticker)
        ).scalar_one()
        period = s.execute(
            select(FilingPeriod).join(Filing, Filing.period_id == FilingPeriod.id).where(
                Filing.company_id == company.id,
                FilingPeriod.fiscal_year == chunk.fiscal_year,
                FilingPeriod.fiscal_quarter == chunk.fiscal_quarter,
                FilingPeriod.filing_type == chunk.filing_type,
            )
        ).scalar_one_or_none()

        if period is None:
            logger.warning(f"No period found for chunk {chunk_id}")
            return None

        filing = s.execute(
            select(Filing).where(
                Filing.company_id == company.id,
                Filing.period_id == period.id,
            )
        ).scalar_one_or_none()

        if filing is None:
            logger.warning(f"No filing in DB for chunk {chunk_id}")
            return None

        filing_id = build_filing_id_from_db_filing(filing, period, company)

        return build_citation(
            filing_id=filing_id,
            page_number=chunk.page_number,
            company_ticker=company.ticker,
            filing_type=period.filing_type,
            fiscal_year=period.fiscal_year,
            fiscal_quarter=period.fiscal_quarter,
            pdf_s3_bucket=filing.s3_bucket,
            pdf_s3_key=filing.s3_key,
            snippet=chunk.raw_content[:300],
            source_type="chunk",
            include_presigned=include_presigned,
        )


# ============================================================
# LOOKUP BY FINANCIAL METRIC
# ============================================================

def get_citation_for_financial_metric(
    metric_id: int,
    include_presigned: bool = True,
) -> Optional[Citation]:
    """Resolve a citation from a financial_metric_id."""
    with get_session() as s:
        metric = s.get(FinancialMetric, metric_id)
        if metric is None:
            logger.warning(f"No financial_metric found: id={metric_id}")
            return None
        if metric.source_page_number is None:
            logger.warning(f"financial_metric {metric_id} has no source_page_number")
            return None

        filing = s.get(Filing, metric.filing_id)
        period = s.get(FilingPeriod, metric.period_id)
        company = s.get(Company, metric.company_id)
        filing_id = build_filing_id_from_db_filing(filing, period, company)

        snippet = f"financial_metric row id={metric_id} | revenue={metric.revenue}"

        return build_citation(
            filing_id=filing_id,
            page_number=metric.source_page_number,
            company_ticker=company.ticker,
            filing_type=period.filing_type,
            fiscal_year=period.fiscal_year,
            fiscal_quarter=period.fiscal_quarter,
            pdf_s3_bucket=filing.s3_bucket,
            pdf_s3_key=filing.s3_key,
            snippet=snippet,
            source_type="financial_metric",
            include_presigned=include_presigned,
        )


# ============================================================
# LOOKUP BY EXPLICIT (FILING_ID, PAGE)
# ============================================================

def get_citation(
    filing_id: str,
    page_number: int,
    include_presigned: bool = True,
) -> Optional[Citation]:
    """
    Spec-compliant resolver: takes filing_id + page, returns Citation.
    
    filing_id format: {TICKER}__{FILING_TYPE}__{QUARTER}_{YEAR}
    e.g., GOOGL__10-Q__Q1_2025
    """
    # Parse the filing_id
    try:
        ticker, filing_type, period_part = filing_id.split("__")
        quarter, year_str = period_part.rsplit("_", 1)
        fiscal_year = int(year_str)
    except (ValueError, AttributeError):
        logger.warning(f"Invalid filing_id format: {filing_id}")
        return None

    with get_session() as s:
        company = s.execute(
            select(Company).where(Company.ticker == ticker)
        ).scalar_one_or_none()
        if company is None:
            return None

        period = s.execute(
            select(FilingPeriod).join(Filing, Filing.period_id == FilingPeriod.id).where(
                Filing.company_id == company.id,
                FilingPeriod.fiscal_year == fiscal_year,
                FilingPeriod.fiscal_quarter == quarter,
                FilingPeriod.filing_type == filing_type,
            )
        ).scalar_one_or_none()
        if period is None:
            return None

        filing = s.execute(
            select(Filing).where(
                Filing.company_id == company.id,
                Filing.period_id == period.id,
            )
        ).scalar_one_or_none()
        if filing is None:
            return None

        return build_citation(
            filing_id=filing_id,
            page_number=page_number,
            company_ticker=ticker,
            filing_type=filing_type,
            fiscal_year=fiscal_year,
            fiscal_quarter=quarter,
            pdf_s3_bucket=filing.s3_bucket,
            pdf_s3_key=filing.s3_key,
            source_type="explicit",
            include_presigned=include_presigned,
        )
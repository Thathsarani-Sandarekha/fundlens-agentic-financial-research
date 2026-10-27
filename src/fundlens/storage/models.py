"""
SQLAlchemy ORM models that mirror storage/schema.sql.

Each class = one Postgres table.
Use these throughout the codebase for type-safe DB access.

Example:
    from storage.models import Company, FinancialMetric
    company = Company(ticker="AAPL", name="Apple Inc.")
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional, List

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from fundlens.storage.db import Base
from pgvector.sqlalchemy import Vector

# ============================================================
# ENUMS — mirror the Postgres ENUM types from schema.sql
# ============================================================
# Note: native_enum=False means we use the existing PG enums
# rather than letting SQLAlchemy create new ones.

FILING_TYPES = ("10-K", "10-Q", "8-K")
FISCAL_QUARTERS = ("Q1", "Q2", "Q3", "Q4", "FY")
MARKET_CAP_BUCKETS = ("mega_cap", "large_cap", "mid_cap", "small_cap")
SENTIMENTS = ("positive", "neutral", "negative", "mixed")
COMMENTARY_TOPICS = (
    "revenue", "earnings", "profitability", "guidance",
    "segments", "macroeconomic", "product", "competition",
    "ai_strategy", "cost_management", "capital_allocation",
    "risk_factor", "other",
)


# ============================================================
# TABLE: companies
# ============================================================

class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(10), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sector: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    market_cap_bucket: Mapped[Optional[str]] = mapped_column(
        Enum(*MARKET_CAP_BUCKETS, name="market_cap_bucket_enum", create_type=False)
    )
    cik: Mapped[Optional[str]] = mapped_column(String(20))
    fiscal_year_end_month: Mapped[Optional[int]] = mapped_column(SmallInteger)
    headquarters_country: Mapped[str] = mapped_column(String(50), default="USA")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships (back-references)
    filings: Mapped[List["Filing"]] = relationship(back_populates="company", cascade="all, delete-orphan")
    financial_metrics: Mapped[List["FinancialMetric"]] = relationship(back_populates="company", cascade="all, delete-orphan")
    segment_metrics: Mapped[List["SegmentMetric"]] = relationship(back_populates="company", cascade="all, delete-orphan")
    commentary: Mapped[List["ExecutiveCommentary"]] = relationship(back_populates="company", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Company {self.ticker}>"


# ============================================================
# TABLE: filing_periods
# ============================================================

class FilingPeriod(Base):
    __tablename__ = "filing_periods"
    __table_args__ = (
        UniqueConstraint("fiscal_year", "fiscal_quarter", "filing_type", "period_end_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fiscal_year: Mapped[int] = mapped_column(SmallInteger, nullable=False, index=True)
    fiscal_quarter: Mapped[str] = mapped_column(
        Enum(*FISCAL_QUARTERS, name="fiscal_quarter_enum", create_type=False),
        nullable=False,
        index=True,
    )
    period_end_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    filing_type: Mapped[str] = mapped_column(
        Enum(*FILING_TYPES, name="filing_type_enum", create_type=False),
        nullable=False,
    )
    period_label: Mapped[Optional[str]] = mapped_column(String(20))  # auto-set by trigger
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    filings: Mapped[List["Filing"]] = relationship(back_populates="period", cascade="all, delete-orphan")
    financial_metrics: Mapped[List["FinancialMetric"]] = relationship(back_populates="period", cascade="all, delete-orphan")
    segment_metrics: Mapped[List["SegmentMetric"]] = relationship(back_populates="period", cascade="all, delete-orphan")
    commentary: Mapped[List["ExecutiveCommentary"]] = relationship(back_populates="period", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<FilingPeriod {self.period_label}>"


# ============================================================
# TABLE: filings
# ============================================================

class Filing(Base):
    __tablename__ = "filings"
    __table_args__ = (
        UniqueConstraint("company_id", "period_id"),
        UniqueConstraint("s3_bucket", "s3_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    period_id: Mapped[int] = mapped_column(
        ForeignKey("filing_periods.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filing_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    accession_number: Mapped[Optional[str]] = mapped_column(String(30))
    s3_bucket: Mapped[str] = mapped_column(String(100), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(500), nullable=False)
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    sha256_hash: Mapped[Optional[str]] = mapped_column(String(64))
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    company: Mapped["Company"] = relationship(back_populates="filings")
    period: Mapped["FilingPeriod"] = relationship(back_populates="filings")
    financial_metrics: Mapped[List["FinancialMetric"]] = relationship(back_populates="filing", cascade="all, delete-orphan")
    segment_metrics: Mapped[List["SegmentMetric"]] = relationship(back_populates="filing", cascade="all, delete-orphan")
    commentary: Mapped[List["ExecutiveCommentary"]] = relationship(back_populates="filing", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Filing id={self.id} s3_key={self.s3_key}>"


# ============================================================
# TABLE: financial_metrics
# ============================================================

class FinancialMetric(Base):
    __tablename__ = "financial_metrics"
    __table_args__ = (UniqueConstraint("company_id", "period_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    period_id: Mapped[int] = mapped_column(
        ForeignKey("filing_periods.id", ondelete="CASCADE"), nullable=False
    )
    filing_id: Mapped[int] = mapped_column(
        ForeignKey("filings.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Income statement (millions USD)
    revenue: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    cost_of_revenue: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    gross_profit: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    gross_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    operating_expenses: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    rd_expense: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    sga_expense: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    operating_income: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    operating_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))
    net_income: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    net_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))

    # Per-share (actual dollars)
    eps_basic: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    eps_diluted: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    shares_outstanding_basic: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    shares_outstanding_diluted: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))

    # Balance sheet (millions USD)
    total_assets: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    total_liabilities: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    total_equity: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    cash_and_equivalents: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    total_debt: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))

    # Cash flow (millions USD)
    operating_cash_flow: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    capex: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    free_cash_flow: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))

    # Pre-computed YoY growth
    revenue_yoy_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), index=True)
    net_income_yoy_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    eps_yoy_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), index=True)
    operating_income_yoy_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    free_cash_flow_yoy_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))

    # Pre-computed QoQ growth
    revenue_qoq_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    eps_qoq_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))

    # Metadata
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    is_restated: Mapped[bool] = mapped_column(Boolean, default=False)
    source_page_number: Mapped[Optional[int]] = mapped_column(Integer)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    company: Mapped["Company"] = relationship(back_populates="financial_metrics")
    period: Mapped["FilingPeriod"] = relationship(back_populates="financial_metrics")
    filing: Mapped["Filing"] = relationship(back_populates="financial_metrics")

    def __repr__(self) -> str:
        return f"<FinancialMetric company={self.company_id} period={self.period_id} revenue={self.revenue}>"


# ============================================================
# TABLE: segment_metrics
# ============================================================

class SegmentMetric(Base):
    __tablename__ = "segment_metrics"
    __table_args__ = (UniqueConstraint("company_id", "period_id", "segment_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    period_id: Mapped[int] = mapped_column(
        ForeignKey("filing_periods.id", ondelete="CASCADE"), nullable=False
    )
    filing_id: Mapped[int] = mapped_column(
        ForeignKey("filings.id", ondelete="CASCADE"), nullable=False
    )

    segment_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    segment_name_raw: Mapped[Optional[str]] = mapped_column(String(255))
    segment_type: Mapped[Optional[str]] = mapped_column(String(50))

    segment_revenue: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    segment_operating_income: Mapped[Optional[Decimal]] = mapped_column(Numeric(18, 2))
    segment_operating_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(6, 4))

    segment_revenue_yoy_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), index=True)
    segment_revenue_qoq_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))

    source_page_number: Mapped[Optional[int]] = mapped_column(Integer)
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    company: Mapped["Company"] = relationship(back_populates="segment_metrics")
    period: Mapped["FilingPeriod"] = relationship(back_populates="segment_metrics")
    filing: Mapped["Filing"] = relationship(back_populates="segment_metrics")

    def __repr__(self) -> str:
        return f"<SegmentMetric {self.segment_name} revenue={self.segment_revenue}>"


# ============================================================
# TABLE: executive_commentary
# ============================================================

class ExecutiveCommentary(Base):
    __tablename__ = "executive_commentary"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("companies.id", ondelete="CASCADE"), nullable=False
    )
    period_id: Mapped[int] = mapped_column(
        ForeignKey("filing_periods.id", ondelete="CASCADE"), nullable=False
    )
    filing_id: Mapped[int] = mapped_column(
        ForeignKey("filings.id", ondelete="CASCADE"), nullable=False
    )

    executive_name: Mapped[Optional[str]] = mapped_column(String(100))
    executive_title: Mapped[Optional[str]] = mapped_column(String(100))
    topic: Mapped[str] = mapped_column(
        Enum(*COMMENTARY_TOPICS, name="commentary_topic_enum", create_type=False),
        nullable=False,
        index=True,
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    full_quote: Mapped[Optional[str]] = mapped_column(Text)
    sentiment: Mapped[Optional[str]] = mapped_column(
        Enum(*SENTIMENTS, name="sentiment_enum", create_type=False),
        index=True,
    )

    source_page_number: Mapped[Optional[int]] = mapped_column(Integer)
    chunk_id: Mapped[Optional[str]] = mapped_column(String(64))
    extracted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    company: Mapped["Company"] = relationship(back_populates="commentary")
    period: Mapped["FilingPeriod"] = relationship(back_populates="commentary")
    filing: Mapped["Filing"] = relationship(back_populates="commentary")

    def __repr__(self) -> str:
        return f"<ExecutiveCommentary {self.executive_name}: {self.topic}>"
    

# ============================================================
# TABLE: document_chunks (pgvector)
# ============================================================

class DocumentChunk(Base):
    """Embeddings + metadata for narrative text chunks."""
    __tablename__ = "document_chunks"

    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)

    # Filing identity
    filing_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    company_ticker: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    filing_type: Mapped[str] = mapped_column(Text, nullable=False)
    fiscal_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fiscal_quarter: Mapped[str] = mapped_column(Text, nullable=False)

    # Location
    section: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False)
    overlap_with_prev: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk {self.chunk_id[:40]}...>"
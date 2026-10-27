"""
Map VerifiedTable → Postgres financial_metrics / segment_metrics rows.

Workflow:
  1. Take a VerifiedTable (cleaned by TODO 2.3)
  2. Use LLM + Instructor to map row labels to schema columns
  3. Pick the right period column (e.g., "Q1 2025")
  4. Compute derived fields (margins, growth rates)
  5. Insert/update Postgres
  6. Compare against ground truth, log discrepancies

Usage:
    from ingestion.financial_mapper import extract_and_store
    result = extract_and_store(
        verified_tables,
        ticker="GOOGL",
        fiscal_year=2025,
        fiscal_quarter="Q1",
        filing_type="10-Q",
        s3_key="filings/GOOGL/GOOGL_2025_Q1_10Q.pdf",
        period_end_date=date(2025, 3, 31),
        filing_date=date(2025, 4, 25),
    )
"""

import logging
from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import select

from fundlens.config import settings
from fundlens.storage.db import get_session
from fundlens.storage.models import (
    Company, FilingPeriod, Filing,
    FinancialMetric, SegmentMetric,
)
from fundlens.models.verified import VerifiedTable
from fundlens.llm.groq_client import get_instructor_client, call_with_backoff

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

MAPPING_MODEL = settings.groq_model_default


# ============================================================
# PYDANTIC SCHEMA — what we want LLM to return
# ============================================================

class FinancialMetricMapping(BaseModel):
    """
    LLM extracts these specific fields from a financial table.
    Each field is for ONE specific period (the one we're targeting).
    All monetary values in millions USD (the schema's expected unit).
    """

    # Income statement
    revenue: Optional[float] = Field(None, description="Total revenue / net sales")
    cost_of_revenue: Optional[float] = Field(None, description="Cost of revenue / cost of goods sold")
    gross_profit: Optional[float] = Field(None, description="Gross profit (revenue - cost of revenue)")
    rd_expense: Optional[float] = Field(None, description="Research and development expenses")
    sga_expense: Optional[float] = Field(None, description="Selling, general and administrative")
    operating_income: Optional[float] = Field(None, description="Operating income / income from operations")
    net_income: Optional[float] = Field(None, description="Net income / net earnings")

    # Per-share (actual dollars, not millions)
    eps_basic: Optional[float] = Field(None, description="Basic earnings per share, in actual dollars")
    eps_diluted: Optional[float] = Field(None, description="Diluted earnings per share, in actual dollars")

    # Balance sheet (when applicable)
    total_assets: Optional[float] = Field(None, description="Total assets")
    total_liabilities: Optional[float] = Field(None, description="Total liabilities")
    total_equity: Optional[float] = Field(None, description="Total stockholders equity")
    cash_and_equivalents: Optional[float] = Field(None, description="Cash and cash equivalents")

    # Cash flow (when applicable)
    operating_cash_flow: Optional[float] = Field(None, description="Net cash from operating activities")
    capex: Optional[float] = Field(None, description="Capital expenditures (as positive number)")

    # Metadata
    period_label_used: str = Field(..., description="Which column heading we extracted from, e.g., 'Q1 2025'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: Optional[str] = Field(None, description="Any issues, e.g., 'no balance sheet data in this table'")


class SegmentMapping(BaseModel):
    """One segment extracted from a segment table."""
    segment_name_raw: str = Field(..., description="Original segment name from filing")
    segment_name_standardized: str = Field(
        ...,
        description="Reported business line (keep Intelligent Cloud distinct from Azure), Google Cloud, iPhone, etc.",
    )
    segment_revenue: Optional[float] = Field(None, description="Segment revenue in millions USD")
    segment_operating_income: Optional[float] = Field(None, description="Segment operating income in millions USD")


class SegmentMappingList(BaseModel):
    segments: list[SegmentMapping] = Field(default_factory=list)
    period_label_used: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    notes: Optional[str] = None


# ============================================================
# PROMPT BUILDER FOR FINANCIAL MAPPING
# ============================================================

FIN_SYSTEM_PROMPT = """You map cleaned financial table data to database schema fields.

═══════════════════════════════════════════════════════════════
🚨 RULE #1: DO NOT TRANSFORM VALUES — COPY THEM AS-IS 🚨
═══════════════════════════════════════════════════════════════

The numeric values in the input have ALREADY been cleaned and parsed
to floats. They are already in MILLIONS USD.

Your ONLY job is to:
  1. Find the right row label (e.g., "Revenues" → revenue field)
  2. Find the right period column (e.g., "Q1 2025")
  3. COPY the float value as-is to that field

DO NOT:
  ❌ Multiply by 1000
  ❌ Divide by 1000
  ❌ Convert units
  ❌ Reformat numbers
  ❌ Round
  ❌ Modify decimal places

If the input shows: `'Cash and cash equivalents': {'Q1 2025': 23466.0}`
You output: cash_and_equivalents=23466.0    ← EXACT COPY

If the input shows: `'Total assets': {'Q1 2025': 475374.0}`
You output: total_assets=475374.0           ← EXACT COPY

NEVER output 23.466 or 475.374 — that would be WRONG.

═══════════════════════════════════════════════════════════════
SCHEMA MAPPING (just match labels to fields)
═══════════════════════════════════════════════════════════════

1. Extract values for THE TARGET PERIOD only. Ignore other periods.
2. If a metric isn't in the table, return null. Never guess.
3. EPS values stay in actual dollars (single-digit decimals are normal for EPS).

ROW LABEL MAPPING:
  "Total revenues" / "Net sales" / "Revenues" → revenue
  "Cost of revenues" / "Cost of goods sold" → cost_of_revenue
  "Gross profit" → gross_profit
  "Research and development" → rd_expense
  "Selling, general and administrative" → sga_expense
  "Income from operations" / "Operating income" → operating_income
  "Net income" / "Net earnings" → net_income
  "Diluted" + "per share" → eps_diluted
  "Basic" + "per share" → eps_basic
  "Total assets" → total_assets
  "Total liabilities" → total_liabilities
  "Total stockholders' equity" / "Total equity" → total_equity
  "Cash and cash equivalents" → cash_and_equivalents
  "Net cash provided by operating activities" → operating_cash_flow
  "Purchases of property and equipment" / "Capital expenditures" → capex (as positive number)
"""


SEGMENT_SYSTEM_PROMPT = """You extract business segment data from financial filings.

═══════════════════════════════════════════════════════════════
🚨 CRITICAL RULE: ONE TOOL CALL ONLY 🚨
═══════════════════════════════════════════════════════════════

You MUST call the SegmentMappingList tool EXACTLY ONCE.
Include ALL segments in the single 'segments' list inside that one call.

❌ WRONG — Multiple tool calls (one per segment):
   Call 1: SegmentMappingList(segments=[iPhone])
   Call 2: SegmentMappingList(segments=[Mac])
   Call 3: SegmentMappingList(segments=[iPad])

✅ CORRECT — Single tool call with all segments:
   Call 1: SegmentMappingList(segments=[iPhone, Mac, iPad, Services, ...])

═══════════════════════════════════════════════════════════════
WHAT TO EXTRACT
═══════════════════════════════════════════════════════════════

For EACH business segment in the table, create one SegmentMapping with:
- segment_name_raw: name exactly as it appears in the filing
- segment_name_standardized: canonical name (see mapping below)
- segment_revenue: revenue for the target period (millions USD, copy as-is)
- segment_operating_income: operating income for target period (or null)

═══════════════════════════════════════════════════════════════
NAME STANDARDIZATION
═══════════════════════════════════════════════════════════════

Apple:
  "iPhone" → "iPhone"
  "Mac" → "Mac"
  "iPad" → "iPad"
  "Wearables, Home and Accessories" → "Wearables"
  "Services" → "Services"

Microsoft:
  "Productivity and Business Processes" → "Productivity"
  "Intelligent Cloud" → "Intelligent Cloud"
  "More Personal Computing" → "Personal Computing"

Alphabet/Google:
  "Google Services" → "Google Services"
  "Google Cloud" → "Google Cloud"
  "Other Bets" → "Other Bets"

Amazon:
  "North America" → "North America"
  "International" → "International"
  "AWS" → "AWS"

═══════════════════════════════════════════════════════════════
SKIP THESE (NOT REAL SEGMENTS)
═══════════════════════════════════════════════════════════════

❌ "Total net sales" / "Total revenue" — this is a sum, not a segment
❌ "Deferred revenue" — accounting concept, not a segment
❌ Row labels that are clearly totals or footnotes

═══════════════════════════════════════════════════════════════
VALUE RULES (SAME AS FINANCIAL MAPPING)
═══════════════════════════════════════════════════════════════

1. Values are already in millions USD — copy as-is
2. Do NOT multiply, divide, or transform numbers
3. If a value is missing, use null (don't guess)
4. Extract values for the TARGET PERIOD only
"""


def build_financial_prompt(table: VerifiedTable, target_period: str) -> str:
    rows_str = "\n".join(
        f"  '{row.label}': {row.values}"
        for row in table.rows[:25]
    )

    return f"""TARGET PERIOD: {target_period}
TABLE TYPE: {table.table_type}
TABLE HEADERS: {table.headers}

ROWS (values are ALREADY in millions USD — copy them exactly as shown):
{rows_str}

Extract financial metrics for the TARGET PERIOD ({target_period}).
Copy the numeric values EXACTLY as shown — do not multiply, divide, or transform.
Return null for fields not present in this table.
"""


def build_segment_prompt(table: VerifiedTable, target_period: str) -> str:
    """Build prompt for extracting segment data."""
    rows_str = "\n".join(
        f"  '{row.label}': {row.values}"
        for row in table.rows[:30]
    )

    return f"""TARGET PERIOD: {target_period}
TABLE TYPE: {table.table_type}
TABLE UNITS: {table.units} {table.currency}
TABLE HEADERS: {table.headers}

ROWS:
{rows_str}

TASK:
Extract ALL business segments from this table in ONE tool call.

Return a single SegmentMappingList containing:
- segments: a LIST of ALL segments (iPhone, Mac, iPad, Services, etc.)
- period_label_used: which column header you read from
- confidence: 0.0-1.0
- notes: any issues

DO NOT call the tool multiple times. Put ALL segments in the segments array.

For each segment, extract revenue for {target_period}.
Skip rows that are totals ("Total net sales") or accounting concepts ("Deferred revenue").
"""

# ============================================================
# LLM EXTRACTION FUNCTIONS
# ============================================================

def extract_financial_metrics(
    table: VerifiedTable,
    target_period: str,
) -> Optional[FinancialMetricMapping]:
    """Use LLM to extract financial metric values from a table."""
    try:
        client = get_instructor_client()
        result: FinancialMetricMapping = call_with_backoff(
        client,
        model=MAPPING_MODEL,
        response_model=FinancialMetricMapping,
        messages=[
                {"role": "system", "content": FIN_SYSTEM_PROMPT},
                {"role": "user", "content": build_financial_prompt(table, target_period)},
            ],
        max_retries=2,
        temperature=0.0,
        max_tokens=2048,
    )
        return result
    except Exception as e:
        logger.error(f"  ❌ Financial mapping failed: {e}")
        return None


def extract_segments(
    table: VerifiedTable,
    target_period: str,
) -> Optional[SegmentMappingList]:
    """Extract segment-level data from a segment table."""
    try:
        client = get_instructor_client()
        result: SegmentMappingList = call_with_backoff(
            client,
            model=MAPPING_MODEL,
            response_model=SegmentMappingList,
            messages=[
                {"role": "system", "content": SEGMENT_SYSTEM_PROMPT},
                {"role": "user", "content": build_segment_prompt(table, target_period)},
            ],
            max_retries=2,
            temperature=0.0,
            max_tokens=2048,
        )
        return result
    except Exception as e:
        logger.error(f"  ❌ Segment mapping failed: {e}")
        return None


# ============================================================
# DATABASE OPERATIONS
# ============================================================

def get_or_create_company(session, ticker: str, name: str = None) -> Company:
    """Get company by ticker, creating if needed."""
    company = session.execute(
        select(Company).where(Company.ticker == ticker)
    ).scalar_one_or_none()

    if company is None:
        company = Company(
            ticker=ticker,
            name=name or ticker,
            sector="Technology",  # default, should be enriched
            market_cap_bucket="mega_cap",
        )
        session.add(company)
        session.flush()
        logger.info(f"  + Created company: {ticker}")

    return company


def get_or_create_period(
    session, fiscal_year: int, fiscal_quarter: str,
    period_end_date: date, filing_type: str,
) -> FilingPeriod:
    """Get period, creating if needed."""
    period = session.execute(
        select(FilingPeriod).where(
            FilingPeriod.fiscal_year == fiscal_year,
            FilingPeriod.fiscal_quarter == fiscal_quarter,
            FilingPeriod.filing_type == filing_type,
            FilingPeriod.period_end_date == period_end_date,
        )
    ).scalar_one_or_none()

    if period is None:
        period = FilingPeriod(
            fiscal_year=fiscal_year,
            fiscal_quarter=fiscal_quarter,
            period_end_date=period_end_date,
            filing_type=filing_type,
        )
        session.add(period)
        session.flush()
        logger.info(f"  + Created period: {fiscal_quarter} {fiscal_year}")

    return period


def get_or_create_filing(
    session, company_id: int, period_id: int,
    filing_date: date, s3_bucket: str, s3_key: str,
) -> Filing:
    """Get filing, creating if needed."""
    filing = session.execute(
        select(Filing).where(
            Filing.company_id == company_id,
            Filing.period_id == period_id,
        )
    ).scalar_one_or_none()

    if filing is None:
        filing = Filing(
            company_id=company_id,
            period_id=period_id,
            filing_date=filing_date,
            s3_bucket=s3_bucket,
            s3_key=s3_key,
        )
        session.add(filing)
        session.flush()
        logger.info(f"  + Created filing: {s3_key}")

    return filing


def upsert_financial_metric(
    session, company_id: int, period_id: int, filing_id: int,
    mapping: FinancialMetricMapping, source_page: int,
) -> FinancialMetric:
    """
    Insert or update a financial_metrics row.
    
    If a row already exists for (company, period), we UPDATE non-null fields
    only — so balance sheet data added later doesn't wipe income statement data.
    """
    existing = session.execute(
        select(FinancialMetric).where(
            FinancialMetric.company_id == company_id,
            FinancialMetric.period_id == period_id,
        )
    ).scalar_one_or_none()

    # Helper: convert float → Decimal or None
    def D(v):
        return Decimal(str(v)) if v is not None else None

    if existing is None:
        metric = FinancialMetric(
            company_id=company_id,
            period_id=period_id,
            filing_id=filing_id,
            revenue=D(mapping.revenue),
            cost_of_revenue=D(mapping.cost_of_revenue),
            gross_profit=D(mapping.gross_profit),
            rd_expense=D(mapping.rd_expense),
            sga_expense=D(mapping.sga_expense),
            operating_income=D(mapping.operating_income),
            net_income=D(mapping.net_income),
            eps_basic=D(mapping.eps_basic),
            eps_diluted=D(mapping.eps_diluted),
            total_assets=D(mapping.total_assets),
            total_liabilities=D(mapping.total_liabilities),
            total_equity=D(mapping.total_equity),
            cash_and_equivalents=D(mapping.cash_and_equivalents),
            operating_cash_flow=D(mapping.operating_cash_flow),
            capex=D(mapping.capex),
            source_page_number=source_page,
        )
        session.add(metric)
        session.flush()
        logger.info(f"  + Inserted financial_metrics id={metric.id}")
        return metric
    else:
        # Only update fields where:
        # 1. New value is non-null, AND
        # 2. Existing value is NULL (don't overwrite existing good data)
        # This prevents later partial tables from corrupting earlier complete extractions.
        updates = []
        for field in [
            "revenue", "cost_of_revenue", "gross_profit",
            "rd_expense", "sga_expense", "operating_income", "net_income",
            "eps_basic", "eps_diluted",
            "total_assets", "total_liabilities", "total_equity", "cash_and_equivalents",
            "operating_cash_flow", "capex",
        ]:
            new_val = getattr(mapping, field)
            existing_val = getattr(existing, field)
            if new_val is not None and existing_val is None:
                setattr(existing, field, D(new_val))
                updates.append(field)
            elif new_val is not None and existing_val is not None:
                # Don't overwrite — log it instead so we can review
                if D(new_val) != existing_val:
                    logger.debug(
                        f"  ⏭️  Preserving existing {field}={existing_val} "
                        f"(would have been overwritten with {new_val})"
                    )

        if updates:
            logger.info(f"  ~ Updated financial_metrics id={existing.id}: {', '.join(updates)}")

        return existing


def compute_derived_fields(session, metric: FinancialMetric):
    """Compute gross_margin, operating_margin, net_margin, free_cash_flow."""
    updated = []

    if metric.gross_profit is not None and metric.revenue is not None and metric.revenue > 0:
        metric.gross_margin = (metric.gross_profit / metric.revenue).quantize(Decimal("0.0001"))
        updated.append("gross_margin")

    if metric.operating_income is not None and metric.revenue is not None and metric.revenue > 0:
        metric.operating_margin = (metric.operating_income / metric.revenue).quantize(Decimal("0.0001"))
        updated.append("operating_margin")

    if metric.net_income is not None and metric.revenue is not None and metric.revenue > 0:
        metric.net_margin = (metric.net_income / metric.revenue).quantize(Decimal("0.0001"))
        updated.append("net_margin")

    if metric.operating_cash_flow is not None and metric.capex is not None:
        # capex is typically reported as negative (outflow), so add it
        # but some filings report it as positive — we treat as positive expense
        metric.free_cash_flow = metric.operating_cash_flow - metric.capex
        updated.append("free_cash_flow")

    if updated:
        logger.info(f"  Δ Computed: {', '.join(updated)}")


def compute_yoy_growth(session, metric: FinancialMetric, company_id: int):
    """
    Compare current metric to the same quarter previous year.
    Only computes if prior-year data exists in DB.
    """
    # Fetch current period
    current_period = session.get(FilingPeriod, metric.period_id)

    # Find same quarter, previous year
    prior_period = session.execute(
        select(FilingPeriod).join(Filing, Filing.period_id == FilingPeriod.id).where(
            Filing.company_id == company_id,
            FilingPeriod.fiscal_year == current_period.fiscal_year - 1,
            FilingPeriod.fiscal_quarter == current_period.fiscal_quarter,
            FilingPeriod.filing_type == current_period.filing_type,
        )
    ).scalar_one_or_none()

    if prior_period is None:
        logger.debug("  No prior-year period found — skipping YoY")
        return

    prior_metric = session.execute(
        select(FinancialMetric).where(
            FinancialMetric.company_id == company_id,
            FinancialMetric.period_id == prior_period.id,
        )
    ).scalar_one_or_none()

    if prior_metric is None:
        logger.debug("  No prior-year metric found — skipping YoY")
        return

    updated = []

    def growth(curr, prev):
        if curr is not None and prev is not None and prev != 0:
            return ((curr - prev) / abs(prev)).quantize(Decimal("0.0001"))
        return None

    metric.revenue_yoy_growth = growth(metric.revenue, prior_metric.revenue)
    if metric.revenue_yoy_growth is not None:
        updated.append("revenue_yoy_growth")

    metric.net_income_yoy_growth = growth(metric.net_income, prior_metric.net_income)
    if metric.net_income_yoy_growth is not None:
        updated.append("net_income_yoy_growth")

    metric.eps_yoy_growth = growth(metric.eps_diluted, prior_metric.eps_diluted)
    if metric.eps_yoy_growth is not None:
        updated.append("eps_yoy_growth")

    if updated:
        logger.info(f"  📈 YoY growth computed: {', '.join(updated)}")



def normalize_verified_financial_table(table: VerifiedTable) -> VerifiedTable:
    """Convert monetary table cells to millions of USD; never scale per-share values.

    This is intentionally conservative. Currency conversion is not performed.
    Row values are preserved on a deep copy so the original verified extraction
    remains available for comparison and audit.
    """
    if table.currency != "USD":
        raise ValueError(f"Unsupported filing currency: {table.currency}")
    scale = {"actual": 1e-6, "thousands": 1e-3, "millions": 1.0, "billions": 1e3}[table.units]
    normalized = table.model_copy(deep=True)
    for row in normalized.rows:
        label = row.label.casefold()
        is_per_share = any(x in label for x in ("per share", "earnings per share", "eps"))
        if not is_per_share:
            row.values = {period: value * scale if value is not None else None
                          for period, value in row.values.items()}
    normalized.units = "millions"
    return normalized


def validate_mapping_provenance(mapping: FinancialMetricMapping, table: VerifiedTable, target_period: str) -> bool:
    """Reject obviously mismatched period and low-confidence LLM extractions."""
    if mapping.confidence < 0.70:
        return False
    period_used = mapping.period_label_used.strip()
    available = {h.strip() for h in table.headers}
    for row in table.rows:
        available.update(str(key).strip() for key in row.values)
    if period_used not in available or target_period.rsplit(" ", 1)[-1] not in period_used:
        return False
    values = [v for row in table.rows for k,v in row.values.items() if str(k).strip()==period_used and v is not None]
    mapped = [getattr(mapping,name) for name in (
        "revenue", "cost_of_revenue", "gross_profit", "rd_expense", "sga_expense",
        "operating_income", "net_income", "eps_basic", "eps_diluted", "total_assets",
        "total_liabilities", "total_equity", "cash_and_equivalents", "operating_cash_flow", "capex"
    )]
    return all(any(abs(v - float(m)) <= max(0.0001,abs(float(m))*0.000001) or
                   abs(v + float(m)) <= max(0.0001,abs(float(m))*0.000001)
                   for v in values) for m in mapped if m is not None)

# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def extract_and_store(
    verified_tables_by_page: dict[int, list[VerifiedTable]],
    *,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    fiscal_quarter: str,
    filing_type: str,
    s3_bucket: str,
    s3_key: str,
    period_end_date: date,
    filing_date: date,
) -> dict:
    """
    Take verified tables for one filing and:
    1. Map them to schema columns via LLM
    2. Insert into Postgres
    3. Compute derived fields and YoY growth
    
    Returns a dict summary of what happened.
    """
    target_period = f"{fiscal_quarter} {fiscal_year}"
    logger.info(f"\n💾 Extract & store: {ticker} {target_period}")
    logger.info(f"   Tables to process: {sum(len(ts) for ts in verified_tables_by_page.values())}")

    summary = {
        "ticker": ticker,
        "target_period": target_period,
        "income_statement_processed": False,
        "balance_sheet_processed": False,
        "cash_flow_processed": False,
        "segments_inserted": 0,
        "mappings_extracted": [],
        "errors": [],
    }

    # ── Track segments added during this run to prevent duplicate inserts ──
    # A single filing may have multiple segment tables. Without this guard,
    # we'd try to insert the same segment twice and hit a UNIQUE constraint.
    seen_segments: set[str] = set()

    with get_session() as session:
        # ── 1. Setup company / period / filing ──────────────
        company = get_or_create_company(session, ticker, company_name)
        period = get_or_create_period(
            session, fiscal_year, fiscal_quarter, period_end_date, filing_type
        )
        filing = get_or_create_filing(
            session, company.id, period.id, filing_date, s3_bucket, s3_key
        )

        # ── 2. Process each verified table ──────────────────
        for page_num, tables in verified_tables_by_page.items():
            for vtable in tables:
                logger.info(f"\n  📄 Page {page_num} | {vtable.table_type} (conf {vtable.confidence_score:.2f})")

                # Skip low-confidence
                if vtable.confidence_score < 0.7:
                    logger.warning(f"    ⏭️  Skipping low confidence")
                    continue

                if vtable.table_type in ("income_statement", "balance_sheet", "cash_flow"):
                    try:
                        normalized_table = normalize_verified_financial_table(vtable)
                    except ValueError as exc:
                        summary["errors"].append(f"Page {page_num}: {exc}")
                        continue
                    mapping = extract_financial_metrics(normalized_table, target_period)
                    if mapping is None or not validate_mapping_provenance(mapping, normalized_table, target_period):
                        summary["errors"].append(f"Page {page_num}: mapping unavailable or failed provenance validation")
                        continue

                    summary["mappings_extracted"].append({
                        "page": page_num,
                        "table_type": vtable.table_type,
                        "period_used": mapping.period_label_used,
                        "confidence": mapping.confidence,
                    })

                    if vtable.table_type == "income_statement":
                        summary["income_statement_processed"] = True
                    elif vtable.table_type == "balance_sheet":
                        summary["balance_sheet_processed"] = True
                    elif vtable.table_type == "cash_flow":
                        summary["cash_flow_processed"] = True

                    upsert_financial_metric(
                        session, company.id, period.id, filing.id,
                        mapping, source_page=page_num,
                    )

                elif vtable.table_type == "segment":
                    try:
                        normalized_table = normalize_verified_financial_table(vtable)
                    except ValueError as exc:
                        summary["errors"].append(f"Page {page_num}: {exc}")
                        continue
                    seg_list = extract_segments(normalized_table, target_period)
                    if seg_list is None or seg_list.confidence < 0.7:
                        continue

                    if seg_list.period_label_used.strip() not in {h.strip() for h in normalized_table.headers}:
                        summary["errors"].append(f"Page {page_num}: invalid segment period")
                        continue
                    for seg in seg_list.segments:
                        # Skip if already added in this transaction
                        if seg.segment_name_standardized in seen_segments:
                            logger.debug(f"    ⏭️  Skipping duplicate: {seg.segment_name_standardized}")
                            continue

                        # Skip if already in DB from a previous run
                        existing_seg = session.execute(
                            select(SegmentMetric).where(
                                SegmentMetric.company_id == company.id,
                                SegmentMetric.period_id == period.id,
                                SegmentMetric.segment_name == seg.segment_name_standardized,
                            )
                        ).scalar_one_or_none()

                        if existing_seg:
                            logger.debug(f"    ⏭️  Already in DB: {seg.segment_name_standardized}")
                            continue

                        seg_metric = SegmentMetric(
                            company_id=company.id,
                            period_id=period.id,
                            filing_id=filing.id,
                            segment_name=seg.segment_name_standardized,
                            segment_name_raw=seg.segment_name_raw,
                            segment_revenue=Decimal(str(seg.segment_revenue)) if seg.segment_revenue is not None else None,
                            segment_operating_income=Decimal(str(seg.segment_operating_income)) if seg.segment_operating_income is not None else None,
                            source_page_number=page_num,
                        )
                        session.add(seg_metric)
                        seen_segments.add(seg.segment_name_standardized)
                        session.flush()  # KEY: flush so next DB check sees it
                        summary["segments_inserted"] += 1
                        logger.info(f"    + Segment: {seg.segment_name_standardized}")
        # ── 3. Compute derived fields + YoY growth ──────────
        session.flush()
        metric = session.execute(
            select(FinancialMetric).where(
                FinancialMetric.company_id == company.id,
                FinancialMetric.period_id == period.id,
            )
        ).scalar_one_or_none()

        if metric:
            compute_derived_fields(session, metric)
            compute_yoy_growth(session, metric, company.id)

    logger.info(f"\n✅ Extract & store complete")
    return summary
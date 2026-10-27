"""
LLM-powered table extraction & verification.

Workflow:
  1. Take a raw ExtractedTable (from Docling)
  2. Send to Groq with Instructor-enforced Pydantic schema
  3. Get back a VerifiedTable with cleaned data + classification
  4. Log low-confidence results for review

Usage:
    from ingestion.table_extractor import verify_table
    verified = verify_table(raw_table, page_number=8)
"""

import logging
from typing import Optional, Any, Literal

from pydantic import BaseModel, Field
from pathlib import Path

import time

from fundlens.config import settings
from fundlens.models.parsed import ExtractedTable
from fundlens.models.verified import VerifiedTable, TableRow, CONFIDENCE_THRESHOLD
from fundlens.llm.groq_client import get_instructor_client, call_with_backoff

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

# We use the structured model for this task (best at JSON output)
# llama-3.3-70b has 30K TPM on free tier vs 8K for gpt-oss-120b
TABLE_VERIFICATION_MODEL = settings.groq_model_default # llama-3.3-70b-versatile

# Where to log low-confidence extractions
REVIEW_LOG_PATH = settings.data_processed_dir / "low_confidence_tables.jsonl"  

# ============================================================
# LOOSE LLM RESPONSE + STRICT APPLICATION VALIDATION
# ============================================================

class _GroqTableResponse(BaseModel):
    """Tool-call contract that accepts BOTH common row formats.

    Groq may emit {"Metric": "Revenue", "Q1 2025": 90234} even when the
    requested model requires {"label": "Revenue", "values": {...}}.
    Let the tool call complete, then normalize and validate locally.
    """
    table_type: Literal[
        "income_statement", "balance_sheet", "cash_flow", "segment",
        "equity", "shareholder_returns", "other",
    ]
    headers: list[str]
    rows: list[dict[str, Any]]
    units: Literal["actual", "thousands", "millions", "billions"] = "millions"
    currency: str = "USD"
    confidence_score: float = Field(ge=0.0, le=1.0)
    notes: Optional[str] = None


def _clean_numeric(value: Any) -> Optional[float]:
    """Accept JSON numbers, blanks, commas, accounting negatives and $ signs."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a financial number")
    if isinstance(value, (float, int)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"Unexpected cell value type: {type(value).__name__}")
    cell = value.strip()
    if cell.lower() in {"", "-", "—", "–", "n/a", "null", "none"}:
        return None
    negative = cell.startswith("(") and cell.endswith(")")
    if negative:
        cell = cell[1:-1].strip()
    cell = cell.replace(",", "").replace("$", "").strip()
    number = float(cell)  # invalid data must fail rather than invent a value
    return -abs(number) if negative else number


def normalize_groq_table(raw: _GroqTableResponse) -> VerifiedTable:
    """Convert Groq's flat or nested rows to the canonical VerifiedTable model."""
    normalized_rows = []
    for row in raw.rows:
        if "label" in row and isinstance(row.get("values"), dict):
            label = str(row["label"]).strip()
            values = row["values"]
        else:
            # Common Groq output: {"Metric": "Revenue", "Q1 2025": 90234.0}
            candidates = ["Metric", "metric", "Line item", "line_item", "item", "Item", "name", "Name"]
            if raw.headers:
                candidates.append(raw.headers[0])
            label_key = next((k for k in candidates if k in row and isinstance(row[k], str)), None)
            if label_key is not None:
                label = row[label_key].strip()
                values = {k: v for k, v in row.items() if k != label_key}
            elif len(row) == 1 and isinstance(next(iter(row.values())), dict):
                label, values = next(iter(row.items()))
            else:
                raise ValueError(f"Cannot identify a row label in keys: {list(row)}")
        if not label:
            raise ValueError("Empty financial row label")
        cleaned = {str(k): _clean_numeric(v) for k, v in values.items()}
        normalized_rows.append({"label": label, "values": cleaned})

    return VerifiedTable.model_validate({
        "table_type": raw.table_type,
        "headers": raw.headers,
        "rows": normalized_rows,
        "units": raw.units,
        "currency": raw.currency,
        "confidence_score": raw.confidence_score,
        "notes": raw.notes,
    })


# ============================================================
# PROMPT BUILDER
# ============================================================

SYSTEM_PROMPT = """You are a financial document specialist who cleans up messy table extractions from SEC filings (10-K, 10-Q).

Your job is to take a raw table extraction and return a clean, structured version.

CRITICAL RULES:
1. NEVER fabricate data. If a cell is empty/unclear, use null.
2. NEVER change the meaning. If row label is "Net income", keep it as "Net income".
3. Numbers must be parsed cleanly:
   - "$ 80,539"  → 80539.0
   - "(1,234)"   → -1234.0  (parentheses = negative)
   - "—" or "-"  → null
   - blank       → null
4. Classify the table type correctly. Look at the row labels:
   - "Revenue", "Costs", "Net income" → income_statement
   - "Assets", "Liabilities", "Equity" → balance_sheet
   - "Operating activities", "Investing activities" → cash_flow
   - Business unit names with revenue → segment
5. Clean up duplicate headers from merged cells. If headers are
   ["", "Three Months Ended March 31,", "Three Months Ended March 31,"]
   and the first data row is ["", "2024", "2025"], the real headers are
   ["Metric", "Q1 2024", "Q1 2025"].
6. Be honest about confidence. If the table is a TOC or junk, confidence = 0.3.
7. Identify units carefully:
   - "in millions" → "millions"
   - "in thousands" → "thousands"
   - EPS values (single-digit decimals) → "actual"

The output tool accepts row objects in either flattened or nested format. Prefer the nested TableRow format.
Every item in `rows` MUST have EXACTLY this nested structure:
    {"label": "Revenue", "values": {"Q1 2025": 90234.0}}

The outer object must have these fields:
    "table_type", "headers", "rows", "units", "currency", "confidence_score", "notes".

Preferred complete example:
{
  "table_type": "income_statement",
  "headers": ["Metric", "Q1 2025"],
  "rows": [
    {"label": "Revenue", "values": {"Q1 2025": 90234.0}},
    {"label": "Net income", "values": {"Q1 2025": 34540.0}}
  ],
  "units": "millions",
  "currency": "USD",
  "confidence_score": 0.95,
  "notes": null
}

NOT PREFERRED (but will be normalized if returned):
    {"Metric": "Revenue", "Q1 2025": 90234.0}
    {"Revenue": {"Q1 2025": 90234.0}}
These are NOT TableRow objects. Prefer `label` and `values` fields.
The `values` dictionary should contain numeric values or null, keyed by the
corresponding cleaned financial-period column heading. Do not put the row label
inside `values`. Use one tool call for the complete VerifiedTable.
"""


def build_user_prompt(raw_table: ExtractedTable, page_number: int) -> str:
    """Build the user-facing prompt with the table data."""

    headers_str = " | ".join(f'"{h}"' for h in raw_table.headers) if raw_table.headers else "(none detected)"

    MAX_ROWS_IN_PROMPT = 20
    rows_str = "\n".join(
        f"  Row {i + 1}: {row}"
        for i, row in enumerate(raw_table.rows[:MAX_ROWS_IN_PROMPT])
    )
    if len(raw_table.rows) > MAX_ROWS_IN_PROMPT:
        rows_str += f"\n  ... ({len(raw_table.rows) - MAX_ROWS_IN_PROMPT} more rows omitted)"

    return f"""Here is a raw table extracted from a SEC filing (page {page_number}).
The extraction may have issues like merged headers or empty header cells.

RAW HEADERS:
  {headers_str}

RAW ROWS:
{rows_str}

TABLE CAPTION (may contain the monetary units):
{raw_table.caption or "(none supplied)"}

Return ONE tool call. For each row, preferably produce the nested
TableRow object. For example, raw row ["Revenue", "90234"] under ["Metric", "Q1 2025"]
MUST become {{"label": "Revenue", "values": {{"Q1 2025": 90234.0}}}}.
Avoid flattened rows when possible; FundLens also normalizes them if returned.
Ensure every `rows` element has both `label` (string) and `values` (dict).

Pay special attention to:
- Real column headers (often the first data row is actually the year labels)
- Parsing numbers correctly (strip $, commas, handle parentheses as negatives)
- Classifying what kind of table this is
- Detecting actual financial units from source evidence rather than assuming
- Honest confidence score
"""

# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def verify_table(
    raw_table: ExtractedTable,
    page_number: int,
    max_retries: int = 2,
) -> Optional[VerifiedTable]:
    """
    Send a raw table to Groq for verification and cleanup.
    """
    # Filter 1: Skip empty tables
    if not raw_table.rows or len(raw_table.rows) == 0:
        logger.debug(f"  Skipping empty table on page {page_number}")
        return None

    # Filter 2: Skip tiny tables (likely artifacts)
    if len(raw_table.rows) < 2:
        logger.debug(f"  Skipping single-row table on page {page_number}")
        return None

    # Filter 3: Skip TOC (last column is small page numbers)
    if len(raw_table.rows) >= 3:
        last_col = [str(row[-1]).strip() for row in raw_table.rows[:10] if row]
        page_nums = sum(1 for v in last_col if v.isdigit() and 1 <= int(v) <= 200)
        if last_col and page_nums / len(last_col) > 0.7:
            logger.debug(f"  Skipping TOC table on page {page_number}")
            return None

    # Filter 4: Skip non-financial tables (low numeric content) [if you added this earlier]
    if len(raw_table.rows) >= 2:
        non_label_cells = []
        for row in raw_table.rows[:5]:
            for cell in row[1:]:
                cell_str = str(cell).strip()
                if cell_str:
                    non_label_cells.append(cell_str)
        
        if non_label_cells:
            numeric_cells = sum(
                1 for c in non_label_cells
                if any(ch.isdigit() for ch in c)
            )
            numeric_ratio = numeric_cells / len(non_label_cells)
            
            if numeric_ratio < 0.3:
                logger.debug(
                    f"  Skipping non-financial table on page {page_number} "
                    f"({numeric_ratio:.0%} numeric content)"
                )
                return None

    # Filter 5: Skip junk tables (cover pages, registration listings) ← NEW
    JUNK_INDICATORS = [
        'forward-looking', 'signature',
        'common stock', 'preferred stock', 'notes due',
        'incorporated by reference', 'trading symbol',
        'name of each exchange',
    ]
    FINANCIAL_KEYWORDS = [
        'revenue', 'income', 'assets', 'cash', 'liabilities',
        'expenses', 'earnings', 'sales', 'cost', 'equity',
        'profit', 'margin', 'eps', 'depreciation',
    ]
    sample_text = ' '.join(
        str(c) for row in raw_table.rows[:3] for c in row
    ).lower()
    has_junk_indicator = any(ind in sample_text for ind in JUNK_INDICATORS)
    has_financial_keyword = any(kw in sample_text for kw in FINANCIAL_KEYWORDS)
    if has_junk_indicator and not has_financial_keyword:
        logger.debug(
            f"  Skipping junk/registration table on page {page_number}"
        )
        return None

    # Now proceed to the LLM call
    user_prompt = build_user_prompt(raw_table, page_number)
    start = time.time()

    try:
        client = get_instructor_client()
        raw_response: _GroqTableResponse = call_with_backoff(
            client,
            model=TABLE_VERIFICATION_MODEL,
            response_model=_GroqTableResponse,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_retries=max_retries,
            temperature=0.0,
            max_tokens=2048,  # ← was 8192 (also reduce this for optimization)
        )

        # Accept flat Groq rows but enforce strict validated output for the app.
        verified = normalize_groq_table(raw_response)

        duration = time.time() - start
        logger.info(
            f"  ✅ Page {page_number}: {verified.table_type} "
            f"({len(verified.rows)} rows, conf={verified.confidence_score:.2f}, "
            f"{duration:.1f}s)"
        )

        # Log low-confidence for review
        if not verified.is_high_confidence:
            log_for_review(raw_table, verified, page_number)

        return verified

    except Exception as e:
        logger.error(f"  ❌ Page {page_number} verification failed: {e}")
        return None


# ============================================================
# LOW-CONFIDENCE REVIEW LOG
# ============================================================

def log_for_review(
    raw_table: ExtractedTable,
    verified: VerifiedTable,
    page_number: int,
) -> None:
    """Append a low-confidence extraction to the review log."""
    import json

    REVIEW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "page_number": page_number,
        "confidence_score": verified.confidence_score,
        "table_type": verified.table_type,
        "notes": verified.notes,
        "raw_headers": raw_table.headers,
        "raw_first_rows": raw_table.rows[:3],
        "verified_headers": verified.headers,
        "verified_row_count": len(verified.rows),
    }

    with open(REVIEW_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    logger.warning(
        f"  ⚠️  Low confidence ({verified.confidence_score:.2f}) "
        f"on page {page_number} — logged to {REVIEW_LOG_PATH.name}"
    )


# ============================================================
# BATCH PROCESSING (for a whole document)
# ============================================================

def verify_all_tables_in_document(parsed_doc) -> dict[int, list[VerifiedTable]]:
    """
    Run verification across every table in a ParsedDocument.

    Returns:
        { page_number: [VerifiedTable, ...] }
    """
    results: dict[int, list[VerifiedTable]] = {}

    total_tables = parsed_doc.total_tables
    logger.info(f"🧠 Verifying {total_tables} tables across {len(parsed_doc.pages)} pages...")

    table_idx = 0
    for page in parsed_doc.pages:
        for raw_table in page.tables:
            table_idx += 1
            logger.info(f"  [{table_idx}/{total_tables}] Page {page.page_number}...")

            verified = verify_table(raw_table, page.page_number)
            if verified is not None:
                results.setdefault(page.page_number, []).append(verified)

    high_conf = sum(
        1 for tables in results.values()
        for t in tables
        if t.is_high_confidence
    )
    total_extracted = sum(len(tables) for tables in results.values())

    logger.info(
        f"✅ Verified {total_extracted}/{total_tables} tables "
        f"({high_conf} high-confidence)"
    )

    return results
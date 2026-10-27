"""
Result Formatter: turns SQL result rows into a natural-language answer.

Uses Groq (8B model — fast/cheap, doesn't need reasoning) to synthesize
a conversational answer with proper number formatting and citations.
"""

import logging
from typing import Optional

from fundlens.config import settings
from fundlens.llm.groq_client import get_raw_client, call_with_backoff
from fundlens.models.sql import SQLResult

logger = logging.getLogger(__name__)


# Use the fast 8B model — formatting is easy, doesn't need 70B
FORMATTER_MODEL = settings.groq_model_fast  # llama-3.1-8b-instant


FORMATTER_SYSTEM_PROMPT = """You are a financial data analyst writing concise answers for hedge fund analysts.

Given a user question and SQL result rows, write a clear, conversational answer.

═══════════════════════════════════════════════════════════════
🚨 ABSOLUTE RULES (NEVER VIOLATE) 🚨
═══════════════════════════════════════════════════════════════

1. NEVER fabricate numbers. Only use values from the provided rows.
2. NEVER fabricate trends or analysis. Only state facts visible in the rows.
3. If a value is None/null, say "data not available" — DO NOT INVENT a value.
4. If most values in a column are null, say "this metric isn't consistently captured 
   in our data" rather than inventing trends.
5. If results are empty, say "No data found" — do not make up data.
6. READ THE ROWS LITERALLY. Don't interpret values to fit a narrative.

═══════════════════════════════════════════════════════════════
FORMATTING RULES
═══════════════════════════════════════════════════════════════

7. Be concise (1-3 sentences for simple lookups, 2-4 for comparisons).
8. Format money values intelligently:
   - $95,359 million → "$95.4 billion"
   - $456 million → "$456 million"
9. Format percentages with 1 decimal: 0.4705 → "47.1%"
10. For comparisons, ORDER values correctly (highest to lowest).
11. Don't include the SQL — just the answer.
"""


def format_result(
    question: str,
    sql_result: SQLResult,
    sql: str,
) -> str:
    """Generate a natural-language answer from SQL results."""
    
    if not sql_result.success:
        return "The financial query could not be completed."
    
    if sql_result.row_count == 0:
        return "No data found matching that query."
    
    # Build prompt
    rows_preview = sql_result.rows[:20]
    truncated = sql_result.row_count > len(rows_preview)
    
    # Count nulls per column — helps the LLM know which fields lack data
    null_summary = _summarize_nulls(rows_preview, sql_result.column_names)
    
    user_prompt = f"""USER QUESTION: {question}

SQL QUERY:
{sql}

SQL RESULTS ({sql_result.row_count} rows total; displaying first {len(rows_preview)} only):
{_format_rows_for_prompt(rows_preview, sql_result.column_names)}

DATA QUALITY NOTES:
{null_summary}
{ "IMPORTANT: Preview is incomplete; do not claim to summarize all rows." if truncated else "All result rows are shown." }

Write a clear, concise answer to the user's question using these results.
If a metric is mostly null, state that the data is not available — DO NOT invent values or trends."""
    
    try:
        raw_client = get_raw_client()
        response = call_with_backoff(
            raw_client,
            model=FORMATTER_MODEL,
            messages=[
                {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        # FUNDLENS_NONEMPTY_ANSWERS_V1
        # Some model responses are successful HTTP calls but have no visible text.
        content = response.choices[0].message.content
        answer = (content or "").strip()
        if not answer:
            logger.warning("SQL formatter returned an empty answer; using SQL row fallback")
            answer = _fundlens_verified_sql_fallback(sql_result)
        logger.info(f"💬 Answer: {answer[:100]}...")
        return answer
    
    except Exception as e:
        logger.error(f"Formatting failed: {e}")
        # Fallback: just return a basic summary
        return _fallback_format(sql_result)


def _format_rows_for_prompt(rows: list[dict], columns: list[str]) -> str:
    """Format rows as a markdown-style table for the LLM."""
    if not rows:
        return "(no rows)"
    
    header = " | ".join(columns)
    separator = " | ".join("---" for _ in columns)
    body = "\n".join(
        " | ".join(str(row.get(col, "")) for col in columns)
        for row in rows
    )
    return f"{header}\n{separator}\n{body}"


def _fallback_format(sql_result: SQLResult) -> str:
    """If LLM formatting fails, return basic structured info."""
    if sql_result.row_count == 1:
        row = sql_result.rows[0]
        items = ", ".join(f"{k}: {v}" for k, v in row.items())
        return f"Result: {items}"
    return f"{sql_result.row_count} rows returned. First row: {sql_result.rows[0] if sql_result.rows else 'empty'}"



def _fundlens_verified_sql_fallback(sql_result: SQLResult) -> str:
    """Render a simple, auditable answer directly from successful SQL rows.

    fundlens.financial_metrics records monetary amounts in USD millions.
    Other/nonstandard queries use the formatter's existing generic rows fallback.
    """
    if sql_result.row_count == 1 and sql_result.rows:
        row = sql_result.rows[0]
        if isinstance(row, dict):
            for metric, display_name in (("revenue", "revenue"), ("net_income", "net income")):
                value = row.get(metric)
                from decimal import Decimal
                if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
                    import math
                    if not math.isfinite(value):
                        break
                    ticker = str(row.get("ticker") or "The company").strip()
                    period = str(row.get("period_label") or "").strip()
                    # Preserve 1-million-dollar precision for original integer USD-million rows.
                    if abs(value) >= 1000:
                        amount = f"${value / 1000:,.3f} billion"
                    else:
                        amount = f"${value:,.3f}".rstrip("0").rstrip(".") + " million"
                    return f"{ticker} {period} {display_name}: {amount}.".replace("  ", " ")
    return _fallback_format(sql_result)

def _summarize_nulls(rows: list[dict], columns: list[str]) -> str:
    """Tell the LLM which columns have null values, to prevent hallucination."""
    if not rows:
        return "(no rows)"
    
    notes = []
    for col in columns:
        null_count = sum(1 for r in rows if r.get(col) is None)
        if null_count == len(rows):
            notes.append(f"- '{col}' is NULL in ALL {len(rows)} rows (data not available)")
        elif null_count > 0:
            notes.append(f"- '{col}' is NULL in {null_count}/{len(rows)} rows")
    
    return "\n".join(notes) if notes else "All values are populated."
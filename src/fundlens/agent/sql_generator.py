"""
NL→SQL Generator.

Takes a natural-language question and retrieved table schemas,
returns a structured GeneratedSQL object (Pydantic-validated).

Uses Groq Strict Structured Outputs (JSON Schema mode) for structured outputs.
The Pydantic field validators on GeneratedSQL enforce safety
(no DROP/DELETE/etc) at the model level — defense in depth
before the SQLValidator pass.
"""

import logging
import json
from typing import Optional

from fundlens.config import settings
from fundlens.llm.groq_client import get_raw_client, call_with_backoff
from fundlens.models.sql import GeneratedSQL
from fundlens.agent.schema_rag import TableSchema

logger = logging.getLogger(__name__)


# ============================================================
# MODEL CHOICE
# ============================================================

SQL_GEN_MODEL = settings.groq_model_default  # openai/gpt-oss-120b or supported strict-output model


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an expert PostgreSQL query writer for a financial database called FundLens.

You convert natural-language questions into valid PostgreSQL SELECT queries.

═══════════════════════════════════════════════════════════════
🚨 HARD RULES (NEVER VIOLATE) 🚨
═══════════════════════════════════════════════════════════════

1. ONLY generate SELECT statements (or CTEs starting with WITH).
2. NEVER use: DROP, DELETE, UPDATE, INSERT, CREATE, ALTER, TRUNCATE, GRANT, REVOKE.
3. ALWAYS include a LIMIT clause (default LIMIT 50) unless the question is a single aggregate.
4. ALWAYS qualify tables with the schema prefix: `fundlens.financial_metrics`, not just `financial_metrics`.
5. ALWAYS use table aliases: `fundlens.financial_metrics fm`, `fundlens.companies c`.

═══════════════════════════════════════════════════════════════
🚨 SCOPE — YOU ONLY HANDLE NUMBERS
═══════════════════════════════════════════════════════════════

You ONLY answer the numeric/structured part of the question.
If the user asks for narrative context ("what did they say", "why",
"how does management describe", "explain"), IGNORE that part.
A separate narrative RAG system handles those.

Example of question with mixed intent:
  Question: "What was Apple's Q1 revenue, and what did management say drove it?"
  Your job: Just return the revenue.
  Correct SQL: SELECT c.ticker, fp.period_label, fm.revenue FROM ...
  ❌ WRONG: Trying to JOIN executive_commentary to get quotes.

Never JOIN executive_commentary UNLESS the question specifically asks
for executive quotes or commentary AND provides a topic filter.

When you MUST reference executive_commentary, always use LEFT JOIN so
missing commentary doesn't zero out your entire result.

═══════════════════════════════════════════════════════════════
SCHEMA NOTES
═══════════════════════════════════════════════════════════════

- All monetary values in `financial_metrics` and `segment_metrics` are in MILLIONS USD.
- Percentages (gross_margin, *_yoy_growth) are stored as DECIMALS (0.4523 = 45.23%).
  Multiply by 100 in SELECT if user wants percentage display.
- `fiscal_quarter` values: 'Q1', 'Q2', 'Q3', 'Q4', 'FY' (NOT integers).
- `filing_type` values: '10-K', '10-Q', '8-K' (with hyphens).
- For YoY/QoQ comparisons, USE THE PRE-COMPUTED COLUMNS:
  - `revenue_yoy_growth`, `eps_yoy_growth`, `net_income_yoy_growth`, etc.
  - These already contain the growth as a decimal. NO window functions needed.
- For segment queries, `segment_name` is STANDARDIZED:
  - Apple: 'iPhone', 'Mac', 'iPad', 'Services', 'Wearables'
  - Google: 'Google Services', 'Google Cloud', 'Other Bets'
  - Microsoft: 'Intelligent Cloud' (do not relabel it as 'Azure'), 'Productivity', 'Personal Computing'

═══════════════════════════════════════════════════════════════
JOIN PATTERNS
═══════════════════════════════════════════════════════════════

To get company ticker + period + metric, JOIN like this:

  SELECT c.ticker, fp.period_label, fm.revenue
  FROM fundlens.financial_metrics fm
  JOIN fundlens.companies c       ON c.id  = fm.company_id
  JOIN fundlens.filing_periods fp ON fp.id = fm.period_id
  WHERE c.ticker = 'AAPL'
    AND fp.fiscal_year = 2025
    AND fp.fiscal_quarter = 'Q1'
  LIMIT 1;

═══════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

Return EXACTLY one JSON object with three fields:
  - sql: the SELECT query
  - reasoning: 1-2 sentences explaining your approach
  - tables_used: list of table names you used (without schema prefix)
Do not add prose or Markdown around the JSON object.
"""


# ============================================================
# FEW-SHOT EXAMPLES
# ============================================================

FEW_SHOT_EXAMPLES = """
═══════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════

Question: "What was Apple's Q1 2025 revenue?"
SQL:
  SELECT c.ticker, fp.period_label, fm.revenue
  FROM fundlens.financial_metrics fm
  JOIN fundlens.companies c       ON c.id  = fm.company_id
  JOIN fundlens.filing_periods fp ON fp.id = fm.period_id
  WHERE c.ticker = 'AAPL'
    AND fp.fiscal_year = 2025
    AND fp.fiscal_quarter = 'Q1'
  LIMIT 1;

Question: "Top 3 companies by 2024 revenue"
SQL:
  SELECT c.ticker, c.name, fm.revenue
  FROM fundlens.financial_metrics fm
  JOIN fundlens.companies c       ON c.id  = fm.company_id
  JOIN fundlens.filing_periods fp ON fp.id = fm.period_id
  WHERE fp.fiscal_year = 2024
    AND fp.fiscal_quarter = 'FY'
  ORDER BY fm.revenue DESC NULLS LAST
  LIMIT 3;

Question: "Compare iPhone and Services revenue for Apple in Q1 2025"
SQL:
  SELECT sm.segment_name, sm.segment_revenue
  FROM fundlens.segment_metrics sm
  JOIN fundlens.companies c       ON c.id  = sm.company_id
  JOIN fundlens.filing_periods fp ON fp.id = sm.period_id
  WHERE c.ticker = 'AAPL'
    AND fp.fiscal_year = 2025
    AND fp.fiscal_quarter = 'Q1'
    AND sm.segment_name IN ('iPhone', 'Services')
  ORDER BY sm.segment_revenue DESC
  LIMIT 10;
"""


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_user_prompt(question: str, schemas: list[TableSchema]) -> str:
    """Build the user-side prompt with retrieved schemas + the question."""
    from fundlens.agent.schema_rag import get_schema_retriever
    
    schema_blocks = "\n\n".join(s.to_prompt_block() for s in schemas)
    
    # Get the actual ticker mappings from the DB
    retriever = get_schema_retriever()
    company_hints = retriever.get_company_ticker_hints()
    
    return f"""RELEVANT TABLES (retrieved by semantic search):

{schema_blocks}

═══════════════════════════════════════════════════════════════
{company_hints}
═══════════════════════════════════════════════════════════════

{FEW_SHOT_EXAMPLES}

═══════════════════════════════════════════════════════════════
USER QUESTION
═══════════════════════════════════════════════════════════════

{question}

Generate a single SELECT query that answers this question.
Remember: use schema prefix `fundlens.`, table aliases, and ALWAYS include LIMIT.
"""

# ============================================================
# MAIN ENTRY POINT
# ============================================================

# Strict JSON Schema is supported by Groq GPT-OSS 120B and 20B. Unlike
# Instructor TOOLS mode, it does not depend on the model making a tool call.
# The strict schema requires every property and forbids extra fields.
GENERATED_SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "One read-only PostgreSQL SELECT query over fundlens tables",
        },
        "reasoning": {"type": "string"},
        "tables_used": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sql", "reasoning", "tables_used"],
    "additionalProperties": False,
}


def generate_sql(
    question: str,
    schemas: list[TableSchema],
    error_context: Optional[str] = None,
) -> Optional[GeneratedSQL]:
    """Generate and Pydantic-validate SQL through strict JSON-schema output."""
    client = get_raw_client()
    user_prompt = build_user_prompt(question, schemas)

    if error_context:
        user_prompt += (
            "\n\nPREVIOUS ATTEMPT FAILED:\n" + error_context +
            "\nReturn a corrected JSON object with a valid read-only SELECT query."
        )

    logger.info("Generating SQL via strict JSON schema for: %s", question[:100])

    try:
        response = call_with_backoff(
            client,
            model=SQL_GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "generated_sql",
                    "strict": True,
                    "schema": GENERATED_SQL_SCHEMA,
                },
            },
            reasoning_effort="low",
            temperature=0.0,
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty SQL generation response")
        # Keep existing safety checks in GeneratedSQL and execute_sql.
        result = GeneratedSQL.model_validate(json.loads(content))
        logger.info("SQL generated successfully using tables: %s", result.tables_used)
        return result
    except Exception as exc:
        logger.error("SQL structured generation failed: %s", exc, exc_info=True)
        return None

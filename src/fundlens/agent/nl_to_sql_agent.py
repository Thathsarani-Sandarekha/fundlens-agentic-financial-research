"""
NLToSQLAgent: end-to-end orchestrator with self-correction.

Pipeline:
  1. Retrieve relevant schemas
  2. Generate SQL (with error context if retrying)
  3. Execute SQL
  4. If execution fails: feed error back to LLM, retry (up to MAX_RETRIES)
  5. Format result as natural-language answer
"""

import logging
import time
from typing import Optional

from fundlens.models.sql import QueryResponse, SQLResult
from fundlens.agent.schema_rag import get_schema_retriever
from fundlens.agent.sql_generator import generate_sql
from fundlens.agent.sql_executor import execute_sql
from fundlens.agent.result_formatter import format_result

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG
# ============================================================

MAX_CORRECTION_ATTEMPTS = 3  # how many self-correction retries before giving up


# ============================================================
# AGENT
# ============================================================

class NLToSQLAgent:
    """
    The full NL→SQL agent with self-correction loop.
    
    Usage:
        agent = NLToSQLAgent()
        response = agent.query("What was Apple's Q1 2025 revenue?")
        print(response.answer)
    """
    
    def __init__(self):
        self.retriever = get_schema_retriever()
        self.retriever.initialize()
    
    def query(self, question: str, top_k_schemas: int = 4) -> QueryResponse:
        """Answer a natural-language question using SQL over the financial DB."""
        start = time.time()
        logger.info(f"\n{'=' * 70}")
        logger.info(f"❓ QUESTION: {question}")
        logger.info(f"{'=' * 70}")
        
        # ── 1. Retrieve relevant schemas ──────────────────
        schemas = self.retriever.retrieve(question, top_k=top_k_schemas)
        
        # ── 2 + 3 + 4. Generate + Execute + Self-correct ──
        sql_result, generated, attempts = self._generate_execute_with_correction(
            question, schemas
        )
        
        if not sql_result.success or generated is None:
            return self._failure_response(
                question, start,
                sql=generated.sql if generated else "",
                error=sql_result.error.error_message if sql_result.error else "Unknown",
                tables_used=generated.tables_used if generated else [],
                reasoning=generated.reasoning if generated else None,
                attempts=attempts,
            )
        
        # ── 5. Format answer ──────────────────────────────
        answer = format_result(question, sql_result, generated.sql)
        
        # ── Build response ────────────────────────────────
        total_ms = (time.time() - start) * 1000
        response = QueryResponse(
            question=question,
            answer=answer,
            sql=generated.sql,
            rows=sql_result.rows,
            success=True,
            attempts=attempts,
            self_corrections_made=attempts - 1,
            execution_time_ms=total_ms,
            tables_used=generated.tables_used,
            reasoning=generated.reasoning,
        )
        
        correction_note = f" ({attempts - 1} self-correction{'s' if attempts - 1 != 1 else ''})" if attempts > 1 else ""
        logger.info(f"\n✅ SUCCESS in {total_ms:.0f}ms{correction_note}")
        logger.info(f"💬 ANSWER: {answer}")
        return response
    
    def _generate_execute_with_correction(
        self,
        question: str,
        schemas: list,
    ) -> tuple[SQLResult, Optional[object], int]:
        """
        Generate SQL and execute it. If execution fails, feed error back
        to the LLM and retry up to MAX_CORRECTION_ATTEMPTS times.
        
        Returns:
            (sql_result, generated_sql_object, total_attempts)
        """
        error_context: Optional[str] = None
        last_generated = None
        last_result = None
        
        for attempt in range(1, MAX_CORRECTION_ATTEMPTS + 1):
            if attempt > 1:
                logger.info(f"\n🔄 SELF-CORRECTION ATTEMPT {attempt}/{MAX_CORRECTION_ATTEMPTS}")
            
            # Generate SQL (with error context on retries)
            generated = generate_sql(question, schemas, error_context=error_context)
            
            if generated is None:
                # Groq/Instructor can occasionally reject a tool response before
                # any SQL exists. The self-correction loop previously returned
                # immediately in this case ("after 1 attempt") rather than
                # retrying, even though a repeat may succeed.
                logger.warning(
                    "  SQL generation returned no valid structured output "
                    "(attempt %s/%s)", attempt, MAX_CORRECTION_ATTEMPTS,
                )
                if attempt < MAX_CORRECTION_ATTEMPTS:
                    error_context = (
                        "The previous attempt did not return a valid GeneratedSQL "
                        "object. Call the GeneratedSQL tool with 'sql', 'reasoning', "
                        "and 'tables_used'. The sql must be a single valid "
                        "read-only PostgreSQL SELECT with a LIMIT clause."
                    )
                    continue
                # Exhausted all retries. Preserve an earlier SQL execution
                # error if one existed; otherwise report generation failure.
                if last_result is not None:
                    return last_result, last_generated, attempt
                from fundlens.models.sql import SQLExecutionError
                return SQLResult(
                    success=False,
                    error=SQLExecutionError(
                        error_type="LLMGenerationFailed",
                        error_message=(
                            f"LLM failed to generate valid SQL after "
                            f"{attempt} attempts; inspect console for Groq errors"
                        ),
                        failed_sql="",
                    )
                ), None, attempt
            
            last_generated = generated
            
            # Execute the SQL
            sql_result = execute_sql(generated.sql)
            last_result = sql_result
            
            if sql_result.success:
                return sql_result, generated, attempt
            
            # Execution failed — build error context for retry
            error_context = self._build_error_context(generated.sql, sql_result)
            logger.warning(
                f"  ⚠️  Attempt {attempt} failed: {sql_result.error.error_type}. "
                f"{'Retrying with error context...' if attempt < MAX_CORRECTION_ATTEMPTS else 'Out of retries.'}"
            )
        
        # All retries exhausted
        return last_result, last_generated, MAX_CORRECTION_ATTEMPTS
    
    def _build_error_context(self, failed_sql: str, sql_result: SQLResult) -> str:
        """Build a structured error context to feed back to the LLM."""
        err = sql_result.error
        
        # Pattern-specific fix suggestions
        fix_hints = self._suggest_fix(failed_sql, err.error_message)
        
        return (
            f"FAILED SQL:\n```\n{failed_sql}\n```\n\n"
            f"DATABASE ERROR: {err.error_type}\n"
            f"ERROR MESSAGE: {err.error_message}\n\n"
            f"SPECIFIC FIX NEEDED:\n{fix_hints}"
        )
    
    def _suggest_fix(self, failed_sql: str, error_message: str) -> str:
        """Detect specific error patterns and suggest targeted fixes."""
        err_lower = error_message.lower()
        sql_upper = failed_sql.upper()
        
        # UNION + LIMIT issue
        if "syntax error" in err_lower and "union" in err_lower and "LIMIT" in sql_upper:
            return (
                "⚠️  PostgreSQL does not allow LIMIT inside individual SELECTs of a UNION.\n"
                "FIX OPTIONS:\n"
                "  Option A: Wrap each SELECT in parentheses:\n"
                "    (SELECT ... FROM table1 LIMIT 50)\n"
                "    UNION ALL\n"
                "    (SELECT ... FROM table2 LIMIT 50)\n"
                "  Option B: Move LIMIT to the very end:\n"
                "    SELECT ... FROM table1\n"
                "    UNION ALL\n"
                "    SELECT ... FROM table2\n"
                "    LIMIT 200\n"
            )
        
        # Double ON in JOIN
        if "syntax error" in err_lower and " ON " in failed_sql and failed_sql.count(" ON ") > failed_sql.count(" JOIN "):
            return (
                "⚠️  You have too many ON clauses. Each JOIN gets exactly ONE ON clause.\n"
                "FIX: Check that each JOIN has only one ON, and ensure the JOIN order makes sense."
            )
        
        # Column doesn't exist
        if "column" in err_lower and ("does not exist" in err_lower or "not found" in err_lower):
            return (
                "⚠️  A column name doesn't match the schema.\n"
                "FIX: Check the column names in the schema reference above. "
                "Common mistakes: using 'name' instead of column-specific names like 'segment_name', "
                "or forgetting to qualify with a table alias."
            )
        
        # Table doesn't exist (missing schema prefix)
        if "relation" in err_lower and "does not exist" in err_lower:
            return (
                "⚠️  Table name is wrong or missing the 'fundlens.' schema prefix.\n"
                "FIX: Use 'fundlens.financial_metrics' not just 'financial_metrics'."
            )
        
        # Ambiguous column
        if "ambiguous" in err_lower:
            return (
                "⚠️  A column name is ambiguous (exists in multiple tables in the JOIN).\n"
                "FIX: Qualify every column with its table alias (e.g., 'fm.revenue' not 'revenue')."
            )
        
        # Generic fallback
        return (
            "Analyze the error message carefully. Common PostgreSQL issues:\n"
            "- Each JOIN gets exactly one ON clause\n"
            "- LIMIT goes at the end of UNION queries (or wrap each SELECT in parens)\n"
            "- Table names need the 'fundlens.' schema prefix\n"
            "- Column names must match the schema exactly\n"
            "- Qualify columns with table aliases when joining multiple tables"
        )
    
    def _failure_response(
        self,
        question: str,
        start_time: float,
        sql: str = "",
        error: str = "",
        tables_used: list[str] = None,
        reasoning: Optional[str] = None,
        attempts: int = 1,
    ) -> QueryResponse:
        return QueryResponse(
            question=question,
            answer=f"Sorry, I couldn't answer that question after {attempts} attempt(s).",
            sql=sql,
            rows=[],
            success=False,
            attempts=attempts,
            self_corrections_made=max(0, attempts - 1),
            execution_time_ms=(time.time() - start_time) * 1000,
            tables_used=tables_used or [],
            reasoning=reasoning,
            error_message=error,
        )
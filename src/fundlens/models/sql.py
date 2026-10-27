"""
Pydantic models for NL→SQL agent.

These define the contracts between agent components:
  - GeneratedSQL: what the LLM returns
  - SQLResult: what the executor returns
  - QueryResponse: what the user receives
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================
# LLM OUTPUT
# ============================================================

class GeneratedSQL(BaseModel):
    """Structured output from the SQL generator LLM."""

    sql: str = Field(
        ...,
        description=(
            "A valid PostgreSQL SELECT query. Must start with SELECT. "
            "Must include LIMIT if returning multiple rows. "
            "Never use DROP, DELETE, UPDATE, INSERT, CREATE, ALTER, TRUNCATE."
        ),
    )
    reasoning: str = Field(
        ...,
        description="Brief 1-2 sentence explanation of why this SQL answers the question.",
    )
    tables_used: list[str] = Field(
        default_factory=list,
        description="List of table names used in the SQL (e.g., ['financial_metrics', 'companies']).",
    )

    @field_validator("sql")
    @classmethod
    def must_be_select(cls, value: str) -> str:
        from fundlens.utils.sql_safety import validate_read_only_sql
        return validate_read_only_sql(value)


# ============================================================
# EXECUTOR OUTPUT
# ============================================================

class SQLExecutionError(BaseModel):
    """Structured error info when SQL execution fails."""
    error_type: str = Field(..., description="e.g., 'UndefinedColumn', 'SyntaxError'")
    error_message: str
    failed_sql: str


class SQLResult(BaseModel):
    """Result of executing SQL against Postgres."""
    success: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    column_names: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    error: Optional[SQLExecutionError] = None


# ============================================================
# AGENT OUTPUT (what the user sees)
# ============================================================

class QueryResponse(BaseModel):
    """Final response from the NL→SQL agent."""

    question: str = Field(..., description="The original user question")
    answer: str = Field(..., description="Natural-language answer with key numbers")
    sql: str = Field(..., description="The SQL that was executed")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Raw result rows")
    
    # Metadata
    success: bool
    attempts: int = Field(1, description="How many LLM calls (1 = success on first try)")
    self_corrections_made: int = Field(0, description="How many self-correction retries")
    execution_time_ms: float = 0.0
    tables_used: list[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    error_message: Optional[str] = None
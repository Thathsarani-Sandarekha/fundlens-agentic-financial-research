"""
Verified table models — output of LLM table verification.

These are the cleaned, classified tables that financial_mapper consumes.
"""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


CONFIDENCE_THRESHOLD = 0.70


class TableRow(BaseModel):
    """A single row of a financial table: row label → values per column."""

    model_config = ConfigDict(allow_inf_nan=False)

    label: str = Field(
        ...,
        description="The row label, e.g., 'Revenue', 'Net income', 'Total assets'",
    )
    values: dict[str, Optional[float]] = Field(
        default_factory=dict,
        description=(
            "Column name → numeric value. Use null for blank cells. "
            "Numbers parsed (no $, no commas, no parens). "
            "Negatives as actual negatives: '(123)' → -123."
        ),
    )


class VerifiedTable(BaseModel):
    """Cleaned, verified table extracted from a SEC filing."""

    model_config = ConfigDict(allow_inf_nan=False)

    table_type: Literal[
        "income_statement",
        "balance_sheet",
        "cash_flow",
        "segment",
        "equity",
        "shareholder_returns",
        "other",
    ] = Field(
        ...,
        description="Classification of what this table represents.",
    )

    headers: list[str] = Field(
        ...,
        description="Clean column headers, usually period labels.",
    )

    rows: list[TableRow] = Field(..., description="Data rows.")

    units: Literal["actual", "thousands", "millions", "billions"] = Field(
        ...,
        description=(
            "Explicit unit shown by the source financial table. "
            "Do not silently assume millions."
        ),
    )

    currency: str = Field("USD", description="Currency code: USD, EUR, etc.")

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM's confidence (0.0-1.0) that extraction is accurate.",
    )

    notes: Optional[str] = Field(
        None,
        description="Any issues encountered or assumptions made.",
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        code = v.upper().strip()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("currency must be a three-letter code")
        return code

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence_score >= CONFIDENCE_THRESHOLD
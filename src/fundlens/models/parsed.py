"""
Parsed PDF models — output of the document parser.

These define what comes out of parser.py: pages, text blocks, raw tables.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


class TextBlock(BaseModel):
    """A single block of text on a page (paragraph, header, etc.)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Total net sales increased 2% to $383.3 billion...",
                "block_type": "narrative",
                "bbox": [72.0, 140.0, 540.0, 200.0],
            }
        }
    )

    content: str = Field(..., description="The actual text content")
    block_type: str = Field(
        ...,
        description="narrative | header | footnote | list_item | caption | other",
    )
    bbox: Optional[tuple[float, float, float, float]] = Field(
        None,
        description="Bounding box (x0, y0, x1, y1) on the page, or None",
    )


class ExtractedTable(BaseModel):
    """A raw table extracted from a page (before LLM verification)."""

    raw_html: str = Field("", description="Original HTML or markdown representation")
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: Optional[str] = Field(None)
    bbox: Optional[tuple[float, float, float, float]] = Field(None)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.headers) if self.headers else (
            len(self.rows[0]) if self.rows else 0
        )


class ParsedPage(BaseModel):
    """All content extracted from a single PDF page."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text_blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[ExtractedTable] = Field(default_factory=list)
    thumbnail_s3_path: Optional[str] = Field(None)

    @property
    def total_text_length(self) -> int:
        return sum(len(b.content) for b in self.text_blocks)


class ParsedDocument(BaseModel):
    """Complete parsed representation of a SEC filing."""

    # Identity
    filing_id: str = Field(..., description="Unique ID: ticker + filing_type + period")
    company_ticker: str
    filing_type: str

    # Source
    s3_bucket: str
    s3_key: str

    # Parsed content
    pages: list[ParsedPage] = Field(default_factory=list)

    # Metadata
    total_pages_in_pdf: int = Field(0, ge=0)
    pages_parsed_successfully: int = Field(0, ge=0)
    parser_used: str = "docling"
    parse_duration_seconds: float = Field(0.0, ge=0)
    parser_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page_counts(self):
        if self.pages_parsed_successfully > self.total_pages_in_pdf:
            raise ValueError("pages_parsed_successfully cannot exceed total_pages_in_pdf")
        return self

    @property
    def total_tables(self) -> int:
        return sum(len(p.tables) for p in self.pages)

    @property
    def total_text_blocks(self) -> int:
        return sum(len(p.text_blocks) for p in self.pages)

    @property
    def parse_success_rate(self) -> float:
        if self.total_pages_in_pdf == 0:
            return 0.0
        return self.pages_parsed_successfully / self.total_pages_in_pdf

    def summary(self) -> str:
        return (
            f"{self.company_ticker} {self.filing_type} | "
            f"{self.pages_parsed_successfully}/{self.total_pages_in_pdf} pages | "
            f"{self.total_tables} tables | "
            f"{self.total_text_blocks} text blocks | "
            f"{self.parse_duration_seconds:.1f}s"
        )
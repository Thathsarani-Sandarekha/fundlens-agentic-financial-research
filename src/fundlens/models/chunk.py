"""Text chunk model — output of the chunker, input to the embedder."""

from pydantic import BaseModel, Field


class TextChunk(BaseModel):
    """A single chunk of narrative text ready for embedding + storage."""

    chunk_id: str = Field(
        ...,
        description="Deterministic ID — same content produces same ID",
    )
    content: str = Field(
        ...,
        description="Chunk text WITH the contextual prefix",
    )
    raw_content: str = Field(
        ...,
        description="Chunk text WITHOUT the prefix (original)",
    )

    # Filing identity
    filing_id: str
    company_ticker: str
    filing_type: str
    fiscal_year: int
    fiscal_quarter: str

    # Location
    section: str
    page_number: int = Field(..., ge=1)

    # Stats
    token_count: int = Field(..., ge=0)
    char_count: int = Field(..., ge=0)
    chunk_index: int = Field(..., ge=0, description="0-based index within the filing")
    overlap_with_prev: bool = Field(False)
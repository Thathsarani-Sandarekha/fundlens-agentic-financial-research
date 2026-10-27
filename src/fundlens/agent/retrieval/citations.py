"""
Citation formatter — turns ScoredChunks into structured citation objects
and human-readable strings ready to embed in agent responses.
"""

from dataclasses import dataclass, asdict
from typing import Optional

from fundlens.agent.retrieval.bm25 import ScoredChunk


@dataclass
class Citation:
    """A structured citation pointing to a specific location in a filing."""
    
    company_ticker: str
    filing_type: str          # "10-K", "10-Q"
    fiscal_year: int
    fiscal_quarter: str       # "Q1", "FY", etc.
    section: str              # "MD&A", "Risk Factors", etc.
    page_number: int
    chunk_id: str
    relevance_score: float
    source: str               # "bm25", "vector", "both"
    
    def to_short_string(self) -> str:
        """One-line citation for inline use.
        
        Example: 'AAPL 10-Q Q1 2025, Risk Factors, page 8'
        """
        return (
            f"{self.company_ticker} {self.filing_type} "
            f"{self.fiscal_quarter} {self.fiscal_year}, "
            f"{self.section}, page {self.page_number}"
        )
    
    def to_markdown(self) -> str:
        """Markdown-formatted citation with score.
        
        Example: '**AAPL 10-Q Q1 2025** — Risk Factors, page 8 (score: 0.87)'
        """
        return (
            f"**{self.company_ticker} {self.filing_type} "
            f"{self.fiscal_quarter} {self.fiscal_year}** — "
            f"{self.section}, page {self.page_number} "
            f"(score: {self.relevance_score:.2f})"
        )
    
    def to_dict(self) -> dict:
        """Serialize for JSON responses."""
        return asdict(self)


def chunk_to_citation(chunk: ScoredChunk) -> Citation:
    """Convert a ScoredChunk into a structured Citation."""
    return Citation(
        company_ticker=chunk.company_ticker,
        filing_type=chunk.filing_type,
        fiscal_year=chunk.fiscal_year,
        fiscal_quarter=chunk.fiscal_quarter,
        section=chunk.section,
        page_number=chunk.page_number,
        chunk_id=chunk.chunk_id,
        relevance_score=chunk.score,
        source=chunk.source,
    )


def format_chunks_with_citations(chunks: list[ScoredChunk]) -> str:
    """
    Format chunks + citations as a prompt-ready block.
    
    Used to inject retrieved context into synthesis prompts:
        "Here are relevant excerpts from SEC filings:
         [1] AAPL 10-Q Q1 2025, Risk Factors, page 8 (score: 0.87)
         > Excerpt content here...
         [2] MSFT 10-Q Q1 2025, MD&A, page 23 (score: 0.82)
         > Excerpt content here...
        "
    """
    if not chunks:
        return "(no relevant excerpts found)"
    
    lines = []
    for i, chunk in enumerate(chunks, 1):
        citation = chunk_to_citation(chunk)
        lines.append(f"[{i}] {citation.to_short_string()} (score: {chunk.score:.2f})")
        # Truncate content preview for prompt efficiency
        content = chunk.content.strip().replace("\n", " ")
        if len(content) > 400:
            content = content[:400] + "..."
        lines.append(f"    {content}")
        lines.append("")
    
    return "\n".join(lines)
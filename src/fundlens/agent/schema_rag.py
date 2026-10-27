"""
Schema RAG: gives the LLM the right table descriptions for each question.

Strategy:
  1. Extract schema descriptions from Postgres COMMENTs (one chunk per table)
  2. Embed using your existing bge-base embedder
  3. Cache embeddings in-memory (move to pgvector in TODO 3.14)
  4. On each user question, retrieve top-K most relevant schemas

This prevents the LLM from hallucinating column names that don't exist.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sqlalchemy import text

from fundlens.storage.db import get_session
from fundlens.ingestion.embedder import embed_texts, embed_query

logger = logging.getLogger(__name__)

# Example questions each table answers — used to enrich embeddings for better retrieval
TABLE_EXAMPLE_QUESTIONS = {
    "companies": (
        "What companies do we track? List companies in a sector. "
        "What is Apple's ticker? Show me all tech companies."
    ),
    "filing_periods": (
        "When was the latest filing? What quarters are available? "
        "Show me Q1 2025 filings. When did fiscal year 2024 end?"
    ),
    "filings": (
        "Where is Apple's 10-K stored? What SEC filings do we have for MSFT? "
        "How many pages was the latest 10-Q?"
    ),
    "financial_metrics": (
        "What was Apple's revenue? Top 5 companies by revenue growth. "
        "Which companies had negative EPS growth? Compare net income across companies. "
        "What is Apple's gross margin? Show year-over-year revenue change. "
        "EPS growth, revenue, operating income, net income, cash flow, total assets, "
        "balance sheet, income statement metrics."
    ),
    "segment_metrics": (
        "How did AWS revenue grow? Compare iPhone vs Services revenue. "
        "What were Apple's segments last quarter? Compare Google Cloud vs Azure. "
        "Business segment performance, product line revenue, segment breakdown."
    ),
    "executive_commentary": (
        "What did Tim Cook say about AI? CEO quotes about guidance. "
        "Management commentary on competition. What did executives discuss about cloud? "
        "Quotes from earnings calls, executive statements, leadership commentary."
    ),
}


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class TableSchema:
    """A single table's schema info, ready to be injected into a prompt."""
    table_name: str
    description: str
    columns: list[dict]  # [{name, type, comment, sample_values}]
    
    def to_prompt_block(self) -> str:
        """Format this schema as a string for the LLM prompt."""
        lines = [f"TABLE: fundlens.{self.table_name}"]
        if self.description:
            lines.append(f"  Purpose: {self.description}")
        lines.append("  Columns:")
        for col in self.columns:
            line = f"    - {col['name']} ({col['type']})"
            if col.get('comment'):
                line += f" — {col['comment']}"
            if col.get('sample_values'):
                samples = ', '.join(str(v) for v in col['sample_values'][:3])
                line += f"  [examples: {samples}]"
            lines.append(line)
        return "\n".join(lines)


# ============================================================
# SCHEMA EXTRACTION (one-time setup)
# ============================================================

# Tables we want the agent to know about
RELEVANT_TABLES = [
    "companies",
    "filings",
    "filing_periods",
    "financial_metrics",
    "segment_metrics",
    "executive_commentary",
]


def extract_table_schemas() -> list[TableSchema]:
    """Pull table + column metadata from Postgres information_schema + pg_description."""
    schemas: list[TableSchema] = []
    
    with get_session() as s:
        for table_name in RELEVANT_TABLES:
            # Get column info
            col_query = text("""
                SELECT 
                    a.attname AS column_name,
                    format_type(a.atttypid, a.atttypmod) AS column_type,
                    col_description(a.attrelid, a.attnum) AS column_comment
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'fundlens'
                  AND c.relname = :table_name
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                ORDER BY a.attnum
            """)
            cols_result = s.execute(col_query, {"table_name": table_name}).fetchall()
            
            # Get table description
            desc_query = text("""
                SELECT obj_description(c.oid)
                FROM pg_class c
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'fundlens' AND c.relname = :table_name
            """)
            table_desc = s.execute(desc_query, {"table_name": table_name}).scalar() or ""
            
            # Get sample values per column (3 distinct non-null values)
            columns = []
            for col_name, col_type, col_comment in cols_result:
                samples = _get_sample_values(s, table_name, col_name)
                columns.append({
                    "name": col_name,
                    "type": col_type,
                    "comment": col_comment or "",
                    "sample_values": samples,
                })
            
            schemas.append(TableSchema(
                table_name=table_name,
                description=table_desc,
                columns=columns,
            ))
    
    logger.info(f"Extracted schemas for {len(schemas)} tables")
    return schemas


def _get_sample_values(session, table_name: str, column_name: str, n: int = 3) -> list:
    """Get up to N distinct non-null sample values for a column."""
    try:
        query = text(f"""
            SELECT DISTINCT "{column_name}"
            FROM fundlens.{table_name}
            WHERE "{column_name}" IS NOT NULL
            LIMIT :n
        """)
        results = session.execute(query, {"n": n}).fetchall()
        return [r[0] for r in results]
    except Exception:
        # Some columns can't be queried this way (e.g., vector columns)
        return []

# ============================================================
# SCHEMA RETRIEVER (called on each user question)
# ============================================================

class SchemaRetriever:
    """
    Retrieves the most relevant table schemas for a user question.
    
    Lazy-loads schemas + embeddings on first call. Reuses after that.
    """
    
    def __init__(self):
        self._schemas: list[TableSchema] = []
        self._embeddings: np.ndarray = np.array([])
        self._initialized = False
    
    def initialize(self) -> None:
        """Extract schemas from DB and embed their descriptions."""
        if self._initialized:
            return
        
        logger.info("📚 Initializing Schema RAG...")
        self._schemas = extract_table_schemas()
        
        # Embed each table's description + example questions it answers.
        # Example questions dramatically improve retrieval — the embedding model
        # learns to associate question wording with the right table.
        descriptions = []
        for s in self._schemas:
            col_summary = ", ".join(c["name"] for c in s.columns[:10])
            example_questions = TABLE_EXAMPLE_QUESTIONS.get(s.table_name, "")
            text_to_embed = (
                f"Table: {s.table_name}. "
                f"Purpose: {s.description}. "
                f"Columns: {col_summary}. "
                f"Example questions answered by this table: {example_questions}"
            )
            descriptions.append(text_to_embed)
        
        self._embeddings = embed_texts(descriptions, show_progress=False)
        self._initialized = True
        logger.info(f"✅ Schema RAG ready: {len(self._schemas)} tables embedded")
    
    def retrieve(self, question: str, top_k: int = 4) -> list[TableSchema]:
        """Return the top-K most relevant table schemas for the question."""
        self.initialize()
        
        query_emb = np.array(embed_query(question))
        
        # Cosine similarity (both already normalized by embed_texts)
        scores = self._embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        retrieved = [self._schemas[i] for i in top_indices]
        scores_log = ", ".join(
            f"{self._schemas[i].table_name}={scores[i]:.3f}" 
            for i in top_indices
        )
        logger.info(f"🔍 Schema retrieval: {scores_log}")
        return retrieved
    
    def get_company_ticker_hints(self) -> str:
        """
        Build a prompt block with all known ticker→name mappings.
        Used by the SQL generator to map user-friendly names to tickers.
        """
        with get_session() as s:
            from sqlalchemy import text
            rows = s.execute(text(
                "SELECT ticker, name FROM fundlens.companies ORDER BY ticker"
            )).fetchall()
        
        if not rows:
            return ""
        
        lines = ["KNOWN COMPANIES IN DATABASE (use the EXACT ticker in WHERE clauses):"]
        for ticker, name in rows:
            lines.append(f"  '{ticker}' = {name}")
        lines.append("")
        lines.append("CRITICAL: When users say a company name, use its TICKER, not the name.")
        lines.append("  ❌ WHERE c.ticker = 'Google'   ← will return 0 rows")
        lines.append("  ✅ WHERE c.ticker = 'GOOGL'")
        return "\n".join(lines)


# ============================================================
# SINGLETON ACCESS
# ============================================================

@lru_cache(maxsize=1)
def get_schema_retriever() -> SchemaRetriever:
    """Get the singleton SchemaRetriever instance."""
    return SchemaRetriever()


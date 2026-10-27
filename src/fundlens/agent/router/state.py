"""
QueryState — the shared state that flows through the LangGraph agent.

Every node reads what it needs and writes back updates. LangGraph merges
each node's returned dict into the state automatically.

Design principle: Optional fields default to None. That way we can tell
'this node didn't run' apart from 'this node ran but returned nothing.'
"""

from typing import TypedDict, Optional, Literal, Any


class QueryState(TypedDict, total=False):
    """
    State schema for the FundLens agent.
    
    total=False means all keys are optional — each node populates only what it owns.
    """
    
    # ─── INPUT (set once at start) ─────────────────────────
    question: str
    
    # ─── FROM INTENT CLASSIFIER ────────────────────────────
    intent: Literal["pure_sql", "pure_rag", "hybrid"]
    intent_confidence: float
    intent_reasoning: str
    
    # ─── FROM SQL PATH (populated if SQL runs) ─────────────
    sql_success: Optional[bool]
    sql_answer: Optional[str]           # natural-language SQL answer
    sql_query: Optional[str]            # the actual SQL executed
    sql_rows: Optional[list[dict]]      # raw result rows
    sql_error: Optional[str]
    
    # ─── FROM RAG PATH (populated if RAG runs) ─────────────
    rag_success: Optional[bool]
    rag_answer: Optional[str]           # natural-language RAG answer
    rag_chunks: Optional[list[dict]]    # retrieved chunks as dicts
    rag_error: Optional[str]
    
    # ─── FINAL SYNTHESIS ───────────────────────────────────
    final_answer: str
    citations: list[dict]               # unified list of citations
    
    # ─── METADATA ──────────────────────────────────────────
    route_taken: str                    # e.g., "intent→sql→synthesize"
    total_latency_ms: float
    errors: list[str]                   # accumulated non-fatal errors
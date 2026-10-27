"""
Intent classifier: routes user questions into pipeline categories.

Categories:
  - pure_sql:  Structured numeric question. Just SQL.
               "What was Apple's Q1 2025 revenue?"
               "Top 5 companies by revenue growth"
  
  - pure_rag:  Narrative/qualitative question. Just RAG.
               "What did the CEO say about AI?"
               "What are Apple's main risk factors?"
  
  - hybrid:    Needs both structured data + narrative context.
               "Companies with declining EPS and why?"
               "Apple Q4 revenue AND what management said about it"

Uses Groq llama-3.1-8b-instant (cheap, fast, high daily quota).
"""

import logging
import time
from typing import Literal

import instructor
from pydantic import BaseModel, Field

from fundlens.llm.groq_client import get_instructor_client, call_with_backoff
from fundlens.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# CONFIG — using 8B model for speed + higher daily quota
# ============================================================

CLASSIFIER_MODEL = settings.groq_model_fast   # llama-3.1-8b-instant


# ============================================================
# PYDANTIC OUTPUT
# ============================================================

class IntentClassification(BaseModel):
    """Structured output from the intent classifier."""
    
    intent: Literal["pure_sql", "pure_rag", "hybrid"] = Field(
        ...,
        description=(
            "Route this question to: "
            "'pure_sql' for numeric/aggregation questions answerable from the financial database. "
            "'pure_rag' for narrative/qualitative questions about what companies said or discussed. "
            "'hybrid' when the question needs BOTH numeric data AND narrative context."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="How confident you are in this classification (0.0-1.0).",
    )
    reasoning: str = Field(
        ...,
        description="Brief 1-sentence justification for this choice.",
    )


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a routing classifier for a financial document intelligence platform.

Given a user question, decide which pipeline(s) should handle it.

═══════════════════════════════════════════════════════════════
THE 3 CATEGORIES
═══════════════════════════════════════════════════════════════

**pure_sql** — Numeric or aggregation questions answerable from structured 
data in a financial database (revenue, EPS, margins, growth rates, segments).
No narrative context needed.

Examples:
  ✓ "What was Apple's Q1 2025 revenue?"
  ✓ "Top 5 companies by revenue growth in 2024"
  ✓ "Compare AWS, Azure, and Google Cloud revenue"
  ✓ "Which companies had negative EPS growth?"
  ✓ "What's the average gross margin across tech companies?"

**pure_rag** — Narrative or qualitative questions about what a company 
or executive SAID, DISCUSSED, or EXPLAINED. Needs actual text from filings.
No specific numbers required.

Examples:
  ✓ "What did the CEO say about AI?"
  ✓ "What are Apple's main risk factors?"
  ✓ "How does Google describe its cloud strategy?"
  ✓ "What did Microsoft's forward-looking statements cover?"
  ✓ "Explain Apple's supply chain risks"

**hybrid** — Question that needs BOTH structured numbers AND narrative context.
Usually connects a metric filter to an explanation.

Examples:
  ✓ "Which companies had declining revenue AND why?"
  ✓ "Compare Apple and Microsoft segment growth, with management commentary"
  ✓ "Show revenue trends and explain what drove them"
  ✓ "Companies with EPS growth over 15%, and what they said about it"

═══════════════════════════════════════════════════════════════
GUIDELINES
═══════════════════════════════════════════════════════════════

1. If in doubt between pure_rag and hybrid, choose pure_rag.
2. If in doubt between pure_sql and hybrid, choose pure_sql.
3. Only classify as hybrid when the question EXPLICITLY needs both types.
4. Words that suggest narrative: "say", "said", "discuss", "explain", "why", "how", "describe"
5. Words that suggest structured: "top N", "compare", "revenue", "growth", "highest", "average"
6. Confidence 0.9+ = clear-cut. 0.7-0.9 = fairly confident. Below 0.7 = ambiguous.
"""


# ============================================================
# ENTRY POINT
# ============================================================

def classify_intent(question: str) -> IntentClassification:
    """
    Classify a question into pure_sql / pure_rag / hybrid.
    
    Returns a Pydantic-validated IntentClassification.
    Raises on Groq failure (caller should handle).
    """
    logger.info(f"🧭 Classifying: '{question[:80]}...'")
    
    client = get_instructor_client()
    start = time.time()
    
    try:
        result = call_with_backoff(
            client,
            model=CLASSIFIER_MODEL,
            response_model=IntentClassification,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            max_retries=2,
            temperature=0.0,
            max_tokens=256,
        )
        
        duration = (time.time() - start) * 1000
        logger.info(
            f"  → {result.intent} (conf={result.confidence:.2f}, {duration:.0f}ms)"
        )
        logger.debug(f"  reasoning: {result.reasoning}")
        return result
    
    except Exception as e:
        logger.error(f"  ❌ Classification failed: {e}")
        # Return a safe default — hybrid path is the safest fallback
        # (runs both SQL and RAG, so no capability is lost)
        return IntentClassification(
            intent="hybrid",
            confidence=0.0,
            reasoning=f"Classification failed, defaulting to hybrid. Error: {str(e)[:100]}",
        )
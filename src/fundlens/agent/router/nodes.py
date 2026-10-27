"""
LangGraph nodes for the FundLens agent.

Each node is a function: (state: QueryState) -> dict (partial state update).
Nodes wrap existing Phase 3 (SQL) and Phase 4 (RAG) capabilities.

Design principle: each node handles its own errors internally and writes
error info to state — never raises. This lets the graph continue even when
one path fails (e.g., SQL breaks but RAG can still answer).
"""

import logging
import re
import time
from typing import Any

from fundlens.agent.router.state import QueryState
from fundlens.agent.router.intent_classifier import classify_intent
from fundlens.agent import NLToSQLAgent
from fundlens.agent.retrieval.hybrid import get_hybrid_retriever
from fundlens.agent.retrieval.citations import chunk_to_citation

logger = logging.getLogger(__name__)


# ============================================================
# SINGLETONS — build once, reuse across queries
# ============================================================

_sql_agent: NLToSQLAgent | None = None

def _get_sql_agent() -> NLToSQLAgent:
    global _sql_agent
    if _sql_agent is None:
        logger.info("🏗️  Building NLToSQLAgent (first use)...")
        _sql_agent = NLToSQLAgent()
    return _sql_agent


# ============================================================
# NODE 1: INTENT CLASSIFIER
# ============================================================

def intent_node(state: QueryState) -> dict:
    """Classify the user question into pure_sql / pure_rag / hybrid."""
    logger.info(f"\n[NODE] intent_classifier")
    try:
        result = classify_intent(state["question"])
    except Exception:
        logger.exception("Intent classification failed")
        return {"intent": "hybrid", "intent_confidence": 0.0,
                "intent_reasoning": "Classifier unavailable",
                "route_taken": "intent[hybrid]"}

    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "intent_reasoning": result.reasoning,
        "route_taken": f"intent[{result.intent}]",
    }


def _hybrid_sql_question(question: str) -> str:
    """Extract a self-contained numeric question when a hybrid query has
    a clearly separate narrative clause. Keep the original in other cases.

    Examples:
      "What was Alphabet's FY2024 revenue, and what AI risks did it discuss?"
      -> "What was Alphabet's FY2024 revenue?"

    This does not invent values or run any database query.
    """
    boundary = re.search(
        r"\s*,?\s+and\s+(?=(?:what|how|why|which|where|when)\b)",
        question, flags=re.IGNORECASE,
    )
    if not boundary:
        return question

    numeric_part = question[:boundary.start()].strip().rstrip(" ,.!?;")
    # Do not slice questions that aren't explicitly about a numeric metric.
    metric = re.search(
        r"\b(?:revenue|sales|earnings|income|profit|eps|margin|growth|"
        r"cash\s+flow|assets|liabilities|debt|capex|dividends|expense|"
        r"financial\s+performance)\b",
        numeric_part, flags=re.IGNORECASE,
    )
    return numeric_part + "?" if metric and len(numeric_part) >= 15 else question


# ============================================================
# NODE 2: SQL PATH
# ============================================================

def sql_node(state: QueryState) -> dict:
    """
    Run the Phase 3 NL→SQL agent on the question.
    
    Writes back:
      sql_success, sql_answer, sql_query, sql_rows, sql_error
    """
    logger.info(f"\n[NODE] sql")
    start = time.time()
    
    try:
        agent = _get_sql_agent()
        original_question = state["question"]
        sql_question = (
            _hybrid_sql_question(original_question)
            if state.get("intent") == "hybrid"
            else original_question
        )
        if sql_question != original_question:
            logger.info("  → Hybrid numeric subquestion: %s", sql_question)
        response = agent.query(sql_question)
        duration = (time.time() - start) * 1000
        
        logger.info(f"  → SQL {'succeeded' if response.success else 'failed'} in {duration:.0f}ms")
        
        return {
            "sql_success": response.success,
            "sql_answer": response.answer,
            "sql_query": response.sql,
            "sql_rows": response.rows,
            "sql_error": response.error_message if not response.success else None,
            "route_taken": state.get("route_taken", "") + "→sql",
        }
    
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.exception(f"  ❌ SQL node crashed in {duration:.0f}ms: {e}")
        return {
            "sql_success": False,
            "sql_answer": None,
            "sql_query": None,
            "sql_rows": [],
            "sql_error": f"Unexpected error: {e}",
            "route_taken": state.get("route_taken", "") + "→sql[crashed]",
            "errors": state.get("errors", []) + [f"sql_node: {e}"],
        }


# ============================================================
# NODE 3: RAG PATH
# ============================================================

def _infer_filing_filters(question: str) -> dict:
    """Restrict retrieval to one filing when the question unambiguously names it.

    Multiple years/companies are intentionally left unfiltered for comparisons.
    A filter improves precision; it cannot recover missing indexed passages.
    """
    q = question.lower()
    aliases = {
        "GOOGL": (r"\bgoogle\b", r"\balphabet\b", r"\bgoogl\b"),
        "AAPL": (r"\bapple\b", r"\baapl\b"),
        "MSFT": (r"\bmicrosoft\b", r"\bmsft\b"),
    }
    tickers = [ticker for ticker, patterns in aliases.items()
               if any(re.search(pattern, q) for pattern in patterns)]
    filters = {}
    if len(tickers) == 1:
        filters["company_ticker"] = tickers[0]

    years = set(re.findall(r"\b20\d{2}\b", q))
    if len(years) == 1:
        filters["fiscal_year"] = int(next(iter(years)))

        # Annual reports and FY correspond to 10-K filings in this index.
        if re.search(r"\b(?:annual\s+(?:report|filing)|10\s*-\s*k)\b", q) or re.search(r"\bfy\s*20\d{2}\b", q):
            filters["fiscal_quarter"] = "FY"
        else:
            quarters = set(re.findall(r"\bq([1-4])\b", q))
            if len(quarters) == 1:
                filters["fiscal_quarter"] = "Q" + next(iter(quarters))

    return filters


def _infer_rag_section(question: str) -> str | None:
    """Use explicit section references; never constrain arbitrary narrative questions."""
    q = question.lower()
    if re.search(r"\b(?:risk factors?|risks?|exposures?)\b", q):
        return "Risk Factors"
    return None


def _rag_retrieval_query(question: str, section: str | None) -> str:
    """Expand AI-risk vocabulary to retrieve discussions, not table-of-contents hits.

    The filing itself is still selected exclusively through metadata filters.
    This is a retrieval aid, NOT information to put in the final answer.
    """
    if section == "Risk Factors" and re.search(
        r"\b(?:ai|artificial intelligence|generative ai|machine learning)\b",
        question.lower(),
    ):
        return (
            "generative AI competition significant investment development "
            "operating costs technology risks " + question
        )
    return question


def rag_node(state: QueryState) -> dict:
    """Retrieve grounded narrative evidence from the requested filing/section.

    Never silently broaden a risk-factors query to another filing or section.
    """
    # FUNDLENS_STRICT_RAG_FILTER_V1
    import re
    logger.info("\\n[NODE] rag")
    start = time.time()
    question = state["question"]
    query_lower = question.lower()
    try:
        retriever = get_hybrid_retriever()
        # Metadata constraints apply to BOTH BM25 and vector search before reranking.
        aliases = {
            "GOOGL": (r"\balphabet\b", r"\bgoogle\b", r"\bgoogl\b"),
            "AAPL": (r"\bapple\b", r"\baapl\b"),
            "MSFT": (r"\bmicrosoft\b", r"\bmsft\b"),
        }
        companies = [ticker for ticker, patterns in aliases.items()
                     if any(re.search(p, query_lower) for p in patterns)]
        filters = {}
        if len(companies) == 1:
            filters["company_ticker"] = companies[0]

        years = set(re.findall(r"\b20\d{2}\b", query_lower))
        if len(years) == 1:
            filters["fiscal_year"] = int(next(iter(years)))
            if (re.search(r"\b(?:annual\s+(?:report|filing)|10\s*-\s*k)\b", query_lower)
                    or re.search(r"\bfy\s*20\d{2}\b", query_lower)):
                filters["fiscal_quarter"] = "FY"
            else:
                quarters = set(re.findall(r"\bq([1-4])\b", query_lower))
                if len(quarters) == 1:
                    filters["fiscal_quarter"] = "Q" + next(iter(quarters))

        risks = bool(re.search(r"\b(?:risk|risks|risk factor|risk factors|exposure|exposures)\b", query_lower))
        if risks:
            filters["section"] = "Risk Factors"

        search_query = question
        if risks and re.search(r"\b(?:ai|artificial intelligence|generative ai|machine learning)\b", query_lower):
            # Improve candidate ranking; doesn't add information to the answer.
            search_query = "artificial intelligence generative AI competition infrastructure " + question

        logger.info("RAG strict filters: %s", filters)
        chunks = retriever.search(query=search_query, top_k=5, rerank=True, **filters)
        # Defense in depth. Never show mismatched filing/year/section even if a
        # retriever implementation ignores a metadata filter.
        chunks = [c for c in chunks if all(getattr(c, key, None) == value
                                          for key, value in filters.items())]

        if not chunks:
            logger.warning("No evidence for question within requested filters: %s", filters)
            return {
                "rag_success": False,
                "rag_answer": "No relevant passages found in the requested filing and section.",
                "rag_chunks": [],
                "rag_error": "no_evidence_in_requested_filing",
                "route_taken": state.get("route_taken", "") + "→rag[empty]",
            }

        answer = _synthesize_from_chunks(question, chunks)
        duration = (time.time() - start) * 1000
        chunk_dicts = [_chunk_to_dict(c) for c in chunks]
        logger.info("RAG returned %d filtered chunks in %.0fms", len(chunks), duration)
        return {
            "rag_success": True,
            "rag_answer": answer,
            "rag_chunks": chunk_dicts,
            "rag_error": None,
            "route_taken": state.get("route_taken", "") + "→rag",
        }
    except Exception as e:
        logger.exception("RAG node crashed: %s", e)
        return {
            "rag_success": False,
            "rag_answer": None,
            "rag_chunks": [],
            "rag_error": f"Unexpected error: {e}",
            "route_taken": state.get("route_taken", "") + "→rag[crashed]",
            "errors": state.get("errors", []) + [f"rag_node: {e}"],
        }


# ============================================================
# NODE 4: FINAL SYNTHESIZER
# ============================================================

def synthesizer_node(state: QueryState) -> dict:
    """
    Combine SQL + RAG results (whichever ran) into the final answer.
    
    Three cases:
      1. Only SQL ran → use sql_answer as final
      2. Only RAG ran → use rag_answer as final
      3. Both ran → synthesize combined answer via LLM
    """
    logger.info(f"\n[NODE] synthesizer")
    
    has_sql = state.get("sql_success") is True
    has_rag = state.get("rag_success") is True
    
    citations = _build_citations(state)
    
    # Case 1: Only SQL succeeded
    if has_sql and not has_rag:
        answer = state["sql_answer"]
        return {
            "final_answer": answer,
            "citations": citations,
            "route_taken": state.get("route_taken", "") + "→synth[sql_only]",
        }
    
    # Case 2: Only RAG succeeded
    if has_rag and not has_sql:
        answer = state["rag_answer"]
        return {
            "final_answer": answer,
            "citations": citations,
            "route_taken": state.get("route_taken", "") + "→synth[rag_only]",
        }
    
    # Case 3: Both succeeded — combine them
    if has_sql and has_rag:
        answer = _synthesize_combined(state)
        return {
            "final_answer": answer,
            "citations": citations,
            "route_taken": state.get("route_taken", "") + "→synth[combined]",
        }
    
    # Case 4: Both failed; do not display internal SQL/database errors.
    return {
        "final_answer": "I couldn't answer that question from the available data.",
        "citations": [],
        "route_taken": state.get("route_taken", "") + "→synth[failed]",
    }


# ============================================================
# HELPERS
# ============================================================

def _chunk_to_dict(chunk) -> dict:
    """Serialize a ScoredChunk to a plain dict for state storage."""
    return {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "company_ticker": chunk.company_ticker,
        "filing_type": chunk.filing_type,
        "fiscal_year": chunk.fiscal_year,
        "fiscal_quarter": chunk.fiscal_quarter,
        "section": chunk.section,
        "page_number": chunk.page_number,
        "score": float(chunk.score),
        "source": chunk.source,
    }


def _build_citations(state: QueryState) -> list[dict]:
    """Build a unified citation list from whichever paths ran."""
    citations = []
    
    # From RAG chunks
    for chunk in state.get("rag_chunks") or []:
        citations.append({
            "type": "rag",
            "company_ticker": chunk["company_ticker"],
            "filing_type": chunk["filing_type"],
            "fiscal_year": chunk["fiscal_year"],
            "fiscal_quarter": chunk["fiscal_quarter"],
            "section": chunk["section"],
            "page_number": chunk["page_number"],
            "score": chunk["score"],
        })
    
    # From SQL rows (mark source rows with page_number if available)
    for row in state.get("sql_rows") or []:
        if "source_page_number" in row and row["source_page_number"]:
            citations.append({
                "type": "sql",
                "source_page_number": row["source_page_number"],
                "row_summary": {k: v for k, v in row.items() if k in ("ticker", "period_label")},
            })
    
    return citations


# FUNDLENS_CITATION_SYNTH_V2
# Inline [n] must reference the nth RAG chunk / Sources card, not invented lines.
def _fundlens_normalize_citations(answer: str, chunks) -> str:
    """Normalize only traceable citation syntax; never manufacture new evidence.

    Legacy SEC references with one/multiple page numbers are mapped to the
    corresponding returned RAG chunk indices. A page shared by multiple chunks
    maps to the first matching chunk, so the text must still be reviewed for
    claim support. Unsupported bracket/line indices are removed, not invented.
    """
    if not answer:
        return answer
    import re as _re

    page_to_index = {}
    for i, chunk in enumerate(chunks or [], 1):
        page = chunk.get("page_number") if isinstance(chunk, dict) else getattr(chunk, "page_number", None)
        if page is not None:
            page_to_index.setdefault(str(page), i)
    upper = len(chunks or [])

    # A source card has an index, not numbered text lines: 【4†L1-L9】 -> [4].
    def replace_line_reference(match):
        idx = int(match.group(1))
        if 1 <= idx <= upper:
            return f"[{idx}]"
        logger.warning("Unrecognized source index in generated line reference: %d", idx)
        return ""

    answer = _re.sub(r"【\s*(\d+)\s*†\s*L?\d+\s*[-–]\s*L?\d+\s*】", replace_line_reference, answer)

    # FUNDLENS_INDEXED_PAGE_CITATIONS_V3
    # Groq sometimes writes 【2†page 12】 rather than [2]. Verify that both the
    # index AND page exist in the currently retrieved sources before converting.
    def replace_indexed_page_reference(match):
        idx = int(match.group(1))
        page = int(match.group(2))
        if not (1 <= idx <= upper):
            logger.warning("Generated citation index %d is not in source cards", idx)
            return ""
        chunk = chunks[idx - 1]
        actual_page = chunk.get("page_number") if isinstance(chunk, dict) else getattr(chunk, "page_number", None)
        if actual_page is None or str(actual_page) != str(page):
            logger.warning("Generated citation [%d] claims page %d but source has page %s", idx, page, actual_page)
            return ""
        return f"[{idx}]"

    answer = _re.sub(
        r"【\s*(\d+)\s*†\s*(?:pages?|p\.)\s*(\d+)\s*】",
        replace_indexed_page_reference, answer, flags=_re.I,
    )

    # Map the model's legacy [GOOGL ... page 17; page 20] citations to the
    # real numbered source cards. We only touch brackets that look like filing
    # references, so ordinary bracketed content is left alone.
    def replace_filing_reference(match):
        body = match.group(1)
        if not _re.search(r"(?:10\s*[-‑–]\s*[KQ]|risk factors|cybersecurity|FY\s*20\d{2})", body, _re.I):
            return match.group(0)
        pages = _re.findall(r"\b(?:page|p\.)\s*(\d+)\b", body, _re.I)
        if not pages:
            return match.group(0)
        known = [page_to_index[p] for p in pages if p in page_to_index]
        if len(known) != len(pages):
            logger.warning("Generated citation page not present in retrieved sources: %s", pages)
        return "".join(f"[{idx}]" for idx in dict.fromkeys(known))

    answer = _re.sub(r"\[([^\[\]\n]{5,220})\]", replace_filing_reference, answer)

    def replace_source_index(match):
        idx = int(match.group(1))
        if 1 <= idx <= upper:
            return f"[{idx}]"
        logger.warning("Generated citation [%d] has no matching source card", idx)
        return ""

    answer = _re.sub(r"\[(\d{1,2})\]", replace_source_index, answer)
    # Model sometimes repeats the same source when referencing two fake lines.
    answer = _re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", answer)
    return answer


def _synthesize_from_chunks(question: str, chunks) -> str:
    """LLM synthesis for RAG-only answers. Delegates to existing narrative_rag pattern."""
    from fundlens.llm.groq_client import get_raw_client, call_with_backoff
    from fundlens.config import settings
    
    # Build context block
    context_lines = []
    for i, c in enumerate(chunks, 1):
        citation = f"{c.company_ticker} {c.filing_type} {c.fiscal_quarter} {c.fiscal_year}, {c.section}, page {c.page_number}"
        context_lines.append(f"[{i}] {citation}")
        content = c.content[:2200].strip().replace("\n", " ")
        context_lines.append(f"    {content}")
    context = "\n\n".join(context_lines)
    
    system = """You are a financial research analyst. Answer the user's question using ONLY the provided excerpts.

RULES:
1. NEVER invent facts not in the excerpts.
2. Cite claims ONLY using the actual excerpt indexes: [1], [2], [3], etc. Each number must match its provided excerpt. Do not use long-form page citations or invented line numbers.
3. Be concise: 2-4 sentences typically.
4. If narrative excerpts do not answer the narrative part, say so plainly.
5. If the question also asks for a financial figure, answer only the narrative part here. Structured SQL supplies the financial figure; do not say it is missing from these excerpts."""
    
    user = f"""USER QUESTION: {question}

RELEVANT EXCERPTS:
{context}

Write a concise, cited answer."""
    
    try:
        client = get_raw_client()
        response = call_with_backoff(
            client,
            model=settings.groq_model_default,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return _fundlens_normalize_citations(response.choices[0].message.content.strip(), chunks)
    except Exception as e:
        logger.error(f"RAG synthesis failed: {e}")
        raise RuntimeError("Narrative synthesis unavailable") from e


def _synthesize_combined(state: QueryState) -> str:
    """LLM synthesizes a unified answer combining SQL numbers + RAG narrative."""
    from fundlens.llm.groq_client import get_raw_client, call_with_backoff
    from fundlens.config import settings
    
    question = state["question"]
    sql_answer = state.get("sql_answer", "")
    rag_answer = state.get("rag_answer", "")
    evidence_sources = "\n".join(
        f"[{i}] {c.get('company_ticker', '')} {c.get('filing_type', '')} "
        f"{c.get('fiscal_quarter', '')} {c.get('fiscal_year', '')}, "
        f"{c.get('section', '')}, page {c.get('page_number', '')}"
        for i, c in enumerate(state.get("rag_chunks") or [], 1)
    )
    
    system = """You are a financial research analyst combining structured data (numbers)
with narrative context from SEC filings into a single, well-formed answer.

RULES:
1. Weave the numbers and narrative together naturally.
2. For narrative statements, cite only [1], [2], etc. from the VALID RAG SOURCE INDICES below. Do not use long-form filing/page labels or source line numbers.
3. Don't repeat information — combine, don't concatenate.
4. Be concise: 3-5 sentences typically.
5. If the structured SQL answer contains a verified figure, include it explicitly.
6. Do not invent additional facts, citations, or source line numbers.
7. The verified SQL number is authoritative. Do NOT say that the filing excerpts did not disclose the metric; those excerpts are for narrative support only.
8. Cite ONLY the supplied narrative evidence and do not imply that it supports the SQL figure."""
    
    user = f"""USER QUESTION: {question}

STRUCTURED DATA ANSWER (from database):
{sql_answer}

VALID RAG SOURCE INDICES (cite these only):
{evidence_sources}

NARRATIVE CONTEXT ANSWER (from filings):
{rag_answer}

Combine these into a single, well-formed answer."""
    
    try:
        client = get_raw_client()
        response = call_with_backoff(
            client,
            model=settings.groq_model_default,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        # FUNDLENS_NONEMPTY_ANSWERS_V1
        content = response.choices[0].message.content
        answer = (content or "").strip()
        if not answer:
            logger.warning("Hybrid synthesis returned empty text; preserving the verified SQL and RAG answers")
            answer = f"{sql_answer}\n\n{rag_answer}".strip()
        return _fundlens_normalize_citations(answer, state.get("rag_chunks") or [])
    except Exception as e:
        logger.error(f"Combined synthesis failed: {e}")
        # Fallback: just return SQL + RAG separated
        return _fundlens_normalize_citations(f"{sql_answer}\n\n{rag_answer}", state.get("rag_chunks") or [])
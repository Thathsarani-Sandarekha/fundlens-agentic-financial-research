"""FundLens — Financial Document Intelligence demo.

Run from the project root with ``streamlit run app.py``.
UI-only code: the existing LangGraph SQL/RAG/Hybrid pipeline is unchanged.
"""

import html
import logging
import re
import time

import streamlit as st

# Streamlit page settings must precede any other Streamlit UI calls.
st.set_page_config(
    page_title="FundLens | Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from fundlens.agent.router.graph import ask
from fundlens.config import settings

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("fundlens.demo")


# ─────────────────────────────────────────────────────────────
# PRESENTATION HELPERS
# ─────────────────────────────────────────────────────────────

def _safe(value: object) -> str:
    """HTML-escape source metadata before inserting into styled cards."""
    return html.escape(str(value if value is not None else ""), quote=True)


def _answer_markdown(value: object) -> str:
    """Restore common LLM-escaped Markdown delimiters for native rendering."""
    answer = str(value or "No answer was returned.").replace("\u00a0", " ")
    # A few model responses escape each Markdown star (\\*\\*bold\\*\\*).
    # Unescape only * and _; do not turn arbitrary text into executable HTML.
    return re.sub(r"\\([*_])", r"\1", answer)


def _clear_demo() -> None:
    st.session_state.question_input = ""
    st.session_state.last_demo_result = None
    st.session_state.last_demo_error = None


# ─────────────────────────────────────────────────────────────
# STYLES: scoped to the small HTML elements we own
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.route-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: .84rem;
    font-weight: 700;
    letter-spacing: .01em;
}
.route-sql    { background: #dbeafe; color: #1e40af; }
.route-rag    { background: #dcfce7; color: #166534; }
.route-hybrid { background: #fef3c7; color: #92400e; }
.source-card {
    background: #f8fafc;
    color: #0f172a !important;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    line-height: 1.55;
    font-size: .92rem;
}
.source-card strong { color: #0f172a !important; }
.source-index { color: #4338ca !important; font-weight: 700; margin-right: 8px; }
.source-detail { color: #475569 !important; }
</style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 FundLens")
    st.caption("Agentic financial research")
    st.divider()
    st.markdown("**Research modes**")
    st.markdown(
        "🔢 **SQL** — financial metrics  \n"
        "📚 **RAG** — document evidence  \n"
        "🔀 **Hybrid** — combines both"
    )
    st.divider()
    st.markdown("**Demo data**")
    st.caption(
        "Selected SEC filings and structured financial data for "
        "Alphabet (GOOGL), Apple (AAPL), and Microsoft (MSFT). "
        "Narrative coverage is limited to indexed filings."
    )
    st.divider()
    st.markdown("**Technology**")
    st.caption(
        f"Groq · {settings.groq_model_default.split('/')[-1]} / "
        f"{settings.groq_model_fast.split('/')[-1]}"
    )
    st.caption("LangGraph · PostgreSQL/pgvector · BM25 + vector · MiniLM reranker")
    st.caption("This is a research demo, not investment advice.")


# ─────────────────────────────────────────────────────────────
# SAMPLE QUESTIONS — based on confirmed structured records and indexed text.
# Structured metrics: AAPL 2025 Q1, GOOGL 2024 FY/2025 Q1,
# MSFT 2024 FY. Narrative evidence: GOOGL 2024 10-K Business,
# MD&A and Risk Factors sections.
# The first SQL, RAG and Hybrid examples were verified end-to-end.
# Remaining prompts are relevant to indexed data, but not all were
# individually tested through Groq.
# ─────────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = {
    "🔢 SQL · Financial metrics": [
        "What was Google's Q1 2025 revenue?",
        "What was Apple's Q1 2025 revenue?",
        "What was Alphabet's FY2024 net income?",
        "Compare Alphabet, Apple, and Microsoft by FY2024 revenue.",
    ],
    "📚 RAG · Annual-report evidence": [
        "What AI-related risks did Alphabet discuss in its 2024 annual report?",
        "How did Alphabet describe competitive risks in its 2024 annual report?",
        "What did Alphabet's 2024 annual report say about AI investment?",
        "How did Alphabet describe Google Cloud in its 2024 annual report?",
    ],
    "🔀 Hybrid · Metrics + evidence": [
        "What was Alphabet's FY2024 revenue, and what AI-related risks did its annual report discuss?",
        "What was Alphabet's FY2024 operating income, and what competitive risks did it describe?",
        "What was Alphabet's FY2024 net income, and what AI-related risks did it disclose?",
    ],
}


def _select_sample(question: str) -> None:
    """Fill the input without making an API request until Ask is clicked."""
    st.session_state.question_input = question
    st.session_state.last_demo_result = None  # avoid displaying an older answer
    st.session_state.last_demo_error = None


if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "last_demo_result" not in st.session_state:
    st.session_state.last_demo_result = None
if "last_demo_error" not in st.session_state:
    st.session_state.last_demo_error = None

st.title("📊 FundLens")
st.write("**Financial intelligence grounded in structured metrics and SEC filings.**")
st.caption("Ask a question about the available financial metrics or indexed SEC filings.")

with st.expander("💡 Sample questions to try", expanded=False):
    st.caption("Choose a question to fill the input below, then click Ask FundLens.")
    columns = st.columns(3, gap="medium")
    for column, (category, questions) in zip(columns, SAMPLE_QUESTIONS.items()):
        with column:
            st.markdown(f"**{category}**")
            for index, sample_question in enumerate(questions):
                st.button(
                    sample_question,
                    key=f"sample_{category.split(' ')[0]}_{index}",
                    use_container_width=True,
                    on_click=_select_sample,
                    args=(sample_question,),
                )

st.divider()
question = st.text_input(
    "Research question",
    key="question_input",
    placeholder="Ask about revenue, company risks, or both…",
)

col_ask, col_clear, col_fill = st.columns([1, 1, 5])
with col_ask:
    submit = st.button("🚀 Ask FundLens", type="primary", use_container_width=True)
with col_clear:
    st.button("Clear", on_click=_clear_demo, use_container_width=True)

# Keep the most recent result visible on harmless Streamlit reruns, but never
# rerun the expensive agent unless the user actually requests a query.
if submit:
    if not question.strip():
        st.warning("Enter a question to continue.")
    else:
        with st.spinner("Analyzing your question and retrieving evidence…"):
            start = time.perf_counter()
            try:
                agent_result = ask(question.strip())
                st.session_state.last_demo_result = {
                    "question": question.strip(),
                    "duration": time.perf_counter() - start,
                    "result": agent_result,
                }
                st.session_state.last_demo_error = None
            except Exception:
                logger.exception("FundLens request failed")
                st.session_state.last_demo_result = None
                st.session_state.last_demo_error = (
                    "The request failed. Check your Streamlit terminal for details."
                )

if st.session_state.last_demo_error:
    st.error(st.session_state.last_demo_error)

payload = st.session_state.last_demo_result
if payload is not None:
    result = payload["result"]
    st.divider()
    st.caption(f"Research question: {payload['question']}")

    intent = result.get("intent", "unknown") or "unknown"
    route = result.get("route_taken", "")
    confidence = result.get("intent_confidence")
    if not isinstance(confidence, (int, float)):
        confidence = 0.0

    col_route, col_conf, col_latency = st.columns(3)
    with col_route:
        st.caption("Research route")
        label = {"pure_sql": "🔢 SQL", "pure_rag": "📚 RAG", "hybrid": "🔀 HYBRID"}.get(
            intent, str(intent).upper()
        )
        css_class = {"pure_sql": "route-sql", "pure_rag": "route-rag", "hybrid": "route-hybrid"}.get(
            intent, "route-sql"
        )
        st.markdown(
            f'<span class="route-badge {css_class}">{_safe(label)}</span>',
            unsafe_allow_html=True,
        )
    with col_conf:
        st.metric("Routing confidence", f"{confidence:.0%}")
    with col_latency:
        st.metric("Processing time", f"{payload['duration']:.1f}s")

    if route:
        st.caption(f"Path: `{route}`")
    if result.get("intent_reasoning"):
        st.caption(f"Routing rationale: {result['intent_reasoning']}")

    # The large answer is native Markdown, NOT interpolated into HTML.
    # This renders bold text, bullets and citations properly and avoids
    # injecting untrusted model output into an HTML element.
    st.subheader("💬 Answer")
    with st.container(border=True):
        st.markdown(_answer_markdown(result.get("final_answer")))

    if intent == "hybrid" and not result.get("sql_success", False):
        st.warning("The SQL path did not complete; this response may include only filing excerpts.")

    citations = result.get("citations") or []
    if citations:
        st.subheader(f"📌 Sources ({min(len(citations), 10)})")
        for index, citation in enumerate(citations[:10], start=1):
            kind = citation.get("type")
            if kind == "rag":
                ticker = _safe(citation.get("company_ticker", ""))
                filing_type = _safe(citation.get("filing_type", ""))
                period = _safe(citation.get("fiscal_quarter", ""))
                year = _safe(citation.get("fiscal_year", ""))
                section = _safe(citation.get("section", "Unspecified section"))
                page = _safe(citation.get("page_number", "?"))
                body = (
                    f"<strong>{ticker} · {filing_type} · {period} {year}</strong>"
                    f"<span class='source-detail'> — {section}, page {page}</span>"
                )
            elif kind == "sql":
                page = citation.get("source_page_number")
                page_note = f" · source page {_safe(page)}" if page is not None else ""
                body = f"<strong>Structured financial database</strong><span class='source-detail'>{page_note}</span>"
            else:
                body = "<strong>Supporting source</strong>"

            st.markdown(
                f"<div class='source-card'><span class='source-index'>[{index}]</span>{body}</div>",
                unsafe_allow_html=True,
            )

    # Technical evidence stays available for recruiters/engineers, without
    # showing opaque reranker logits in the public source cards.
    with st.expander("🔍 Technical details · SQL, retrieval and evidence"):
        if result.get("sql_success") is not None:
            st.markdown("#### SQL pipeline")
            st.write("Status:", "✅ Success" if result["sql_success"] else "❌ Failed")
            if result.get("sql_query"):
                st.code(result["sql_query"], language="sql")
            if result.get("sql_answer"):
                st.markdown(f"**SQL answer:** {_answer_markdown(result['sql_answer'])}")
            if result.get("sql_error"):
                st.error(f"SQL error: {result['sql_error']}")
            if result.get("sql_rows"):
                st.caption(f"Returned {len(result['sql_rows'])} rows (showing up to 5)")
                st.json(result["sql_rows"][:5])

        if result.get("rag_success") is not None:
            st.markdown("#### Document retrieval")
            st.write("Status:", "✅ Success" if result["rag_success"] else "❌ Failed")
            if result.get("rag_answer"):
                st.markdown(f"**RAG answer:** {_answer_markdown(result['rag_answer'])}")
            if result.get("rag_error"):
                st.error(f"RAG error: {result['rag_error']}")
            chunks = result.get("rag_chunks") or []
            if chunks:
                st.caption(f"{len(chunks)} chunks retrieved; scores below are raw reranker values, not probabilities.")
                with st.expander("Inspect retrieved passages"):
                    for index, chunk in enumerate(chunks[:5], start=1):
                        score = chunk.get("score")
                        score_text = f" · reranker score {score:.2f}" if isinstance(score, (float, int)) else ""
                        st.markdown(
                            f"**[{index}]** {chunk.get('company_ticker', '')} · "
                            f"{chunk.get('section', '')} · p. {chunk.get('page_number', '?')}"
                            f"{score_text}"
                        )
                        st.text(str(chunk.get("content", ""))[:650])
                        st.divider()

st.divider()
st.caption(
    "Powered by Groq ("
    f"{settings.groq_model_default.split('/')[-1]} / {settings.groq_model_fast.split('/')[-1]}"
    ") · LangGraph · PostgreSQL/pgvector · hybrid retrieval · MiniLM reranking"
)

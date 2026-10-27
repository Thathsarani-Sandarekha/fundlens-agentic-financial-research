"""LangGraph router for SQL, narrative RAG, and hybrid queries."""
from functools import lru_cache
from time import perf_counter
from langgraph.graph import StateGraph, START, END
from fundlens.agent.router.state import QueryState
from fundlens.agent.router.nodes import intent_node, sql_node, rag_node, synthesizer_node


def _route_intent(state: QueryState) -> str:
    if state.get("intent_confidence", 0.0) < 0.7:
        return "hybrid"
    return {"pure_sql":"sql", "pure_rag":"rag", "hybrid":"hybrid"}.get(state.get("intent"), "hybrid")


def _after_sql(state: QueryState) -> str:
    return "rag" if not state.get("sql_success") else "synth"

@lru_cache(maxsize=1)
def get_agent():
    workflow = StateGraph(QueryState)
    workflow.add_node("intent", intent_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("synth", synthesizer_node)
    workflow.add_edge(START, "intent")
    workflow.add_conditional_edges("intent", _route_intent, {"sql":"sql", "rag":"rag", "hybrid":"sql"})
    # In hybrid mode, SQL goes to RAG even if SQL succeeds.
    def after_sql(state: QueryState):
        hybrid = state.get("intent") == "hybrid" or state.get("intent_confidence",0.0) < 0.7
        return "rag" if hybrid or not state.get("sql_success") else "synth"
    workflow.add_conditional_edges("sql", after_sql, {"rag":"rag", "synth":"synth"})
    workflow.add_edge("rag", "synth")
    workflow.add_edge("synth", END)
    return workflow.compile()


def ask(question: str) -> dict:
    if not isinstance(question,str) or not question.strip():
        raise ValueError("A nonempty question is required")
    start=perf_counter()
    output=get_agent().invoke({"question":question.strip(),"route_taken":"","errors":[]})
    output["total_latency_ms"]=(perf_counter()-start)*1000
    return output

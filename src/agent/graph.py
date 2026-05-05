"""
The pipeline is intentionally simple as requested in the exercise.
It's just two nodes: retrieve_node pulls the relevant chunks,
generate_node turns them into an answer, and that's it.

AgentState threads through both nodes so each one can read what the previous wrote.
MemorySaver handles per-session state internally, keyed by thread_id, which we
set equal to session_id on every request.
"""

from typing import Optional, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.nodes import retrieve_node, generate_node


class AgentState(TypedDict, total=False):
    # ── Required inputs ───────────────────────────────────────────────────────
    question: str           # user's input question
    session_id: str         # identifies the conversation session
    # ── Optional input (set by API caller) ───────────────────────────────────
    user_id: Optional[str]  # authenticated user id for cost attribution
    # ── Set by retrieve_node ──────────────────────────────────────────────────
    context: list[dict]     # chunks retrieved from the index
    history: list[dict]     # prior turns loaded from SessionMemory
    from_cache: bool        # True when answer came from semantic cache (skip LLM)
    # ── Set by generate_node ─────────────────────────────────────────────────
    answer: str             # generated answer text
    sources: list[str]      # source filenames cited in the answer
    confidence: Optional[float]   # composite confidence score 0–1
    confidence_level: Optional[str]  # "high" | "medium" | "low"


def build_graph():
    """Builds and compiles the LangGraph graph. Use graph.invoke() to run a question."""
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Module-level singleton — one graph instance shared across all requests
graph = build_graph()

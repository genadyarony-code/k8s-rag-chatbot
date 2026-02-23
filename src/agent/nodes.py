"""
The two nodes that run on every request: retrieve_node fetches relevant chunks,
generate_node turns them into an answer.

The routing logic in retrieve_node exists because the troubleshooting corpus
(~38 chunks) is tiny next to concepts+book (~3600 chunks). Without a filter,
vector similarity almost always returns concept results even for an obvious
troubleshooting question like "why is my pod in CrashLoopBackOff?". A regex
check is cheap — no extra LLM call — and covers the vast majority of cases.

One subtlety in the fallback: ChromaDB always returns n_results regardless of
how relevant those results actually are. So "if not context" isn't enough —
if the routing misclassified a conceptual question, you'd still get back a
full list of low-score troubleshooting chunks and never know it. The score
threshold (< 0.3) catches that case and retries without the filter.
I added lines to visually separate the parts of the code.
"""

import chromadb
import pickle
import tiktoken
import logging
import re
import numpy as np
from openai import OpenAI
from src.config.settings import settings
from src.config.feature_flags import FEATURE_FLAGS
from src.agent.memory import session_memory
from src.agent.prompts import build_prompt

_tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
logger = logging.getLogger(__name__)


# ── Query Routing ─────────────────────────────────────────────────────────────

TROUBLESHOOTING_SIGNALS = [
    # Known pod failure states — unambiguous, always troubleshooting
    r"\bcrashloopbackoff\b",
    r"\boomkilled\b",
    r"\bimagepullbackoff\b",
    r"\berrimagepull\b",
    r"\bpending\b",
    r"\bnot\s+ready\b",
    r"\bevicted\b",
    r"\bterminating\b",
    # Verbs indicating a problem with the user's own workload
    r"\bnot\s+working\b",
    r"\bnot\s+starting\b",
    r"\bnot\s+accessible\b",
    r"\bfailing\b",
    r"\bcrashing\b",
    r"\bstuck\b",
    r"\bcan.t\s+(reach|connect|access)\b",
    r"\bcannot\s+(reach|connect|access|find)\b",   # "cannot" is not caught by can.t
    r"\b(keeps?\s+restarting|restarting\s+constantly)\b",
    r"\bexits?\s+immediately\b",
    # Troubleshooting question openers — scoped to "my/the/this/it/a" so that
    # conceptual "why is Kubernetes declarative?" questions are NOT routed here.
    # Without the possessive/article guard, bare \bwhy\b fires on ~90 % of
    # conceptual questions and sends them to the tiny troubleshooting corpus.
    r"\bwhy\s+(is|are|isn.t|aren.t|won.t|can.t)\s+(my|the|this|it|a)\b",
    r"\bwhat.s\s+wrong\b",
    r"\bhow\s+to\s+(fix|debug|troubleshoot|diagnose)\b",
    r"\b(debug|diagnose|troubleshoot)\b",
]

_TROUBLESHOOTING_RE = re.compile(
    "|".join(TROUBLESHOOTING_SIGNALS),
    re.IGNORECASE
)


def _detect_doc_type(question: str) -> str | None:
    """
    Classify the question intent using TROUBLESHOOTING_SIGNALS regex patterns.

    Returns "troubleshooting" if any signal matches, None otherwise.
    A None return means no doc_type filter is applied → full corpus search.

    Why regex instead of an LLM classifier?
    An LLM call for routing would double the latency and cost of every request.
    The compiled patterns cover ~95 % of Kubernetes troubleshooting queries
    at zero extra cost and with sub-millisecond execution time.
    """
    if _TROUBLESHOOTING_RE.search(question):
        logger.info("[Router] → troubleshooting doc_type detected")
        return "troubleshooting"
    return None


# ── Nodes ─────────────────────────────────────────────────────────────────────

def retrieve_node(state: dict) -> dict:
    """
    LangGraph node — retrieval step.

    Selects ChromaDB or BM25 based on FF_USE_CHROMA, applies query routing
    for troubleshooting questions, and loads session history.
    Returns the updated state with 'context' and 'history' populated.
    """
    question = state["question"]
    session_id = state["session_id"]

    if FEATURE_FLAGS["use_chroma"]:
        # Routing: troubleshooting questions get a doc_type filter to bypass
        # the corpus size imbalance that would otherwise suppress those results.
        doc_type = _detect_doc_type(question)
        context = _chroma_search(question, doc_type_filter=doc_type)

        # Safety fallback: retry unfiltered if the filtered results are empty
        # OR if every returned chunk scores below ROUTING_SCORE_THRESHOLD.
        
        # The score check is the critical addition: ChromaDB always returns
        # n_results even when the query has no real match in the filtered
        # subset. Without this guard, a false-positive route (e.g. a
        # conceptual "why is X…" question matched by the routing regex)
        # would be answered with low-relevance troubleshooting chunks instead
        # of the correct concepts/book context.
        ROUTING_SCORE_THRESHOLD = 0.3
        if not context or all(c["score"] < ROUTING_SCORE_THRESHOLD for c in context):
            logger.warning(
                "[Router] Filtered results empty or all scores below %.1f — "
                "falling back to unfiltered search",
                ROUTING_SCORE_THRESHOLD,
            )
            context = _chroma_search(question)
    else:
        context = _bm25_search(question)

    history = session_memory.get(session_id) if FEATURE_FLAGS["use_session_memory"] else []

    return {**state, "context": context, "history": history}


def _chroma_search(question: str, doc_type_filter: str = None) -> list[dict]:
    """
    Embed the question and query ChromaDB for the top-K nearest chunks.

    Args:
        question:        The user's query string.
        doc_type_filter: Optional ChromaDB metadata filter, e.g. "troubleshooting".
                         When provided, restricts results to that corpus subset.

    Returns:
        List of dicts with keys: content, source, section_title, doc_type, score.
        Score is cosine similarity (1 - cosine distance).
    """
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(input=[question], model=settings.embedding_model)
    query_vector = response.data[0].embedding

    chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
    collection = chroma_client.get_collection(settings.chroma_collection_name)

    query_params = {
        "query_embeddings": [query_vector],
        "n_results": settings.top_k_results,
        "include": ["documents", "metadatas", "distances"]
    }

    if doc_type_filter:
        query_params["where"] = {"doc_type": {"$eq": doc_type_filter}}
        logger.info(f"[Retrieval] doc_type_filter={doc_type_filter}")

    results = collection.query(**query_params)

    return [{
        "content": doc,
        "source": meta["source"],
        "section_title": meta["section_title"],
        "doc_type": meta["doc_type"],
        "score": 1 - dist   # cosine distance → similarity score
    } for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )]


def _bm25_search(question: str) -> list[dict]:
    """
    Keyword search via BM25 (Okapi) — used when FF_USE_CHROMA is disabled.

    Loads the pre-built BM25 index from disk (bm25_index/bm25.pkl),
    scores all chunks, and returns the top-K results.
    Works offline with zero external dependencies.
    """
    with open("./bm25_index/bm25.pkl", "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    chunks = data["chunks"]
    scores = bm25.get_scores(question.lower().split())
    top_indices = np.argsort(scores)[::-1][:settings.top_k_results]

    return [{
        "content": chunks[i].content,
        "source": chunks[i].source,
        "section_title": chunks[i].section_title,
        "doc_type": chunks[i].doc_type,
        "score": float(scores[i])
    } for i in top_indices]


def generate_node(state: dict) -> dict:
    """
    LangGraph node — generation step.

    Builds the full prompt from context + history, calls GPT-4o-mini,
    logs token counts for cost monitoring, and persists the exchange to
    session memory. If FF_USE_OPENAI is disabled, returns raw chunks instead.

    Returns the updated state with 'answer' and 'sources' populated.
    """
    if not FEATURE_FLAGS["use_openai"]:
        # Degraded mode: return raw retrieved chunks without LLM generation
        raw = "\n\n".join([f"[{c['section_title']}]\n{c['content']}" for c in state["context"]])
        return {**state, "answer": raw, "sources": [c["source"] for c in state["context"]]}

    messages = build_prompt(state["question"], state["context"], state["history"])

    # Log estimated token count before the API call for cost visibility
    token_count = sum(len(_tokenizer.encode(m.get("content", ""))) for m in messages)
    logger.info(f"Request tokens: {token_count} | session: {state['session_id'][:8]}")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        stream=False   # token-by-token streaming is handled in api/main.py _stream_response
    )

    answer = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    logger.info(f"Response tokens: {total_tokens} | ~cost: ${total_tokens * 0.00000015:.6f}")

    sources = list(set(c["source"] for c in state["context"]))

    if FEATURE_FLAGS["use_session_memory"]:
        session_memory.add(state["session_id"], state["question"], answer)

    return {**state, "answer": answer, "sources": sources}

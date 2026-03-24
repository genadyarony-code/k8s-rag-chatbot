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

Client singletons: OpenAI and ChromaDB clients are created once at first use
and reused across all requests. Creating a new client per request allocates
HTTP session objects (OpenAI) or performs disk I/O (ChromaDB) on every call.
"""

import pickle
import re
import time

import chromadb
import numpy as np
import tiktoken
from openai import OpenAI

from src.agent.memory import session_memory
from src.agent.prompts import build_prompt
from src.config.settings import settings
from opentelemetry.trace import Status, StatusCode

from src.cost_control.circuit_breaker import (
    CircuitBreakerError,
    openai_breaker,
    with_circuit_breaker,
)
from src.cost_control.cost_tracker import get_cost_tracker
from src.cost_control.token_budget import get_token_budget
from src.observability.logging_config import get_logger
from src.observability.metrics import (
    bm25_query_total,
    chat_tokens_total,
    chroma_query_total,
    generation_latency_seconds,
    retrieval_latency_seconds,
)
from src.advanced.model_fallback import get_fallback_handler
from src.hitl.confidence import calculate_confidence, get_confidence_level
from src.observability.tracing import get_tracer
from src.rag.hybrid_search import hybrid_search
from src.rag.query_decomposition import decompose_query
from src.rag.reranker import get_reranker

_tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
log = get_logger(__name__)
tracer = get_tracer(__name__)


# ── Lazy singletons ────────────────────────────────────────────────────────────
# Initialized on first use so that importing this module in tests doesn't
# trigger network calls or require a live ChromaDB index.

_openai_client: OpenAI | None = None
_chroma_collection = None
_bm25_data: dict | None = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        _chroma_collection = client.get_collection(settings.chroma_collection_name)
    return _chroma_collection


def _get_bm25_data() -> dict:
    """Load BM25 index from disk once; return cached data on subsequent calls."""
    global _bm25_data
    if _bm25_data is None:
        with open("./bm25_index/bm25.pkl", "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


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
    r"\bcannot\s+(reach|connect|access|find)\b",
    r"\b(keeps?\s+restarting|restarting\s+constantly)\b",
    r"\bexits?\s+immediately\b",
    # Troubleshooting question openers — scoped to "my/the/this/it/a" so that
    # conceptual "why is Kubernetes declarative?" questions are NOT routed here.
    r"\bwhy\s+(is|are|isn.t|aren.t|won.t|can.t)\s+(my|the|this|it|a)\b",
    r"\bwhat.s\s+wrong\b",
    r"\bhow\s+to\s+(fix|debug|troubleshoot|diagnose)\b",
    r"\b(debug|diagnose|troubleshoot)\b",
]

_TROUBLESHOOTING_RE = re.compile(
    "|".join(TROUBLESHOOTING_SIGNALS),
    re.IGNORECASE,
)


def _detect_doc_type(question: str) -> str | None:
    """
    Classify the question intent using TROUBLESHOOTING_SIGNALS regex patterns.

    Returns "troubleshooting" if any signal matches, None otherwise.
    A None return means no doc_type filter is applied → full corpus search.

    Why regex instead of an LLM classifier?
    An LLM call for routing would double the latency and cost of every request.
    The compiled patterns cover ~95% of Kubernetes troubleshooting queries
    at zero extra cost and with sub-millisecond execution time.
    """
    if _TROUBLESHOOTING_RE.search(question):
        log.info("query_routed", doc_type="troubleshooting")
        return "troubleshooting"
    return None


# ── Nodes ─────────────────────────────────────────────────────────────────────

def retrieve_node(state: dict) -> dict:
    """
    LangGraph node — retrieval step.

    Pipeline (each stage is individually gated by a feature flag):
    1. Query decomposition — splits complex / comparison questions into
       independent sub-queries (ff_use_query_decomposition).
    2. Hybrid search — RRF fusion of ChromaDB dense + BM25 sparse results
       with doc-type routing preserved (ff_use_hybrid_search).
       Falls back to single-source search when flag is off.
    3. Reranking — FlashRank cross-encoder reorders candidates by true
       relevance (ff_use_reranking).
    4. Deduplication + limiting — merge results from all sub-queries and
       keep at most 10 unique chunks.

    Returns the updated state with 'context' and 'history' populated.
    """
    # Retrieval candidates per sub-query before reranking
    CANDIDATE_COUNT = 20

    with tracer.start_as_current_span("retrieve_node") as span:
        start_time = time.time()
        question = state["question"]
        session_id = state["session_id"]

        span.set_attribute("session_id", session_id)
        span.set_attribute("question_length", len(question))
        span.set_attribute("use_chroma", settings.ff_use_chroma)

        # ── 0. Semantic cache ─────────────────────────────────────────────────
        if settings.ff_use_semantic_cache:
            from src.advanced.semantic_cache import get_semantic_cache
            cached = get_semantic_cache().get(question)
            if cached:
                log.info(
                    "semantic_cache_hit",
                    session_id=session_id,
                    question_preview=question[:60],
                )
                span.set_attribute("from_cache", True)
                return {
                    **state,
                    "context": [],
                    "history": session_memory.get(session_id) if settings.ff_use_session_memory else [],
                    "answer": cached["answer"],
                    "sources": cached["sources"],
                    "confidence": cached.get("metadata", {}).get("confidence"),
                    "confidence_level": cached.get("metadata", {}).get("confidence_level", "medium"),
                    "from_cache": True,
                }

        # ── 1. Query decomposition ────────────────────────────────────────────
        if settings.ff_use_query_decomposition:
            with tracer.start_as_current_span("query_decomposition"):
                sub_queries = decompose_query(question)
        else:
            sub_queries = [question]

        span.set_attribute("sub_queries_count", len(sub_queries))

        all_chunks: list[dict] = []

        for sub_query in sub_queries:
            # ── 2. Retrieval ──────────────────────────────────────────────────
            if settings.ff_use_chroma:
                with tracer.start_as_current_span("detect_doc_type"):
                    doc_type = _detect_doc_type(sub_query)
                span.set_attribute("doc_type_filter", str(doc_type))

                if settings.ff_use_hybrid_search:
                    # Hybrid: ChromaDB (with routing) + BM25 via RRF
                    _dt = doc_type  # capture for closure
                    with tracer.start_as_current_span("hybrid_search"):
                        chunks = hybrid_search(
                            query=sub_query,
                            chroma_search_fn=lambda q, n, dt=_dt: _chroma_search(
                                q, doc_type_filter=dt, top_k=n
                            ),
                            bm25_search_fn=lambda q, n: _bm25_search(q, top_k=n),
                            top_k=CANDIDATE_COUNT,
                        )
                    chroma_query_total.labels(doc_type_filter=str(doc_type)).inc()
                    bm25_query_total.inc()
                else:
                    # Classic single-source search with score-threshold fallback
                    with tracer.start_as_current_span("chroma_search"):
                        chunks = _chroma_search(
                            sub_query,
                            doc_type_filter=doc_type,
                            top_k=CANDIDATE_COUNT,
                        )
                    chroma_query_total.labels(doc_type_filter=str(doc_type)).inc()

                    ROUTING_SCORE_THRESHOLD = 0.3
                    if not chunks or all(c["score"] < ROUTING_SCORE_THRESHOLD for c in chunks):
                        log.warning(
                            "routing_fallback_triggered",
                            session_id=session_id,
                            doc_type_filter=doc_type,
                            threshold=ROUTING_SCORE_THRESHOLD,
                        )
                        with tracer.start_as_current_span("chroma_search_fallback"):
                            chunks = _chroma_search(sub_query, top_k=CANDIDATE_COUNT)
                        chroma_query_total.labels(doc_type_filter="fallback").inc()
            else:
                with tracer.start_as_current_span("bm25_search"):
                    chunks = _bm25_search(sub_query, top_k=CANDIDATE_COUNT)
                bm25_query_total.inc()

            # ── 3. Reranking ──────────────────────────────────────────────────
            if settings.ff_use_reranking and chunks:
                with tracer.start_as_current_span("reranking"):
                    chunks = get_reranker().rerank(
                        sub_query, chunks, top_k=settings.top_k_results
                    )
                span.set_attribute("chunks_after_rerank", len(chunks))

            all_chunks.extend(chunks)

        # ── 4. Deduplicate and limit ──────────────────────────────────────────
        seen: set[int] = set()
        context: list[dict] = []
        for chunk in all_chunks:
            cid = hash(chunk["content"])
            if cid not in seen:
                seen.add(cid)
                context.append(chunk)
            if len(context) == 10:
                break

        history = session_memory.get(session_id) if settings.ff_use_session_memory else []

        elapsed = time.time() - start_time
        retrieval_latency_seconds.observe(elapsed)
        span.set_attribute("chunks_retrieved", len(context))
        span.set_attribute("latency_seconds", round(elapsed, 3))

        log.info(
            "retrieval_completed",
            session_id=session_id,
            sub_queries=len(sub_queries),
            total_candidates=len(all_chunks),
            unique_chunks=len(context),
            latency_seconds=round(elapsed, 3),
            used_chroma=settings.ff_use_chroma,
            used_hybrid=settings.ff_use_hybrid_search,
            used_reranking=settings.ff_use_reranking,
        )

        return {**state, "context": context, "history": history}


def _chroma_search(
    question: str,
    doc_type_filter: str = None,
    top_k: int = None,
) -> list[dict]:
    """
    Embed the question and query ChromaDB for the nearest chunks.

    Args:
        question:        The user's query string.
        doc_type_filter: Optional ChromaDB metadata filter ("troubleshooting").
        top_k:           Number of results; defaults to settings.top_k_results.

    Returns:
        List of dicts with keys: content, source, section_title, doc_type, score.
        Score is cosine similarity (1 - cosine distance).
    """
    n = top_k if top_k is not None else settings.top_k_results

    client = _get_openai_client()
    response = client.embeddings.create(input=[question], model=settings.embedding_model)
    query_vector = response.data[0].embedding

    collection = _get_chroma_collection()

    query_params = {
        "query_embeddings": [query_vector],
        "n_results": n,
        "include": ["documents", "metadatas", "distances"],
    }

    if doc_type_filter:
        query_params["where"] = {"doc_type": {"$eq": doc_type_filter}}
        log.info("chroma_search", doc_type_filter=doc_type_filter)

    results = collection.query(**query_params)

    return [{
        "content": doc,
        "source": meta["source"],
        "section_title": meta["section_title"],
        "doc_type": meta["doc_type"],
        "score": 1 - dist,
    } for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )]


def _bm25_search(question: str, top_k: int = None) -> list[dict]:
    """
    Keyword search via BM25 (Okapi).

    Args:
        question: The user's query string.
        top_k:    Number of results; defaults to settings.top_k_results.
    """
    n = top_k if top_k is not None else settings.top_k_results

    data = _get_bm25_data()
    bm25 = data["bm25"]
    chunks = data["chunks"]

    scores = bm25.get_scores(question.lower().split())
    top_indices = np.argsort(scores)[::-1][:n]

    return [{
        "content": chunks[i].content,
        "source": chunks[i].source,
        "section_title": chunks[i].section_title,
        "doc_type": chunks[i].doc_type,
        "score": float(scores[i]),
    } for i in top_indices]


@with_circuit_breaker
def _call_openai_completion(client: OpenAI, **kwargs):
    """OpenAI completion call wrapped with the circuit breaker."""
    return client.chat.completions.create(**kwargs)


def generate_node(state: dict) -> dict:
    """
    LangGraph node — generation step.

    Builds the full prompt from context + history, enforces token budgets and
    daily cost limits, calls GPT-4o-mini through a circuit breaker, and
    persists the exchange to session memory.
    If ff_use_openai is disabled, returns raw chunks instead.

    Returns the updated state with 'answer' and 'sources' populated.
    """
    with tracer.start_as_current_span("generate_node") as span:
        span.set_attribute("session_id", state["session_id"])
        span.set_attribute("use_openai", settings.ff_use_openai)

        # Skip generation entirely on cache hits — answer already populated
        if state.get("from_cache"):
            log.info("generation_skipped_cache_hit", session_id=state["session_id"])
            return state

        if not settings.ff_use_openai:
            raw = "\n\n".join([f"[{c['section_title']}]\n{c['content']}" for c in state["context"]])
            return {
                **state,
                "answer": raw,
                "sources": [c["source"] for c in state["context"]],
                "confidence": 0.5,
                "confidence_level": "medium",
            }

        messages = build_prompt(
            state["question"],
            state["context"],
            state["history"],
            session_id=state.get("session_id", ""),
        )
        estimated_tokens = sum(len(_tokenizer.encode(m.get("content", ""))) for m in messages)
        span.set_attribute("estimated_input_tokens", estimated_tokens)

        # ── Token budget check ────────────────────────────────────────────────
        budget = get_token_budget()
        allowed, reason = budget.check_and_reserve(state["session_id"], estimated_tokens)
        if not allowed:
            log.error(
                "request_blocked_by_budget",
                session_id=state["session_id"],
                estimated_tokens=estimated_tokens,
                reason=reason,
            )
            span.set_attribute("blocked_by", "token_budget")
            return {
                **state,
                "answer": f"Daily token budget exceeded. {reason}",
                "sources": [],
                "confidence": 0.0,
                "confidence_level": "low",
            }

        # ── Daily cost limit check ────────────────────────────────────────────
        tracker = get_cost_tracker()
        if not tracker.is_budget_available():
            log.error(
                "request_blocked_by_cost_limit",
                session_id=state["session_id"],
                daily_cost=tracker.get_stats()["daily_cost_usd"],
            )
            span.set_attribute("blocked_by", "cost_limit")
            return {
                **state,
                "answer": "Daily cost limit exceeded. System is in degraded mode.",
                "sources": [],
                "confidence": 0.0,
                "confidence_level": "low",
            }

        log.info(
            "generation_started",
            session_id=state["session_id"],
            input_tokens=estimated_tokens,
        )

        start_time = time.time()
        client = _get_openai_client()

        with tracer.start_as_current_span("openai_completion") as llm_span:
            llm_span.set_attribute("model", settings.llm_model)
            llm_span.set_attribute("input_tokens", estimated_tokens)

            if settings.ff_use_model_fallback:
                # Model fallback cascade — tries gpt-4o-mini → gpt-3.5-turbo
                # (→ claude-3-haiku if ANTHROPIC_API_KEY is set)
                try:
                    fb_result = get_fallback_handler().call_with_fallback(
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1000,
                        session_id=state["session_id"],
                    )
                    answer = fb_result["content"]
                    total_tokens = fb_result["total_tokens"]
                    input_tokens = fb_result["input_tokens"]
                    output_tokens = fb_result["output_tokens"]
                    llm_span.set_attribute("model_used", fb_result["model"])
                    llm_span.set_attribute("provider", fb_result["provider"])
                    llm_span.set_attribute("total_tokens", total_tokens)
                except RuntimeError:
                    llm_span.set_status(Status(StatusCode.ERROR, "All models failed"))
                    log.error(
                        "all_models_failed",
                        session_id=state["session_id"],
                    )
                    return {
                        **state,
                        "answer": "All AI models are currently unavailable. Please try again shortly.",
                        "sources": [],
                        "confidence": 0.0,
                        "confidence_level": "low",
                    }
            else:
                # Original single-model path with circuit breaker
                try:
                    response = _call_openai_completion(
                        client,
                        model=settings.llm_model,
                        messages=messages,
                        temperature=0.1,
                        stream=False,
                    )
                    llm_span.set_attribute("total_tokens", response.usage.total_tokens)
                    llm_span.set_attribute("output_tokens", response.usage.completion_tokens)
                except CircuitBreakerError:
                    llm_span.set_status(Status(StatusCode.ERROR, "Circuit breaker open"))
                    log.error(
                        "openai_circuit_breaker_open",
                        session_id=state["session_id"],
                        breaker_state=str(openai_breaker.current_state),
                    )
                    return {
                        **state,
                        "answer": "OpenAI service temporarily unavailable. Please try again shortly.",
                        "sources": [],
                        "confidence": 0.0,
                        "confidence_level": "low",
                    }
                answer = response.choices[0].message.content
                total_tokens = response.usage.total_tokens
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

        # ── Citation validation ───────────────────────────────────────────────
        # Default to True (optimistic) when validation is disabled
        citation_is_valid = True
        if settings.ff_use_citation_validation:
            with tracer.start_as_current_span("citation_validation") as cv_span:
                from src.rag.citation_validator import get_validator
                citation_is_valid, unsupported = get_validator().validate(
                    answer, state["context"], threshold=0.5
                )
                cv_span.set_attribute("citations_valid", citation_is_valid)
                cv_span.set_attribute("unsupported_claims", len(unsupported))
                if not citation_is_valid:
                    log.warning(
                        "answer_contains_unsupported_claims",
                        session_id=state["session_id"],
                        unsupported_count=len(unsupported),
                        unsupported_preview=[s[:60] for s in unsupported[:3]],
                    )

        # ── Confidence scoring ────────────────────────────────────────────────
        sources = list(set(c["source"] for c in state["context"]))
        top_retrieval_score = state["context"][0]["score"] if state["context"] else 0.0

        confidence = calculate_confidence(
            retrieval_score=top_retrieval_score,
            citation_valid=citation_is_valid,
            source_count=len(sources),
        )
        confidence_level = get_confidence_level(confidence)
        span.set_attribute("confidence_score", round(confidence, 3))
        span.set_attribute("confidence_level", confidence_level)

        log.info(
            "response_confidence_calculated",
            session_id=state["session_id"],
            confidence=round(confidence, 3),
            level=confidence_level,
            top_retrieval_score=round(top_retrieval_score, 3),
        )

        # ── Cost tracking ─────────────────────────────────────────────────────
        cost_usd, _ = tracker.track_request(
            session_id=state["session_id"],
            model=settings.llm_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            user_id=state.get("user_id"),
        )

        chat_tokens_total.labels(model=settings.llm_model).inc(total_tokens)

        elapsed = time.time() - start_time
        generation_latency_seconds.observe(elapsed)
        span.set_attribute("cost_usd", round(cost_usd, 8))
        span.set_attribute("latency_seconds", round(elapsed, 3))

        log.info(
            "generation_completed",
            session_id=state["session_id"],
            total_tokens=total_tokens,
            cost_usd=round(cost_usd, 8),
            latency_seconds=round(elapsed, 3),
        )

        if settings.ff_use_session_memory:
            session_memory.add(state["session_id"], state["question"], answer)

        # Cache the result for future identical/similar questions
        if settings.ff_use_semantic_cache:
            from src.advanced.semantic_cache import get_semantic_cache
            get_semantic_cache().set(
                question=state["question"],
                answer=answer,
                sources=sources,
                metadata={"confidence": confidence, "confidence_level": confidence_level},
            )

        return {
            **state,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "confidence_level": confidence_level,
        }

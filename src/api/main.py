"""
Three endpoints: POST /chat for questions, GET /health for index and flag status,
and POST /reset/{session_id} to clear conversation history.

On startup the server checks that the index is ready before accepting any traffic.
A server that starts but serves garbage because the index is missing is harder to
debug than one that simply refuses to start, so check_index_health() crashes loudly.

Streaming mode bypasses LangGraph.invoke() and calls retrieve_node() directly.
This gives tighter control over the token loop; routing it through the graph
wrapper would complicate the SSE generator for no real benefit.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import chromadb
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from openai import AsyncOpenAI
from prometheus_client import CONTENT_TYPE_LATEST
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.agent.graph import graph
from src.agent.memory import session_memory
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse
from src.config.settings import settings
from src.observability.logging_config import configure_logging, get_logger
from src.observability.tracing import configure_tracing
from src.api.auth_routes import router as auth_router
from src.api.approval_routes import router as approval_router
from src.auth.dependencies import optional_auth
from src.cost_control.circuit_breaker import openai_breaker
from src.observability.metrics import (
    chat_latency_seconds,
    chat_requests_total,
    chat_tokens_total,
    feature_flag_status,
    get_metrics,
    index_health,
)
from src.evaluation.live_sampler import get_sampler
from src.security.rate_limiter import limiter
from src.security.validator import validate_chat_input

log = get_logger(__name__)


def check_index_health():
    """
    Checks that the index is built and healthy before the server takes any traffic.
    Raises RuntimeError (which crashes startup) if the manifest is missing or if
    either index reports a failure. Loud crash beats silent wrong answers.
    """
    manifest = "data/processed/index_meta.json"

    if not os.path.exists(manifest):
        raise RuntimeError(
            "\n❌ Index manifest not found!"
            "\n Run: python scripts/ingest.py"
        )

    with open(manifest) as f:
        meta = json.load(f)

    if not meta.get("chroma_ok") or not meta.get("bm25_ok"):
        raise RuntimeError(
            f"\n❌ Index is in an inconsistent state!"
            f"\n chroma_ok={meta.get('chroma_ok')} | bm25_ok={meta.get('bm25_ok')}"
            f"\n Run: python scripts/ingest.py --force"
        )

    log.info(
        "index_health_check_passed",
        chunk_count=meta.get("chunk_count"),
        version=meta.get("chunk_schema_version"),
        built_at=meta.get("built_at", "unknown"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(log_level="INFO")
    log.info("application_startup", version="1.0.0")

    if settings.tracing_enabled:
        configure_tracing(app)
        log.info(
            "tracing_enabled",
            otlp_endpoint=settings.otlp_endpoint or "console",
        )

    check_index_health()
    yield
    log.info("application_shutdown")


app = FastAPI(title="K8s RAG Chatbot", version="1.0.0", lifespan=lifespan)

# Auth routes (/auth/users, /auth/keys, /auth/me, etc.)
app.include_router(auth_router)
# Approval workflow (/approvals, /approvals/{id}/approve, etc.)
app.include_router(approval_router)

# Rate limiter — must be registered before any route decorators
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/chat")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat(
    request: ChatRequest,
    req: Request,
    caller_id: Optional[str] = Depends(optional_auth),
):
    """
    Main chat endpoint with security validation and metrics instrumentation.

    Authentication is opt-in via the ENABLE_AUTH setting:
    - ENABLE_AUTH=false (default/dev): no API key required
    - ENABLE_AUTH=true  (production):  valid API key required

    Returns either a streaming SSE response or a full JSON response depending
    on the ff_use_streaming feature flag.

    Streaming (SSE) response format:
        data: {"token": "...", "done": false}  ← one per token
        data: {"sources": [...], "done": true}  ← final signal

    Batch response format: ChatResponse JSON.
    """
    # ── Auth enforcement ──────────────────────────────────────────────────────
    if settings.enable_auth and not caller_id:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Include 'Authorization: Bearer <api_key>' header.",
        )

    user_id: Optional[str] = caller_id  # None in dev/unauthenticated mode

    start_time = time.time()

    # ── Security validation ───────────────────────────────────────────────────
    validated_question = validate_chat_input(
        request.question,
        request.session_id,
        max_length=settings.max_input_length,
    )

    chat_requests_total.labels(session_id=request.session_id).inc()

    log.info(
        "chat_request_received",
        user_id=user_id,
        session_id=request.session_id,
        question_length=len(validated_question),
    )

    try:
        if settings.ff_use_streaming:
            return StreamingResponse(
                _stream_response(validated_question, request.session_id, user_id=user_id),
                media_type="text/event-stream",
            )
        else:
            # Batch mode — wrap the sync LangGraph call in a thread so it doesn't
            # block the event loop while waiting for the OpenAI response.
            result = await asyncio.to_thread(
                graph.invoke,
                {
                    "question": validated_question,
                    "session_id": request.session_id,
                    "user_id": user_id,
                },
                {"configurable": {"thread_id": request.session_id}},
            )

            elapsed = time.time() - start_time
            chat_latency_seconds.observe(elapsed)

            log.info(
                "chat_request_completed",
                user_id=user_id,
                session_id=request.session_id,
                latency_seconds=round(elapsed, 3),
                sources_count=len(result["sources"]),
            )

            # Live sampling: evaluate ~10 % of responses asynchronously
            get_sampler().schedule(
                question=validated_question,
                answer=result["answer"],
                sources=result["sources"],
                session_id=request.session_id,
            )

            return ChatResponse(
                answer=result["answer"],
                sources=result["sources"],
                session_id=request.session_id,
                confidence=result.get("confidence"),
                confidence_level=result.get("confidence_level"),
            )

    except HTTPException:
        raise
    except Exception as e:
        log.error(
            "chat_request_failed",
            user_id=user_id,
            session_id=request.session_id,
            error=str(e),
            exc_info=True,
        )
        raise


async def _stream_response(
    validated_question: str,
    session_id: str,
    user_id: Optional[str] = None,
):
    """
    Async generator for the streaming chat path (SSE).

    Parity with the batch path:
    - Semantic cache short-circuit (from_cache)
    - Token budget and daily cost limit checks
    - Model fallback cascade (gpt-4o-mini → gpt-3.5-turbo on RateLimitError)
    - Cost tracking, session memory, semantic cache population
    - Live evaluation sampling
    - Confidence scoring in the final done event

    SSE envelope:
        data: {"token": "...", "done": false}   — one per streaming token
        data: {"sources": [...], "confidence": 0.8, "confidence_level": "high",
               "done": true}                    — single completion event
    """
    import tiktoken
    from openai import RateLimitError as OAIRateLimitError

    from src.agent.nodes import retrieve_node
    from src.agent.prompts import build_prompt
    from src.cost_control.cost_tracker import get_cost_tracker
    from src.cost_control.token_budget import get_token_budget
    from src.hitl.confidence import calculate_confidence, get_confidence_level

    _tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    # ── Step 1: Retrieval ─────────────────────────────────────────────────────
    # Wrapped in to_thread: ChromaDB queries + embedding calls are synchronous.
    state = await asyncio.to_thread(
        retrieve_node,
        {"question": validated_question, "session_id": session_id, "user_id": user_id},
    )

    # ── Step 2a: Cache hit — stream the pre-generated answer directly ─────────
    if state.get("from_cache"):
        cached_answer = state.get("answer", "")
        sources = state.get("sources", [])
        yield f"data: {json.dumps({'token': cached_answer, 'done': False})}\n\n"
        yield f"data: {json.dumps({'sources': sources, 'confidence': state.get('confidence'), 'confidence_level': state.get('confidence_level'), 'done': True})}\n\n"
        return

    # ── Step 2b: OpenAI disabled — return raw retrieved chunks ────────────────
    if not settings.ff_use_openai:
        raw = "\n\n".join([f"[{c['section_title']}]\n{c['content']}" for c in state["context"]])
        sources = list(set(c["source"] for c in state["context"]))
        yield f"data: {json.dumps({'token': raw, 'done': False})}\n\n"
        yield f"data: {json.dumps({'sources': sources, 'confidence': 0.5, 'confidence_level': 'medium', 'done': True})}\n\n"
        return

    # ── Step 2c: Pre-generation guards ────────────────────────────────────────
    messages = build_prompt(
        state["question"], state["context"], state["history"], session_id=session_id
    )
    estimated_input_tokens = sum(
        len(_tokenizer.encode(m.get("content", ""))) for m in messages
    )

    budget = get_token_budget()
    allowed, reason = budget.check_and_reserve(session_id, estimated_input_tokens)
    if not allowed:
        yield f"data: {json.dumps({'token': f'Daily token budget exceeded. {reason}', 'done': False})}\n\n"
        yield f"data: {json.dumps({'sources': [], 'confidence': 0.0, 'confidence_level': 'low', 'done': True})}\n\n"
        return

    tracker = get_cost_tracker()
    if not tracker.is_budget_available():
        yield f"data: {json.dumps({'token': 'Daily cost limit exceeded. System is in degraded mode.', 'done': False})}\n\n"
        yield f"data: {json.dumps({'sources': [], 'confidence': 0.0, 'confidence_level': 'low', 'done': True})}\n\n"
        return

    # ── Step 2d: Stream with model fallback cascade ───────────────────────────
    # gpt-4o-mini is the primary model; fall back to gpt-3.5-turbo on rate limits.
    cascade = [settings.llm_model]
    if settings.ff_use_model_fallback and "gpt-3.5-turbo" != settings.llm_model:
        cascade.append("gpt-3.5-turbo")

    async_client = AsyncOpenAI(api_key=settings.openai_api_key)
    full_answer = ""
    used_model = settings.llm_model
    output_tokens = 0

    for model in cascade:
        try:
            stream = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                stream=True,
                stream_options={"include_usage": True},
            )
            used_model = model
            async for chunk in stream:
                # Final usage chunk (include_usage=True) has empty choices
                if not chunk.choices:
                    if chunk.usage:
                        output_tokens = chunk.usage.completion_tokens
                    continue
                token = chunk.choices[0].delta.content or ""
                if token:
                    full_answer += token
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
            break  # Success — exit the cascade

        except OAIRateLimitError:
            if model != cascade[-1]:
                log.warning("stream_model_rate_limited", model=model, session_id=session_id)
                continue  # Try the next model in cascade
            # Exhausted the cascade
            log.error("stream_all_models_rate_limited", session_id=session_id)
            yield f"data: {json.dumps({'token': 'All AI models are currently rate-limited. Please try again shortly.', 'done': False})}\n\n"
            yield f"data: {json.dumps({'sources': [], 'confidence': 0.0, 'confidence_level': 'low', 'done': True})}\n\n"
            return
        except Exception as exc:
            log.error("stream_generation_failed", model=model, error=str(exc), session_id=session_id)
            yield f"data: {json.dumps({'token': 'Generation failed. Please try again.', 'done': False})}\n\n"
            yield f"data: {json.dumps({'sources': [], 'confidence': 0.0, 'confidence_level': 'low', 'done': True})}\n\n"
            return

    # ── Step 3: Post-generation bookkeeping ───────────────────────────────────
    if output_tokens == 0:
        # stream_options not supported by this model/proxy — estimate instead
        output_tokens = len(_tokenizer.encode(full_answer))
    total_tokens = estimated_input_tokens + output_tokens

    tracker.track_request(
        session_id=session_id,
        model=used_model,
        input_tokens=estimated_input_tokens,
        output_tokens=output_tokens,
        user_id=user_id,
    )
    chat_tokens_total.labels(model=used_model).inc(total_tokens)

    if settings.ff_use_session_memory:
        session_memory.add(session_id, validated_question, full_answer)

    # Confidence scoring (citation validation is skipped in the streaming path
    # because the full answer only exists after streaming completes)
    sources = list(set(c["source"] for c in state["context"]))
    top_retrieval_score = state["context"][0]["score"] if state["context"] else 0.0
    confidence = calculate_confidence(
        retrieval_score=top_retrieval_score,
        citation_valid=True,
        source_count=len(sources),
    )
    confidence_level = get_confidence_level(confidence)

    # Populate semantic cache so the next identical question gets a cache hit
    if settings.ff_use_semantic_cache:
        from src.advanced.semantic_cache import get_semantic_cache
        get_semantic_cache().set(
            question=validated_question,
            answer=full_answer,
            sources=sources,
            metadata={"confidence": confidence, "confidence_level": confidence_level},
        )

    # Live evaluation sampling (async — does not block the response)
    get_sampler().schedule(
        question=validated_question,
        answer=full_answer,
        sources=sources,
        session_id=session_id,
    )

    # ── Step 4: Completion event ──────────────────────────────────────────────
    yield f"data: {json.dumps({'sources': sources, 'confidence': confidence, 'confidence_level': confidence_level, 'done': True})}\n\n"


@app.get("/budget")
async def budget_status():
    """
    Current token budget and daily cost status.

    Useful for monitoring dashboards and alerting pipelines.
    Returns live counters — no caching.
    """
    from src.cost_control.cost_tracker import get_cost_tracker
    from src.cost_control.token_budget import get_token_budget

    budget = get_token_budget()
    tracker = get_cost_tracker()

    return {
        "token_budget": {
            "global_limit": budget.global_daily_limit,
            "global_used": budget._global_usage,
            "global_remaining": budget.global_daily_limit - budget._global_usage,
            "session_daily_limit": budget.session_daily_limit,
            "per_request_limit": budget.per_request_limit,
        },
        "cost": tracker.get_stats(),
        "circuit_breaker": {
            "state": str(openai_breaker.current_state),
            "fail_count": openai_breaker.fail_counter,
        },
    }


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Scrape this with Prometheus:
        scrape_configs:
          - job_name: 'k8s-rag-chatbot'
            static_configs:
              - targets: ['localhost:8000']
    """
    return Response(content=get_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    System health check — used by the Streamlit sidebar and Docker healthcheck.
    Returns live status of ChromaDB, BM25 index, and all feature flags.
    Status is "ok" if at least one index is available, "degraded" otherwise.
    """
    chroma_ok = False
    try:
        c = chromadb.PersistentClient(path=settings.chroma_db_path)
        c.get_collection(settings.chroma_collection_name)
        chroma_ok = True
    except Exception as e:
        log.warning("chroma_health_check_failed", error=str(e))

    bm25_ok = Path("./bm25_index/bm25.pkl").exists()

    index_health.labels(index_type="chroma").set(1 if chroma_ok else 0)
    index_health.labels(index_type="bm25").set(1 if bm25_ok else 0)

    feature_flag_status.labels(flag_name="use_chroma").set(1 if settings.ff_use_chroma else 0)
    feature_flag_status.labels(flag_name="use_openai").set(1 if settings.ff_use_openai else 0)
    feature_flag_status.labels(flag_name="use_session_memory").set(1 if settings.ff_use_session_memory else 0)
    feature_flag_status.labels(flag_name="use_streaming").set(1 if settings.ff_use_streaming else 0)

    feature_flags = {
        "use_chroma": settings.ff_use_chroma,
        "use_openai": settings.ff_use_openai,
        "use_session_memory": settings.ff_use_session_memory,
        "use_streaming": settings.ff_use_streaming,
    }

    status = "ok" if (chroma_ok or bm25_ok) else "degraded"

    log.info(
        "health_check",
        status=status,
        chroma_ok=chroma_ok,
        bm25_ok=bm25_ok,
    )

    return HealthResponse(
        status=status,
        chroma_ok=chroma_ok,
        bm25_ok=bm25_ok,
        feature_flags=feature_flags,
    )


@app.get("/healthz/ready")
async def readiness():
    """
    Readiness probe for Kubernetes.

    Returns 200 only if the service can handle traffic.
    Checks: ChromaDB availability, OpenAI client initialization.
    """
    try:
        c = chromadb.PersistentClient(path=settings.chroma_db_path)
        c.get_collection(settings.chroma_collection_name)
    except Exception:
        log.error("readiness_check_failed", reason="chroma_unavailable")
        return Response(status_code=503, content="ChromaDB not ready")

    if settings.ff_use_openai:
        try:
            from src.agent.nodes import _get_openai_client
            client = _get_openai_client()
            assert client.api_key is not None
        except Exception as e:
            log.error("readiness_check_failed", reason="openai_unavailable", error=str(e))
            return Response(status_code=503, content="OpenAI client not ready")

    return {"status": "ready"}


@app.get("/healthz/live")
async def liveness():
    """
    Liveness probe for Kubernetes.

    Returns 200 as long as the process is alive.
    No external dependencies checked.
    """
    return {"status": "alive"}


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """
    Clear all stored conversation history for the given session.
    Useful for debugging and for the UI's "Reset Session" button.
    """
    session_memory.clear(session_id)
    log.info("session_reset", session_id=session_id)
    return {"status": "cleared", "session_id": session_id}

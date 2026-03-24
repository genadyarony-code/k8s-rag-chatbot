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

import chromadb
from fastapi import FastAPI, HTTPException, Request
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
from src.observability.metrics import (
    chat_cost_usd_total,
    chat_latency_seconds,
    chat_requests_total,
    chat_tokens_total,
    feature_flag_status,
    get_metrics,
    index_health,
)
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
    check_index_health()
    yield
    log.info("application_shutdown")


app = FastAPI(title="K8s RAG Chatbot", version="1.0.0", lifespan=lifespan)

# Rate limiter — must be registered before any route decorators
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/chat")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def chat(request: ChatRequest, req: Request):
    """
    Main chat endpoint with security validation and metrics instrumentation.

    Returns either a streaming SSE response or a full JSON response depending
    on the ff_use_streaming feature flag.

    Streaming (SSE) response format:
        data: {"token": "...", "done": false}  ← one per token
        data: {"sources": [...], "done": true}  ← final signal

    Batch response format: ChatResponse JSON.
    """
    start_time = time.time()

    # ── Security validation ───────────────────────────────────────────────────
    # validate_chat_input sanitizes, checks for prompt injection, and redacts PII.
    # It raises HTTPException(400/413) on violations — re-raised automatically.
    validated_question = validate_chat_input(
        request.question,
        request.session_id,
        max_length=settings.max_input_length,
    )

    chat_requests_total.labels(session_id=request.session_id).inc()

    log.info(
        "chat_request_received",
        session_id=request.session_id,
        question_length=len(validated_question),
    )

    try:
        if settings.ff_use_streaming:
            return StreamingResponse(
                _stream_response(validated_question, request.session_id),
                media_type="text/event-stream",
            )
        else:
            # Batch mode — wrap the sync LangGraph call in a thread so it doesn't
            # block the event loop while waiting for the OpenAI response.
            result = await asyncio.to_thread(
                graph.invoke,
                {"question": validated_question, "session_id": request.session_id},
                {"configurable": {"thread_id": request.session_id}},
            )

            elapsed = time.time() - start_time
            chat_latency_seconds.observe(elapsed)

            log.info(
                "chat_request_completed",
                session_id=request.session_id,
                latency_seconds=round(elapsed, 3),
                sources_count=len(result["sources"]),
            )

            return ChatResponse(
                answer=result["answer"],
                sources=result["sources"],
                session_id=request.session_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        log.error(
            "chat_request_failed",
            session_id=request.session_id,
            error=str(e),
            exc_info=True,
        )
        raise


async def _stream_response(validated_question: str, session_id: str):
    """
    Async generator for the streaming chat path.

    Receives the already-validated question so that security checks are not
    duplicated on every token yield. Runs retrieval synchronously (CPU-bound +
    fast local ops), then streams GPT tokens via SSE using AsyncOpenAI.

    Args:
        validated_question: Sanitized + PII-redacted question
        session_id: Session identifier
    """
    from src.agent.nodes import retrieve_node
    from src.agent.prompts import build_prompt

    # Step 1: Retrieval (query routing + session history)
    # Wrapped in to_thread because ChromaDB and the embeddings call are sync.
    state = await asyncio.to_thread(
        retrieve_node,
        {"question": validated_question, "session_id": session_id},
    )

    # Step 2: Generation
    if not settings.ff_use_openai:
        # Degraded mode: return raw retrieved chunks without calling the LLM
        raw = "\n\n".join([f"[{c['section_title']}]\n{c['content']}" for c in state["context"]])
        yield f"data: {json.dumps({'token': raw, 'done': False})}\n\n"
    else:
        messages = build_prompt(state["question"], state["context"], state["history"])
        async_client = AsyncOpenAI(api_key=settings.openai_api_key)

        stream = await async_client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.1,
            stream=True,
        )

        full_answer = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                full_answer += token
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

        if settings.ff_use_session_memory:
            session_memory.add(session_id, validated_question, full_answer)

    # Step 3: Send sources as the completion signal
    sources = list(set(c["source"] for c in state["context"]))
    yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"


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

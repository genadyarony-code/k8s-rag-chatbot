# PHASE 1: Foundation & Infrastructure

> **Week:** 1  
> **Priority:** P0 (Deploy Blocking)  
> **Duration:** 3-4 days  
> **Dependencies:** None

---

## Objective

Transform logging from ad-hoc string formatting to structured JSON logs, add Prometheus metrics, and implement proper health checks.

**Why this matters:**  
Without structured logging and metrics, you're flying blind. This is the foundation for all observability.

---

## Pre-Flight Checklist

- [ ] Repository cloned locally
- [ ] Virtual environment active
- [ ] All existing tests passing: `pytest tests/ -v`
- [ ] Dependencies installed: `pip install -r requirements.txt`

---

## Task 1: Install Dependencies

**Add to `requirements.txt`:**

```
# Structured logging
structlog==24.1.0
python-json-logger==2.0.7

# Metrics
prometheus-client==0.20.0

# Enhanced HTTP client for health checks
httpx==0.27.0
```

**Install:**
```bash
pip install structlog==24.1.0 python-json-logger==2.0.7 prometheus-client==0.20.0 httpx==0.27.0
```

**Verify:**
```bash
python -c "import structlog; import prometheus_client; print('OK')"
```

---

## Task 2: Create Structured Logging Module

**Create:** `src/observability/__init__.py`

```python
# Empty init file
```

**Create:** `src/observability/logging_config.py`

```python
"""
Structured logging configuration using structlog.

All logs are emitted as JSON with consistent fields:
- timestamp (ISO 8601)
- level (debug/info/warning/error)
- logger (module name)
- event (human-readable message)
- context (structured data)

Example output:
{
  "timestamp": "2026-03-24T10:30:45.123Z",
  "level": "info",
  "logger": "src.agent.nodes",
  "event": "chat_request",
  "session_id": "abc123",
  "tokens": 1234,
  "cost_usd": 0.000185
}
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Set up structlog for the entire application.
    
    Call this once at application startup (in FastAPI lifespan or main.py).
    After calling this, all loggers will emit structured JSON.
    
    Args:
        log_level: Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """
    
    # Configure structlog processors
    structlog.configure(
        processors=[
            # Add log level to event dict
            structlog.stdlib.add_log_level,
            # Add logger name
            structlog.stdlib.add_logger_name,
            # Add timestamp in ISO format
            structlog.processors.TimeStamper(fmt="iso"),
            # Stack info for exceptions
            structlog.processors.StackInfoRenderer(),
            # Format exceptions
            structlog.processors.format_exc_info,
            # Render as JSON
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure Python's built-in logging to use structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str) -> Any:
    """
    Get a structured logger for a module.
    
    Usage:
        log = get_logger(__name__)
        log.info("user_login", user_id="abc123", method="api_key")
    
    Args:
        name: Usually __name__ to get the module path
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)
```

**Test it works:**

```python
# Test file: test_logging.py (temporary, run and delete)
from src.observability.logging_config import configure_logging, get_logger

configure_logging("INFO")
log = get_logger(__name__)

log.info("test_event", user_id="test123", tokens=100)
# Should output: {"timestamp": "...", "level": "info", "event": "test_event", ...}
```

---

## Task 3: Add Prometheus Metrics Module

**Create:** `src/observability/metrics.py`

```python
"""
Prometheus metrics for monitoring.

Metrics exported:
- chat_requests_total: Counter of all chat requests
- chat_latency_seconds: Histogram of request latencies
- chat_tokens_total: Counter of tokens used
- chat_cost_usd_total: Counter of cumulative cost
- retrieval_latency_seconds: Histogram of retrieval times
- generation_latency_seconds: Histogram of generation times
- chroma_query_total: Counter of ChromaDB queries
- bm25_query_total: Counter of BM25 fallback queries

Access at GET /metrics for Prometheus scraping.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# ── Chat Metrics ────────────────────────────────────────────────────────

chat_requests_total = Counter(
    "chat_requests_total",
    "Total number of chat requests",
    ["session_id"]
)

chat_latency_seconds = Histogram(
    "chat_latency_seconds",
    "Chat request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

chat_tokens_total = Counter(
    "chat_tokens_total",
    "Total tokens used",
    ["model"]
)

chat_cost_usd_total = Counter(
    "chat_cost_usd_total",
    "Cumulative cost in USD"
)

# ── Retrieval Metrics ───────────────────────────────────────────────────

retrieval_latency_seconds = Histogram(
    "retrieval_latency_seconds",
    "Retrieval operation latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

chroma_query_total = Counter(
    "chroma_query_total",
    "Total ChromaDB queries",
    ["doc_type_filter"]  # "troubleshooting", "None", etc.
)

bm25_query_total = Counter(
    "bm25_query_total",
    "Total BM25 fallback queries"
)

# ── Generation Metrics ──────────────────────────────────────────────────

generation_latency_seconds = Histogram(
    "generation_latency_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# ── System Metrics ──────────────────────────────────────────────────────

index_health = Gauge(
    "index_health",
    "Index health status (1=healthy, 0=degraded)",
    ["index_type"]  # "chroma" or "bm25"
)

feature_flag_status = Gauge(
    "feature_flag_status",
    "Feature flag status (1=enabled, 0=disabled)",
    ["flag_name"]
)


def get_metrics() -> bytes:
    """
    Generate Prometheus exposition format.
    
    Use in FastAPI endpoint:
        @app.get("/metrics")
        async def metrics():
            return Response(content=get_metrics(), media_type=CONTENT_TYPE_LATEST)
    """
    return generate_latest()
```

**Verify:**
```python
from src.observability.metrics import chat_requests_total, get_metrics

chat_requests_total.labels(session_id="test").inc()
output = get_metrics()
assert b"chat_requests_total" in output
print("Metrics OK")
```

---

## Task 4: Update FastAPI to Use Structured Logging

**Modify:** `src/api/main.py`

**Add imports at top:**
```python
from src.observability.logging_config import configure_logging, get_logger
from src.observability.metrics import (
    chat_requests_total,
    chat_latency_seconds,
    chat_tokens_total,
    chat_cost_usd_total,
    get_metrics,
)
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST
import time
```

**Replace old logging with:**
```python
log = get_logger(__name__)
```

**In `lifespan` function, add:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # NEW: Configure structured logging
    configure_logging(log_level="INFO")
    log.info("application_startup", version="1.0.0")
    
    check_index_health()  # existing
    yield
    
    # NEW: Log shutdown
    log.info("application_shutdown")
```

**Replace all `print()` statements with `log.info()` or `log.error()`**

Example in `check_index_health()`:
```python
# OLD
print(f"\033[92m✓ Index OK | chunks={meta.get('chunk_count')}\033[0m")

# NEW
log.info(
    "index_health_check_passed",
    chunk_count=meta.get("chunk_count"),
    version=meta.get("chunk_schema_version"),
    built_at=meta.get("built_at", "unknown")
)
```

---

## Task 5: Add Metrics to Chat Endpoint

**In `src/api/main.py`, update `@app.post("/chat")`:**

```python
@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint with metrics instrumentation.
    """
    
    # Start timer
    start_time = time.time()
    
    # Increment request counter
    chat_requests_total.labels(session_id=request.session_id).inc()
    
    log.info(
        "chat_request_received",
        session_id=request.session_id,
        question_length=len(request.question)
    )
    
    try:
        if settings.ff_use_streaming:
            return StreamingResponse(
                _stream_response(request),
                media_type="text/event-stream"
            )
        else:
            result = await asyncio.to_thread(
                graph.invoke,
                {"question": request.question, "session_id": request.session_id},
                {"configurable": {"thread_id": request.session_id}},
            )
            
            # Record metrics
            elapsed = time.time() - start_time
            chat_latency_seconds.observe(elapsed)
            
            log.info(
                "chat_request_completed",
                session_id=request.session_id,
                latency_seconds=elapsed,
                sources_count=len(result["sources"])
            )
            
            return ChatResponse(
                answer=result["answer"],
                sources=result["sources"],
                session_id=request.session_id
            )
    
    except Exception as e:
        log.error(
            "chat_request_failed",
            session_id=request.session_id,
            error=str(e),
            exc_info=True
        )
        raise
```

---

## Task 6: Add /metrics Endpoint

**In `src/api/main.py`, add new endpoint:**

```python
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
```

**Test:**
```bash
# Start the API
uvicorn src.api.main:app --reload --port 8000

# In another terminal
curl http://localhost:8000/metrics

# Should see:
# chat_requests_total{session_id="..."} 0.0
# chat_latency_seconds_bucket{le="0.1"} 0.0
# ...
```

---

## Task 7: Instrument Agent Nodes

**Modify:** `src/agent/nodes.py`

**Add imports:**
```python
from src.observability.logging_config import get_logger
from src.observability.metrics import (
    retrieval_latency_seconds,
    generation_latency_seconds,
    chroma_query_total,
    bm25_query_total,
    chat_tokens_total,
    chat_cost_usd_total,
)
import time
```

**Replace `logger = logging.getLogger(__name__)` with:**
```python
log = get_logger(__name__)
```

**In `retrieve_node()`, add timing:**

```python
def retrieve_node(state: dict) -> dict:
    start_time = time.time()
    
    question = state["question"]
    session_id = state["session_id"]

    if settings.ff_use_chroma:
        doc_type = _detect_doc_type(question)
        context = _chroma_search(question, doc_type_filter=doc_type)
        
        # Record metric
        chroma_query_total.labels(doc_type_filter=str(doc_type)).inc()
        
        ROUTING_SCORE_THRESHOLD = 0.3
        if not context or all(c["score"] < ROUTING_SCORE_THRESHOLD for c in context):
            log.warning(
                "routing_fallback_triggered",
                session_id=session_id,
                doc_type_filter=doc_type,
                threshold=ROUTING_SCORE_THRESHOLD
            )
            context = _chroma_search(question)
            chroma_query_total.labels(doc_type_filter="fallback").inc()
    else:
        context = _bm25_search(question)
        bm25_query_total.inc()

    history = session_memory.get(session_id) if settings.ff_use_session_memory else []
    
    # Record latency
    elapsed = time.time() - start_time
    retrieval_latency_seconds.observe(elapsed)
    
    log.info(
        "retrieval_completed",
        session_id=session_id,
        chunks_retrieved=len(context),
        latency_seconds=elapsed,
        used_chroma=settings.ff_use_chroma
    )

    return {**state, "context": context, "history": history}
```

**In `generate_node()`, add cost tracking:**

```python
def generate_node(state: dict) -> dict:
    start_time = time.time()
    
    if not settings.ff_use_openai:
        raw = "\n\n".join([f"[{c['section_title']}]\n{c['content']}" for c in state["context"]])
        return {**state, "answer": raw, "sources": [c["source"] for c in state["context"]]}

    messages = build_prompt(state["question"], state["context"], state["history"])

    token_count = sum(len(_tokenizer.encode(m.get("content", ""))) for m in messages)
    
    log.info(
        "generation_started",
        session_id=state["session_id"],
        input_tokens=token_count
    )

    client = _get_openai_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.1,
        stream=False
    )

    answer = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    cost_usd = total_tokens * 0.00000015  # gpt-4o-mini pricing
    
    # Record metrics
    chat_tokens_total.labels(model=settings.llm_model).inc(total_tokens)
    chat_cost_usd_total.inc(cost_usd)
    
    elapsed = time.time() - start_time
    generation_latency_seconds.observe(elapsed)
    
    log.info(
        "generation_completed",
        session_id=state["session_id"],
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_seconds=elapsed
    )

    sources = list(set(c["source"] for c in state["context"]))

    if settings.ff_use_session_memory:
        session_memory.add(state["session_id"], state["question"], answer)

    return {**state, "answer": answer, "sources": sources}
```

---

## Task 8: Enhanced Health Endpoint

**Modify:** `src/api/main.py`

**Update `/health` endpoint:**

```python
@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Enhanced health check with readiness vs liveness.
    
    Returns:
        - status: "ok" (ready) | "degraded" (alive but limited) | "error" (not ready)
        - chroma_ok: ChromaDB availability
        - bm25_ok: BM25 index availability
        - feature_flags: Current flag states
    """
    
    # Check ChromaDB
    chroma_ok = False
    try:
        c = chromadb.PersistentClient(path=settings.chroma_db_path)
        c.get_collection(settings.chroma_collection_name)
        chroma_ok = True
    except Exception as e:
        log.warning("chroma_health_check_failed", error=str(e))

    # Check BM25
    bm25_ok = Path("./bm25_index/bm25.pkl").exists()
    
    # Update Prometheus gauges
    from src.observability.metrics import index_health, feature_flag_status
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
    
    # Determine status
    if chroma_ok or bm25_ok:
        status = "ok"
    else:
        status = "degraded"
    
    log.info(
        "health_check",
        status=status,
        chroma_ok=chroma_ok,
        bm25_ok=bm25_ok
    )

    return HealthResponse(
        status=status,
        chroma_ok=chroma_ok,
        bm25_ok=bm25_ok,
        feature_flags=feature_flags
    )
```

---

## Task 9: Add Kubernetes Readiness/Liveness Endpoints

**In `src/api/main.py`, add:**

```python
@app.get("/healthz/ready")
async def readiness():
    """
    Readiness probe for Kubernetes.
    
    Returns 200 only if the service can handle traffic.
    Checks: index availability, OpenAI connectivity.
    """
    
    # Check index
    try:
        c = chromadb.PersistentClient(path=settings.chroma_db_path)
        c.get_collection(settings.chroma_collection_name)
    except Exception:
        log.error("readiness_check_failed", reason="chroma_unavailable")
        return Response(status_code=503, content="ChromaDB not ready")
    
    # Check OpenAI (quick test)
    if settings.ff_use_openai:
        try:
            from src.agent.nodes import _get_openai_client
            client = _get_openai_client()
            # Quick validation - this doesn't make an API call
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
```

---

## Task 10: Update Docker Healthcheck

**Modify:** `docker-compose.yml`

**Update API service healthcheck:**

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./chroma_db:/app/chroma_db
      - ./bm25_index:/app/bm25_index
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz/ready"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

## Task 11: Write Tests

**Create:** `tests/test_observability.py`

```python
"""
Tests for structured logging and metrics.
"""

import pytest
from src.observability.logging_config import configure_logging, get_logger
from src.observability.metrics import (
    chat_requests_total,
    chat_latency_seconds,
    get_metrics
)


def test_structured_logging():
    """Test that logs are emitted as structured JSON."""
    configure_logging("INFO")
    log = get_logger("test")
    
    # This should not raise
    log.info("test_event", user_id="test123", count=42)


def test_metrics_increment():
    """Test that Prometheus counters work."""
    initial = chat_requests_total.labels(session_id="test")._value.get()
    
    chat_requests_total.labels(session_id="test").inc()
    
    after = chat_requests_total.labels(session_id="test")._value.get()
    assert after == initial + 1


def test_metrics_export():
    """Test that metrics can be exported."""
    output = get_metrics()
    
    assert b"chat_requests_total" in output
    assert b"chat_latency_seconds" in output
```

**Run:**
```bash
pytest tests/test_observability.py -v
```

---

## Verification Steps

**1. Start the API:**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**2. Check structured logs:**

All output should be valid JSON:
```json
{"timestamp": "2026-03-24T10:30:45.123Z", "level": "info", "event": "application_startup", ...}
```

**3. Test metrics endpoint:**
```bash
curl http://localhost:8000/metrics

# Should return Prometheus exposition format:
# chat_requests_total{session_id="..."} 0.0
# chat_latency_seconds_bucket{le="0.1"} 0.0
```

**4. Test health endpoints:**
```bash
curl http://localhost:8000/health         # Full health check
curl http://localhost:8000/healthz/ready  # Readiness
curl http://localhost:8000/healthz/live   # Liveness
```

**5. Send a test chat request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a Pod?",
    "session_id": "test123"
  }'
```

**6. Check metrics again:**
```bash
curl http://localhost:8000/metrics | grep chat_requests_total

# Should show counter incremented:
# chat_requests_total{session_id="test123"} 1.0
```

**7. Run all tests:**
```bash
pytest tests/ -v
```

---

## Success Criteria

- [ ] All logs are valid JSON
- [ ] `/metrics` endpoint returns Prometheus format
- [ ] Counters increment on requests
- [ ] Histograms record latencies
- [ ] `/healthz/ready` returns 200 when healthy
- [ ] `/healthz/live` always returns 200
- [ ] All existing tests still pass
- [ ] New observability tests pass

---

## Rollback Plan

If anything breaks:

```bash
git diff HEAD  # Review changes
git checkout -- .  # Revert all changes
pip install -r requirements.txt  # Reinstall original deps
```

---

## Next Phase

Once this phase is complete and verified:

**→ Proceed to `PHASE_02_SECURITY.md`**

Security builds on the logging/metrics foundation you just created.

---

**Questions? Issues?**

- Check logs for error messages (they're structured now!)
- Review `src/observability/logging_config.py` docstrings
- Verify all imports are correct

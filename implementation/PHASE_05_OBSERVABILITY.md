# PHASE 5: Advanced Observability

> **Week:** 3-4  
> **Priority:** P1 (Production Readiness)  
> **Duration:** 5-6 days  
> **Dependencies:** PHASE 1 (metrics), PHASE 4 (user_id in logs)

---

## Objective

Implement distributed tracing (OpenTelemetry), Grafana dashboards, Prometheus alert rules, and log aggregation.

**Why this matters:**  
From reference: *"If you can't measure it, you can't improve it."*  
Right now you have logs and metrics, but no visualization or alerting.

---

## Pre-Flight Checklist

- [ ] PHASE 1-4 completed
- [ ] Prometheus `/metrics` endpoint working
- [ ] Structured logging enabled
- [ ] All tests passing

---

## Task 1: Install Dependencies

**Add to `requirements.txt`:**

```
# OpenTelemetry
opentelemetry-api==1.22.0
opentelemetry-sdk==1.22.0
opentelemetry-instrumentation-fastapi==0.43b0
opentelemetry-exporter-otlp==1.22.0

# Optional: Jaeger exporter (alternative to OTLP)
opentelemetry-exporter-jaeger==1.22.0
```

**Install:**
```bash
pip install opentelemetry-api==1.22.0 opentelemetry-sdk==1.22.0 \
            opentelemetry-instrumentation-fastapi==0.43b0 \
            opentelemetry-exporter-otlp==1.22.0
```

---

## Task 2: Configure OpenTelemetry

**Create:** `src/observability/tracing.py`

```python
"""
Distributed tracing with OpenTelemetry.

Traces:
- HTTP requests (automatic via FastAPI instrumentation)
- LLM calls
- Retrieval operations
- Cost calculations

Export to:
- Jaeger (local development)
- OTLP (production - sends to collectors like Grafana Tempo)
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from src.config.settings import settings
from src.observability.logging_config import get_logger

log = get_logger(__name__)


def configure_tracing(app=None):
    """
    Configure OpenTelemetry tracing.
    
    Args:
        app: FastAPI app instance (for auto-instrumentation)
    """
    
    # Create resource (identifies this service)
    resource = Resource.create({
        "service.name": "k8s-rag-chatbot",
        "service.version": "1.0.0",
        "deployment.environment": getattr(settings, 'environment', 'development'),
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure exporter (OTLP for production, console for dev)
    if getattr(settings, 'otlp_endpoint', None):
        # Production: send to OTLP collector
        exporter = OTLPSpanExporter(
            endpoint=settings.otlp_endpoint,
            insecure=True  # Use TLS in production
        )
        log.info("tracing_configured", exporter="otlp", endpoint=settings.otlp_endpoint)
    else:
        # Development: log spans to console
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()
        log.info("tracing_configured", exporter="console")
    
    # Add span processor
    provider.add_span_processor(BatchSpanProcessor(exporter))
    
    # Set as global default
    trace.set_tracer_provider(provider)
    
    # Auto-instrument FastAPI if app provided
    if app:
        FastAPIInstrumentor.instrument_app(app)
        log.info("fastapi_instrumented", message="Auto-tracing enabled for HTTP requests")
    
    return provider


def get_tracer(name: str = __name__):
    """
    Get a tracer for manual instrumentation.
    
    Usage:
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("my_operation"):
            # ... do work
    """
    return trace.get_tracer(name)
```

**Update settings:**

In `src/config/settings.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # ── Observability ─────────────────────────────────────────────────────
    # OpenTelemetry
    otlp_endpoint: Optional[str] = None  # e.g., "http://localhost:4317"
    environment: str = "development"  # development | staging | production
    
    # Tracing
    tracing_enabled: bool = True
```

---

## Task 3: Instrument Agent with Tracing

**Modify:** `src/agent/nodes.py`

**Add imports:**
```python
from src.observability.tracing import get_tracer

tracer = get_tracer(__name__)
```

**Add tracing to retrieve_node:**

```python
def retrieve_node(state: dict) -> dict:
    with tracer.start_as_current_span("retrieve_node") as span:
        start_time = time.time()
        
        question = state["question"]
        session_id = state["session_id"]
        
        # Add span attributes
        span.set_attribute("session_id", session_id)
        span.set_attribute("question_length", len(question))
        
        if settings.ff_use_chroma:
            with tracer.start_as_current_span("detect_doc_type"):
                doc_type = _detect_doc_type(question)
                span.set_attribute("doc_type_filter", str(doc_type))
            
            with tracer.start_as_current_span("chroma_search"):
                context = _chroma_search(question, doc_type_filter=doc_type)
                span.set_attribute("chunks_retrieved", len(context))
            
            # ... rest of code
```

**Add tracing to generate_node:**

```python
def generate_node(state: dict) -> dict:
    with tracer.start_as_current_span("generate_node") as span:
        start_time = time.time()
        
        span.set_attribute("session_id", state["session_id"])
        span.set_attribute("use_openai", settings.ff_use_openai)
        
        if not settings.ff_use_openai:
            # ... degraded mode
        
        # ... existing code ...
        
        with tracer.start_as_current_span("openai_completion") as llm_span:
            token_count = sum(len(_tokenizer.encode(m.get("content", ""))) for m in messages)
            llm_span.set_attribute("input_tokens", token_count)
            llm_span.set_attribute("model", settings.llm_model)
            
            try:
                response = _call_openai_completion(...)
                
                llm_span.set_attribute("total_tokens", response.usage.total_tokens)
                llm_span.set_attribute("output_tokens", response.usage.completion_tokens)
            except CircuitBreakerError:
                llm_span.set_attribute("circuit_breaker", "open")
                llm_span.set_status(Status(StatusCode.ERROR, "Circuit breaker open"))
                raise
        
        # ... rest of code
```

---

## Task 4: Create Grafana Dashboards

**Create:** `monitoring/grafana/dashboards/health_dashboard.json`

```json
{
  "dashboard": {
    "title": "K8s RAG Chatbot - Health",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(chat_requests_total[5m])",
            "legendFormat": "{{session_id}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(chat_requests_total{status=\"error\"}[5m])",
            "legendFormat": "Errors/sec"
          }
        ],
        "type": "graph"
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(chat_latency_seconds_bucket[5m]))",
            "legendFormat": "P95"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Index Health",
        "targets": [
          {
            "expr": "index_health",
            "legendFormat": "{{index_type}}"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

**Create:** `monitoring/grafana/dashboards/cost_dashboard.json`

```json
{
  "dashboard": {
    "title": "K8s RAG Chatbot - Cost",
    "panels": [
      {
        "title": "Daily Cost (USD)",
        "targets": [
          {
            "expr": "chat_cost_usd_total",
            "legendFormat": "Cumulative Cost"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Token Usage by Model",
        "targets": [
          {
            "expr": "rate(chat_tokens_total[1h])",
            "legendFormat": "{{model}}"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Cost per Request",
        "targets": [
          {
            "expr": "rate(chat_cost_usd_total[5m]) / rate(chat_requests_total[5m])",
            "legendFormat": "Avg Cost/Request"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Budget Remaining",
        "targets": [
          {
            "expr": "50 - chat_cost_usd_total",
            "legendFormat": "Remaining (USD)"
          }
        ],
        "type": "gauge"
      }
    ]
  }
}
```

**Create:** `monitoring/grafana/dashboards/quality_dashboard.json`

```json
{
  "dashboard": {
    "title": "K8s RAG Chatbot - Quality",
    "panels": [
      {
        "title": "Retrieval Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(retrieval_latency_seconds_bucket[5m]))",
            "legendFormat": "P95 Retrieval"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Generation Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(generation_latency_seconds_bucket[5m]))",
            "legendFormat": "P95 Generation"
          }
        ],
        "type": "graph"
      },
      {
        "title": "ChromaDB vs BM25 Usage",
        "targets": [
          {
            "expr": "rate(chroma_query_total[5m])",
            "legendFormat": "ChromaDB"
          },
          {
            "expr": "rate(bm25_query_total[5m])",
            "legendFormat": "BM25"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

---

## Task 5: Create Prometheus Alert Rules

**Create:** `monitoring/prometheus/alerts.yml`

```yaml
groups:
  - name: k8s_rag_chatbot_alerts
    interval: 30s
    rules:
      # ── Error Rate Alerts ───────────────────────────────────────────
      
      - alert: HighErrorRate
        expr: |
          (
            rate(chat_requests_total{status="error"}[5m]) 
            / 
            rate(chat_requests_total[5m])
          ) > 0.05
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
      
      - alert: CriticalErrorRate
        expr: |
          (
            rate(chat_requests_total{status="error"}[5m]) 
            / 
            rate(chat_requests_total[5m])
          ) > 0.20
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "CRITICAL: Very high error rate"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 20%)"
      
      # ── Latency Alerts ──────────────────────────────────────────────
      
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95, rate(chat_latency_seconds_bucket[5m])) > 10
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 10s)"
      
      # ── Cost Alerts ─────────────────────────────────────────────────
      
      - alert: CostWarningThreshold
        expr: chat_cost_usd_total > 10
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Daily cost warning threshold exceeded"
          description: "Cost is ${{ $value }} (threshold: $10)"
      
      - alert: CostHardLimit
        expr: chat_cost_usd_total > 50
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CRITICAL: Daily cost hard limit exceeded"
          description: "Cost is ${{ $value }} (limit: $50). System should be in degraded mode."
      
      # ── Index Health Alerts ─────────────────────────────────────────
      
      - alert: IndexUnhealthy
        expr: index_health{index_type="chroma"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "ChromaDB index is unhealthy"
          description: "System may be running in degraded mode (BM25 fallback)"
      
      # ── Circuit Breaker Alerts ──────────────────────────────────────
      
      - alert: CircuitBreakerOpen
        expr: |
          # This requires custom metric - add in next task
          circuit_breaker_state{breaker="openai_api"} == 2  # 2 = OPEN
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "OpenAI circuit breaker is open"
          description: "OpenAI service may be down or rate limited"
```

---

## Task 6: Add Circuit Breaker State Metric

**Modify:** `src/observability/metrics.py`

**Add:**
```python
# ── Circuit Breaker Metrics ─────────────────────────────────────────────

circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["breaker"]
)
```

**Modify:** `src/cost_control/circuit_breaker.py`

**Update listeners:**
```python
from src.observability.metrics import circuit_breaker_state

def on_circuit_open(breaker, *args, **kwargs):
    log.error(...)
    circuit_breaker_state.labels(breaker=breaker.name).set(2)  # ← ADD THIS


def on_circuit_half_open(breaker, *args, **kwargs):
    log.warning(...)
    circuit_breaker_state.labels(breaker=breaker.name).set(1)  # ← ADD THIS


def on_circuit_close(breaker, *args, **kwargs):
    log.info(...)
    circuit_breaker_state.labels(breaker=breaker.name).set(0)  # ← ADD THIS
```

---

## Task 7: Create Docker Compose for Monitoring Stack

**Create:** `docker-compose.monitoring.yml`

```yaml
version: '3.8'

services:
  # ── Prometheus ────────────────────────────────────────────────────────
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
  
  # ── Grafana ───────────────────────────────────────────────────────────
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
      - grafana-data:/var/lib/grafana
    restart: unless-stopped
  
  # ── Jaeger (Tracing) ──────────────────────────────────────────────────
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "4317:4317"    # OTLP gRPC
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    restart: unless-stopped
  
  # ── Alertmanager (Optional) ───────────────────────────────────────────
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

**Create:** `monitoring/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alert rules
rule_files:
  - /etc/prometheus/alerts.yml

# Scrape configs
scrape_configs:
  - job_name: 'k8s-rag-chatbot'
    static_configs:
      - targets: ['host.docker.internal:8000']  # Adjust for your setup
    metrics_path: /metrics
```

**Create:** `monitoring/grafana/datasources/prometheus.yml`

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

---

## Task 8: Integrate Tracing into FastAPI

**Modify:** `src/api/main.py`

**Add imports:**
```python
from src.observability.tracing import configure_tracing
```

**In `lifespan` function:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure logging (existing)
    configure_logging(log_level="INFO")
    log.info("application_startup", version="1.0.0")
    
    # NEW: Configure tracing
    if settings.tracing_enabled:
        configure_tracing(app)
        log.info("tracing_enabled", otlp_endpoint=getattr(settings, 'otlp_endpoint', 'console'))
    
    check_index_health()
    yield
    
    log.info("application_shutdown")
```

---

## Verification Steps

**1. Start monitoring stack:**

```bash
docker-compose -f docker-compose.monitoring.yml up -d

# Check status
docker-compose -f docker-compose.monitoring.yml ps
```

**2. Access Prometheus:**

```bash
open http://localhost:9090

# Try queries:
# - chat_requests_total
# - rate(chat_latency_seconds_bucket[5m])
# - chat_cost_usd_total
```

**3. Access Grafana:**

```bash
open http://localhost:3000

# Login: admin / admin
# Navigate to Dashboards
# Import the 3 dashboards from monitoring/grafana/dashboards/
```

**4. Access Jaeger (tracing UI):**

```bash
open http://localhost:16686

# Send a chat request, then search for traces in Jaeger
```

**5. Test tracing:**

```bash
# Set OTLP endpoint
export OTLP_ENDPOINT="http://localhost:4317"

# Restart API
uvicorn src.api.main:app --reload --port 8000

# Send request
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Pod?", "session_id": "trace_test"}'

# Check Jaeger UI - should see trace with spans:
# - HTTP request
# - retrieve_node
#   - detect_doc_type
#   - chroma_search
# - generate_node
#   - openai_completion
```

**6. Test alerts:**

```bash
# Trigger cost alert (simulate)
# In Python console:
from src.observability.metrics import chat_cost_usd_total
chat_cost_usd_total.inc(15)  # Exceeds $10 warning

# Check Prometheus alerts
open http://localhost:9090/alerts

# Should see "CostWarningThreshold" firing
```

**7. Verify dashboards:**

- **Health Dashboard**: Request rate, error rate, latency, index health
- **Cost Dashboard**: Daily cost, token usage, budget remaining
- **Quality Dashboard**: Retrieval/generation latency, ChromaDB vs BM25

---

## Success Criteria

- [ ] Prometheus scraping metrics from `/metrics`
- [ ] Grafana dashboards displaying live data
- [ ] Jaeger showing distributed traces
- [ ] Alerts firing on test conditions
- [ ] Traces include user_id, session_id, token counts
- [ ] All 3 dashboards functional

---

## Common Issues

**Issue:** Prometheus can't reach API  
**Fix:** Use `host.docker.internal` instead of `localhost` in `prometheus.yml`

**Issue:** No traces in Jaeger  
**Fix:** Verify `OTLP_ENDPOINT` is set, check logs for tracing errors

**Issue:** Grafana dashboards empty  
**Fix:** Check Prometheus datasource is connected, verify metrics are being scraped

---

## Next Phase

Once verified:

**→ Proceed to `PHASE_06_RAG_QUALITY.md`**

RAG quality improvements (reranking, hybrid search) build on observability (need metrics to measure improvement).

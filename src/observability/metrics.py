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
- index_health: Gauge for index availability
- feature_flag_status: Gauge for each feature flag
- circuit_breaker_state: Gauge for circuit breaker state (0=closed,1=half-open,2=open)

Access at GET /metrics for Prometheus scraping.
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── Chat Metrics ─────────────────────────────────────────────────────────────

chat_requests_total = Counter(
    "chat_requests_total",
    "Total number of chat requests",
    ["session_id"],
)

chat_latency_seconds = Histogram(
    "chat_latency_seconds",
    "Chat request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

chat_tokens_total = Counter(
    "chat_tokens_total",
    "Total tokens used",
    ["model"],
)

chat_cost_usd_total = Counter(
    "chat_cost_usd_total",
    "Cumulative cost in USD",
)

# ── Retrieval Metrics ────────────────────────────────────────────────────────

retrieval_latency_seconds = Histogram(
    "retrieval_latency_seconds",
    "Retrieval operation latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

chroma_query_total = Counter(
    "chroma_query_total",
    "Total ChromaDB queries",
    ["doc_type_filter"],
)

bm25_query_total = Counter(
    "bm25_query_total",
    "Total BM25 fallback queries",
)

# ── Generation Metrics ───────────────────────────────────────────────────────

generation_latency_seconds = Histogram(
    "generation_latency_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# ── System Metrics ───────────────────────────────────────────────────────────

index_health = Gauge(
    "index_health",
    "Index health status (1=healthy, 0=degraded)",
    ["index_type"],
)

feature_flag_status = Gauge(
    "feature_flag_status",
    "Feature flag status (1=enabled, 0=disabled)",
    ["flag_name"],
)

# ── Circuit Breaker Metrics ───────────────────────────────────────────────────
# State encoding: 0 = CLOSED (normal), 1 = HALF-OPEN (testing), 2 = OPEN (blocked)

circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=half-open, 2=open)",
    ["breaker"],
)

# ── RAG Quality Metrics ───────────────────────────────────────────────────────

reranking_improvement = Histogram(
    "reranking_improvement",
    "Score delta between rerank score and original vector/RRF score",
    buckets=[-1.0, -0.5, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
)

hybrid_search_fusion_score = Histogram(
    "hybrid_search_fusion_score",
    "RRF fusion score of the top-ranked document",
    buckets=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
)

citation_validation_failures = Counter(
    "citation_validation_failures_total",
    "Number of generated answers that contained at least one unsupported sentence",
)

query_decomposition_count = Counter(
    "query_decomposition_total",
    "Number of queries that triggered LLM-based decomposition",
    ["sub_query_count"],
)


# ── Evaluation / A/B Metrics ─────────────────────────────────────────────────

eval_score = Histogram(
    "eval_score",
    "LLM-as-judge scores for sampled production responses",
    ["metric"],  # accuracy | completeness | relevance | grounding | overall
    buckets=[1, 2, 3, 4, 5],
)

ab_test_assignments = Counter(
    "ab_test_assignments_total",
    "A/B test variant assignments",
    ["test_id", "variant"],
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

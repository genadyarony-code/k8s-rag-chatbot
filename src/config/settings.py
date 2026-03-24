"""
All config in one place, loaded once at import time. Values come from environment
variables or a .env file — pydantic-settings handles loading and type coercion.

Worth calling out: increment chunk_schema_version any time you change the chunking
logic. Both the indexer and the API check it on startup and will fail loudly on a
mismatch instead of silently mixing old and new chunks in the same collection.

Feature flags are co-located here deliberately — they're just env-var-backed booleans
and belong in the same typed config layer as the rest of the settings.
"""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str

    # 1536-dim embeddings, ~$0.02 / million tokens — ideal for technical text
    embedding_model: str = "text-embedding-3-small"
    # Fast, cheap, sufficient quality for grounded RAG answers
    llm_model: str = "gpt-4o-mini"

    # ── Vector DB ─────────────────────────────────────────────────────────────
    chroma_db_path: str = "./chroma_db"
    chroma_collection_name: str = "k8s_docs"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_results: int = 5        # chunks returned per query
    chunk_size: int = 1000        # characters per chunk
    chunk_overlap: int = 200      # overlap between consecutive chunks

    # ⚠️ Increment this whenever the chunking strategy changes.
    # The indexer compares stored vs. current version on startup and
    # fails fast with clear instructions if they don't match.
    chunk_schema_version: str = "v1.0"

    # ── Feature Flags — operational killswitches ───────────────────────────────
    # Plan these at calm time; flip them under pressure without touching code.
    # ff_use_chroma=false       → BM25 keyword search only
    # ff_use_openai=false       → return raw retrieved chunks, no LLM call
    # ff_use_session_memory=false → stateless mode, no conversation history
    # ff_use_streaming=false    → batch JSON response instead of SSE
    ff_use_chroma: bool = True
    ff_use_openai: bool = True
    ff_use_session_memory: bool = True
    ff_use_streaming: bool = True

    # ── Security ──────────────────────────────────────────────────────────────
    # Maximum input length in characters (5K is sufficient for any K8s question)
    max_input_length: int = 5000

    # Individual security features can be disabled for dev / load-testing.
    # Never disable in production without an explicit reason documented here.
    security_enable_prompt_injection_check: bool = True
    security_enable_pii_detection: bool = True
    security_enable_rate_limiting: bool = True

    # Rate limits enforced per session_id (fallback: client IP)
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 100

    # ── Cost Controls ─────────────────────────────────────────────────────────
    # Token budgets — first line of defence against runaway usage
    token_budget_session_daily: int = 100_000   # tokens per session per day
    token_budget_global_daily: int = 1_000_000  # total tokens per day across all sessions
    token_budget_per_request: int = 10_000      # maximum tokens estimated per single request

    # Cost limits in USD — triggers warning log then degraded mode
    cost_warning_threshold_usd: float = 10.0   # log a warning when daily cost exceeds this
    cost_hard_limit_usd: float = 50.0          # block OpenAI calls when daily cost exceeds this

    # Circuit breaker — prevents hammering a failing OpenAI endpoint
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5    # failures before opening
    circuit_breaker_timeout_seconds: int = 60     # seconds to stay open before half-open test

    # ConfigDict replaces the deprecated inner `class Config` (Pydantic V2)
    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

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

    # ConfigDict replaces the deprecated inner `class Config` (Pydantic V2)
    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

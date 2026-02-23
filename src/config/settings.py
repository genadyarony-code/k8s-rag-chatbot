"""
All config in one place, loaded once at import time. Values come from environment
variables or a .env file — pydantic-settings handles loading and type coercion.

Worth calling out: increment chunk_schema_version any time you change the chunking
logic. Both the indexer and the API check it on startup and will fail loudly on a
mismatch instead of silently mixing old and new chunks in the same collection.
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

    # ConfigDict replaces the deprecated inner `class Config` (Pydantic V2)
    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()

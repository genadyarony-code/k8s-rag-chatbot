"""
Request/response models for the API. Pydantic handles input validation,
so malformed requests get a 422 before they reach any business logic.
"""

import uuid
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload for POST /chat."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's Kubernetes question.",
    )
    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique session identifier. Auto-generated if not provided.",
    )


class ChatResponse(BaseModel):
    """Response for POST /chat when FF_USE_STREAMING is disabled (batch mode)."""
    answer: str
    sources: list[str]   # filenames of documents cited in the answer
    session_id: str


class HealthResponse(BaseModel):
    """Response for GET /health — used by the UI sidebar and Docker healthcheck."""
    status: str          # "ok" if at least one index available, "degraded" otherwise
    chroma_ok: bool      # True if ChromaDB collection is accessible
    bm25_ok: bool        # True if bm25_index/bm25.pkl exists on disk
    feature_flags: dict  # current runtime values of all feature flags

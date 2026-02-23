"""
Four on/off switches that let you disable parts of the system at runtime without
touching code. The point is to have these ready before something breaks, not to
scramble adding them when the OpenAI API goes down at 2am.

    use_chroma=false          → falls back to BM25 keyword search
    use_openai=false          → returns raw retrieved chunks, no LLM call
    use_session_memory=false  → stateless, every question starts fresh
    use_streaming=false       → returns a single JSON response instead of SSE

To flip one without redeploying:
    echo "FF_USE_OPENAI=false" >> .env && docker compose restart api
"""

import os


def _parse_bool(val, default: bool) -> bool:
    """Coerce env-var string to bool; return default when var is unset."""
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


FEATURE_FLAGS = {
    "use_chroma":         _parse_bool(os.getenv("FF_USE_CHROMA"),         True),
    "use_openai":         _parse_bool(os.getenv("FF_USE_OPENAI"),         True),
    "use_session_memory": _parse_bool(os.getenv("FF_USE_SESSION_MEMORY"), True),
    "use_streaming":      _parse_bool(os.getenv("FF_USE_STREAMING"),      True),
}

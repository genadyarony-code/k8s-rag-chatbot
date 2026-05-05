"""
Rate limiting using slowapi (Starlette / FastAPI integration).

Default limits:
- 10 requests per minute per session_id (burst protection)
- 100 requests per hour per session_id (sustained usage)

Rate limit key is the session_id from the request body, falling back to the
client IP address when the body is not JSON or the field is absent.

Why rate limit per session_id rather than per IP?
- Multiple users may share an IP (NAT, office networks)
- session_id is the logical unit of usage in this system
- IP fallback protects against unauthenticated callers
"""

import json

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.observability.logging_config import get_logger

log = get_logger(__name__)


def get_session_id(request: Request) -> str:
    """
    Extract session_id from request body for use as the rate limit key.

    slowapi calls this function synchronously, so we use the cached body
    (available after the request body has been read by FastAPI).
    Falls back to the client IP address if session_id cannot be extracted.
    """
    try:
        body_bytes = getattr(request, "_body", None)
        if body_bytes:
            data = json.loads(body_bytes.decode())
            if session_id := data.get("session_id"):
                return session_id
    except Exception:
        pass

    return get_remote_address(request)


limiter = Limiter(
    key_func=get_session_id,
    default_limits=["100 per hour", "10 per minute"],
)

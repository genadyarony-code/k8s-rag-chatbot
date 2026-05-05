"""
FastAPI dependency functions for authentication.

Usage:
    @app.post("/protected")
    async def protected_endpoint(user_id: str = Depends(require_auth)):
        return {"user_id": user_id}

Both `require_auth` and `optional_auth` extract the raw key from the
"Authorization: Bearer <key>" header, validate it via SHA-256 lookup,
and check that the owning user account is still active.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException

from src.auth.api_keys import validate_api_key
from src.auth.audit import log_auth_event
from src.auth.storage import get_auth_storage
from src.observability.logging_config import get_logger

log = get_logger(__name__)


async def get_api_key(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Extract the raw API key from the Authorization header.

    Expects the format: `Authorization: Bearer k8s_rag_...`
    Returns None (rather than raising) so callers can decide whether auth is
    required.
    """
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


async def require_auth(api_key: Optional[str] = Depends(get_api_key)) -> str:
    """
    Enforce authentication. Raises HTTP 401/403 on failure.

    Returns:
        user_id string for the authenticated caller.
    """
    if not api_key:
        log_auth_event("authentication", user_id=None, result="failure", reason="no_header")
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Include 'Authorization: Bearer <api_key>' header.",
        )

    is_valid, user_id = validate_api_key(api_key)

    if not is_valid or not user_id:
        log_auth_event("authentication", user_id=None, result="failure", reason="invalid_key")
        raise HTTPException(status_code=401, detail="Invalid API key.")

    storage = get_auth_storage()
    user = storage.get_user(user_id)

    if not user or not user.is_active:
        log_auth_event("authentication", user_id=user_id, result="denied", reason="account_inactive")
        raise HTTPException(status_code=403, detail="User account is inactive.")

    log_auth_event("authentication", user_id=user_id, result="success")
    return user_id


async def optional_auth(api_key: Optional[str] = Depends(get_api_key)) -> Optional[str]:
    """
    Attempt authentication but do not raise if no key is provided.

    Mirrors the same validation logic as require_auth (key validity + active
    account check) but returns None instead of raising on failure, so callers
    can decide whether to enforce authentication themselves.

    Returns:
        user_id if authenticated and account is active, None otherwise.
    """
    if not api_key:
        return None

    is_valid, user_id = validate_api_key(api_key)
    if not is_valid or not user_id:
        log_auth_event("authentication", user_id=None, result="failure", reason="invalid_key")
        return None

    storage = get_auth_storage()
    user = storage.get_user(user_id)
    if not user or not user.is_active:
        log_auth_event("authentication", user_id=user_id, result="denied", reason="account_inactive")
        return None

    return user_id

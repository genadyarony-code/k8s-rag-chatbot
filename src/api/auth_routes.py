"""
Authentication and user management routes.

Endpoints:
    POST   /auth/users          — create a user account (open; admin-only in production)
    POST   /auth/keys           — create an API key for the authenticated user
    GET    /auth/keys           — list all keys for the authenticated user
    DELETE /auth/keys/{key_id}  — revoke a key (owner only)
    GET    /auth/me             — return the authenticated user's profile

Bootstrap note:
    The first user has no key yet (chicken-and-egg). Use the /auth/users endpoint
    to create an account, then create a key directly via the Python API in a
    management script, or temporarily set ENABLE_AUTH=false and hit POST /auth/keys.
"""

import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.auth.api_keys import create_api_key
from src.auth.audit import log_auth_event
from src.auth.dependencies import require_auth
from src.auth.models import User
from src.auth.storage import get_auth_storage
from src.observability.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


# ── Request / Response schemas ────────────────────────────────────────────────


class CreateUserRequest(BaseModel):
    email: str
    name: str


class CreateUserResponse(BaseModel):
    user_id: str
    email: str
    name: str
    message: str


class CreateKeyRequest(BaseModel):
    name: str


class CreateKeyResponse(BaseModel):
    key_id: str
    api_key: str
    name: str
    message: str


class KeyInfo(BaseModel):
    key_id: str
    name: str
    created_at: str
    last_used_at: Optional[str]
    is_active: bool


class UserInfo(BaseModel):
    user_id: str
    email: str
    name: str
    role: str
    created_at: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/users", response_model=CreateUserResponse)
async def create_user(request: CreateUserRequest):
    """
    Create a new user account.

    In a production system, restrict this endpoint to admin role or replace
    it with an OAuth/OIDC callback. Left open here for demo purposes.
    """
    storage = get_auth_storage()

    if storage.get_user_by_email(request.email):
        raise HTTPException(status_code=400, detail="Email already registered.")

    user = User(
        user_id=f"user_{secrets.token_hex(16)}",
        email=request.email,
        name=request.name,
        created_at=datetime.now(),
    )
    storage.create_user(user)

    log.info("user_created", user_id=user.user_id, email=user.email)
    log_auth_event("user_create", user_id=user.user_id, result="success")

    return CreateUserResponse(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        message="User created. Use POST /auth/keys to generate an API key.",
    )


@router.post("/keys", response_model=CreateKeyResponse)
async def create_key(
    request: CreateKeyRequest,
    user_id: str = Depends(require_auth),
):
    """
    Create a new API key for the authenticated user.

    The full key is returned exactly once. Store it securely — it cannot
    be retrieved again.
    """
    full_key, api_key = create_api_key(user_id, request.name)

    log_auth_event("key_create", user_id=user_id, resource=api_key.key_id, result="success")

    return CreateKeyResponse(
        key_id=api_key.key_id,
        api_key=full_key,
        name=request.name,
        message="Save this key securely. It will not be shown again.",
    )


@router.get("/keys", response_model=list[KeyInfo])
async def list_keys(user_id: str = Depends(require_auth)):
    """List all API keys belonging to the authenticated user."""
    storage = get_auth_storage()
    keys = storage.list_user_keys(user_id)

    return [
        KeyInfo(
            key_id=k.key_id,
            name=k.name,
            created_at=k.created_at.isoformat(),
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            is_active=k.is_active,
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, user_id: str = Depends(require_auth)):
    """Revoke an API key. Only the key's owner may revoke it."""
    storage = get_auth_storage()
    key = storage.get_api_key_by_id(key_id)

    if not key:
        raise HTTPException(status_code=404, detail="Key not found.")

    if key.user_id != user_id:
        log_auth_event("key_revoke", user_id=user_id, resource=key_id, result="denied")
        raise HTTPException(status_code=403, detail="Not authorised to revoke this key.")

    storage.revoke_key(key_id)

    log.info("api_key_revoked", user_id=user_id, key_id=key_id)
    log_auth_event("key_revoke", user_id=user_id, resource=key_id, result="success")

    return {"message": "Key revoked successfully.", "key_id": key_id}


@router.get("/me", response_model=UserInfo)
async def get_current_user(user_id: str = Depends(require_auth)):
    """Return the authenticated user's profile."""
    storage = get_auth_storage()
    user = storage.get_user(user_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    return UserInfo(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        role=user.role,
        created_at=user.created_at.isoformat(),
    )

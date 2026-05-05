"""
API key generation and validation.

Key format:  k8s_rag_<32 random alphanumeric chars>
Example:     k8s_rag_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6

Security model:
- Keys are generated with `secrets.choice` (cryptographically secure RNG).
- Only the SHA-256 hex digest is stored — the raw key is shown ONCE on creation.
- Lookup is O(1): compute sha256(submitted_key) and check the hash dict.

Why SHA-256 instead of bcrypt?
API keys have 32 chars × ~5.95 bits/char ≈ 190 bits of entropy — far beyond
brute-force reach. bcrypt's intentional slowness is designed for low-entropy
passwords, not high-entropy tokens. SHA-256 is the industry standard for
token storage (see GitHub, Stripe API key design).
"""

import hashlib
import hmac
import secrets
import string
from datetime import datetime
from typing import Optional

from src.auth.models import APIKey
from src.auth.storage import get_auth_storage
from src.observability.logging_config import get_logger

log = get_logger(__name__)

KEY_PREFIX = "k8s_rag_"
KEY_LENGTH = 32
KEY_ALPHABET = string.ascii_letters + string.digits


def generate_api_key() -> str:
    """Return a new raw API key. Show this to the user exactly once."""
    random_part = "".join(secrets.choice(KEY_ALPHABET) for _ in range(KEY_LENGTH))
    return f"{KEY_PREFIX}{random_part}"


def hash_api_key(key: str) -> str:
    """Return the SHA-256 hex digest of `key`."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(key: str, stored_hash: str) -> bool:
    """
    Constant-time comparison to guard against timing attacks.
    Computes sha256(key) and compares with `stored_hash`.
    """
    candidate_hash = hash_api_key(key)
    return hmac.compare_digest(candidate_hash, stored_hash)


def create_api_key(user_id: str, name: str) -> tuple[str, APIKey]:
    """
    Generate and store a new API key for the given user.

    Returns:
        (full_key, api_key_record) — full_key must be shown to the user once
        and never stored.
    """
    storage = get_auth_storage()

    full_key = generate_api_key()
    key_hash = hash_api_key(full_key)

    api_key = APIKey(
        key_id=f"key_{secrets.token_hex(8)}",
        key_hash=key_hash,
        user_id=user_id,
        name=name,
        created_at=datetime.now(),
    )

    storage.create_api_key(api_key)

    log.info("api_key_created", user_id=user_id, key_id=api_key.key_id, name=name)

    return full_key, api_key


def validate_api_key(key: str) -> tuple[bool, Optional[str]]:
    """
    Validate a raw API key submitted in the Authorization header.

    Computes SHA-256 of the submitted key and does a single O(1) dict lookup.

    Returns:
        (is_valid, user_id) — user_id is None when is_valid is False.
    """
    storage = get_auth_storage()

    key_hash = hash_api_key(key)
    api_key = storage.get_api_key_by_hash(key_hash)

    if api_key is None or not api_key.is_active:
        log.warning(
            "api_key_validation_failed",
            key_prefix=key[:len(KEY_PREFIX) + 4] if key else None,
        )
        return False, None

    storage.update_key_last_used(key_hash)

    log.info("api_key_validated", user_id=api_key.user_id, key_id=api_key.key_id)
    return True, api_key.user_id

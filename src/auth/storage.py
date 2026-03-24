"""
In-memory storage for users and API keys.

Data is lost on process restart. For production, replace this class with a
database-backed implementation (e.g., SQLAlchemy + PostgreSQL) that exposes
the same interface.
"""

from datetime import datetime
from typing import Optional

from src.auth.models import APIKey, User


class InMemoryAuthStorage:
    """
    Thread-safety note: Python's GIL provides basic safety for dict operations
    in a single-process server. For multi-worker deployments, replace with a
    database-backed or Redis-backed implementation.
    """

    def __init__(self) -> None:
        self._users: dict[str, User] = {}            # user_id → User
        self._users_by_email: dict[str, User] = {}   # email → User
        self._api_keys: dict[str, APIKey] = {}        # key_hash → APIKey
        self._keys_by_id: dict[str, APIKey] = {}      # key_id → APIKey

    # ── Users ─────────────────────────────────────────────────────────────────

    def create_user(self, user: User) -> User:
        if user.user_id in self._users:
            raise ValueError(f"User {user.user_id} already exists")
        if user.email in self._users_by_email:
            raise ValueError(f"Email {user.email} already registered")
        self._users[user.user_id] = user
        self._users_by_email[user.email] = user
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        return self._users_by_email.get(email)

    def list_users(self) -> list[User]:
        return list(self._users.values())

    # ── API Keys ──────────────────────────────────────────────────────────────

    def create_api_key(self, api_key: APIKey) -> APIKey:
        if api_key.key_hash in self._api_keys:
            raise ValueError("Key hash collision — regenerate key")
        self._api_keys[api_key.key_hash] = api_key
        self._keys_by_id[api_key.key_id] = api_key
        return api_key

    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        return self._api_keys.get(key_hash)

    def get_api_key_by_id(self, key_id: str) -> Optional[APIKey]:
        return self._keys_by_id.get(key_id)

    def list_user_keys(self, user_id: str) -> list[APIKey]:
        return [k for k in self._api_keys.values() if k.user_id == user_id]

    def update_key_last_used(self, key_hash: str) -> None:
        if key_hash in self._api_keys:
            self._api_keys[key_hash].last_used_at = datetime.now()

    def revoke_key(self, key_id: str) -> bool:
        key = self._keys_by_id.get(key_id)
        if key:
            key.is_active = False
            return True
        return False


# ── Lazy singleton ────────────────────────────────────────────────────────────

_storage: Optional[InMemoryAuthStorage] = None


def get_auth_storage() -> InMemoryAuthStorage:
    """Get or create the global auth storage."""
    global _storage
    if _storage is None:
        _storage = InMemoryAuthStorage()
    return _storage

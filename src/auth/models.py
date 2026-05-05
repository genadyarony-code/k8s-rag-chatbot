"""
User and APIKey dataclasses.

In-memory for this phase. Replace with an ORM model + PostgreSQL in production.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """
    User account.

    role is "user" by default; set to "admin" to gate future admin-only endpoints.
    """

    user_id: str
    email: str
    name: str
    created_at: datetime
    is_active: bool = True
    role: str = "user"  # user | admin

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "role": self.role,
        }


@dataclass
class APIKey:
    """
    API key record.

    key_hash is the SHA-256 hex digest of the raw key.
    The raw key is shown to the user exactly once and never persisted.
    """

    key_id: str
    key_hash: str      # SHA-256(raw_key) — never expose
    user_id: str
    name: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True

    def to_dict(self) -> dict:
        """Serialize without exposing the hash."""
        return {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
        }

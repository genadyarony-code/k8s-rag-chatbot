"""
Approval queue for pending HITL actions.

Supports two storage backends:
- Redis (production)  — persistent, shared across worker processes
- In-memory (default) — zero-config for development and testing

Queue contract:
  submit()       → request_id     (status = "pending")
  approve(id)    → bool           (status = "approved")
  reject(id)     → bool           (status = "rejected")
  get(id)        → dict | None
  list_pending() → list[dict]

Redis keys:
  approval:<request_id>   — JSON-serialised request dict  (TTL 24 h)
  approval:pending        — SADD/SREM set of pending IDs

When Redis is unavailable the queue falls back silently to in-memory storage
so development and unit tests work without a running Redis instance.
"""

import json
import secrets
from datetime import datetime
from typing import Optional

from src.observability.logging_config import get_logger

log = get_logger(__name__)

_TTL_SECONDS = 86_400  # 24 hours


class ApprovalQueue:
    """
    HITL approval queue with Redis or in-memory storage.

    Instantiate via the `get_approval_queue()` lazy singleton for production
    use, or construct directly with `ApprovalQueue()` for isolated tests.
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        if redis_url:
            try:
                import redis as _redis
                self._redis = _redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                self._store = "redis"
                log.info("approval_queue_initialized", storage="redis", url=redis_url)
            except Exception as exc:
                log.warning(
                    "redis_unavailable_falling_back_to_memory",
                    error=str(exc),
                )
                self._redis = None
                self._store = "memory"
                self._memory: dict[str, dict] = {}
        else:
            self._redis = None
            self._store = "memory"
            self._memory: dict[str, dict] = {}
            log.info("approval_queue_initialized", storage="memory")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _save(self, request: dict) -> None:
        rid = request["request_id"]
        if self._store == "redis":
            self._redis.set(f"approval:{rid}", json.dumps(request), ex=_TTL_SECONDS)
        else:
            self._memory[rid] = request

    def _load(self, request_id: str) -> Optional[dict]:
        if self._store == "redis":
            raw = self._redis.get(f"approval:{request_id}")
            return json.loads(raw) if raw else None
        return self._memory.get(request_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(
        self,
        action_type: str,
        action_data: dict,
        confidence: float,
        session_id: str,
        user_id: str,
    ) -> str:
        """
        Submit an action for human approval.

        Returns:
            Unique approval request ID (e.g. "approval_<32 hex chars>").
        """
        request_id = f"approval_{secrets.token_hex(16)}"

        request = {
            "request_id": request_id,
            "action_type": action_type,
            "action_data": action_data,
            "confidence": confidence,
            "session_id": session_id,
            "user_id": user_id,
            "status": "pending",
            "submitted_at": datetime.now().isoformat(),
            "approved_at": None,
            "approved_by": None,
            "rejection_reason": None,
        }

        self._save(request)

        if self._store == "redis":
            self._redis.sadd("approval:pending", request_id)

        log.info(
            "approval_submitted",
            request_id=request_id,
            action_type=action_type,
            confidence=confidence,
            user_id=user_id,
        )

        return request_id

    def get(self, request_id: str) -> Optional[dict]:
        """Return the approval request dict, or None if not found."""
        return self._load(request_id)

    def approve(self, request_id: str, approver_id: str) -> bool:
        """
        Mark a pending request as approved.

        Returns:
            True on success, False if request not found or already processed.
        """
        request = self._load(request_id)
        if not request:
            log.warning("approval_not_found", request_id=request_id)
            return False
        if request["status"] != "pending":
            log.warning(
                "approval_already_processed",
                request_id=request_id,
                status=request["status"],
            )
            return False

        request["status"] = "approved"
        request["approved_at"] = datetime.now().isoformat()
        request["approved_by"] = approver_id
        self._save(request)

        if self._store == "redis":
            self._redis.srem("approval:pending", request_id)

        log.info(
            "approval_granted",
            request_id=request_id,
            approver_id=approver_id,
            action_type=request["action_type"],
        )

        return True

    def reject(self, request_id: str, rejector_id: str, reason: str) -> bool:
        """
        Mark a pending request as rejected.

        Returns:
            True on success, False if request not found or already processed.
        """
        request = self._load(request_id)
        if not request:
            return False
        if request["status"] != "pending":
            return False

        request["status"] = "rejected"
        request["approved_at"] = datetime.now().isoformat()
        request["approved_by"] = rejector_id
        request["rejection_reason"] = reason
        self._save(request)

        if self._store == "redis":
            self._redis.srem("approval:pending", request_id)

        log.info(
            "approval_rejected",
            request_id=request_id,
            rejector_id=rejector_id,
            reason=reason,
        )

        return True

    def list_pending(self, user_id: Optional[str] = None) -> list[dict]:
        """
        Return all pending requests, optionally filtered by submitting user.
        """
        if self._store == "redis":
            ids = self._redis.smembers("approval:pending")
            requests = []
            for rid in ids:
                req = self._load(rid)
                if req and (not user_id or req["user_id"] == user_id):
                    requests.append(req)
            return requests
        else:
            return [
                r for r in self._memory.values()
                if r["status"] == "pending"
                and (not user_id or r["user_id"] == user_id)
            ]


# ── Lazy singleton ────────────────────────────────────────────────────────────

_queue: Optional[ApprovalQueue] = None


def get_approval_queue() -> ApprovalQueue:
    """Get or create the global approval queue instance."""
    global _queue
    if _queue is None:
        from src.config.settings import settings
        _queue = ApprovalQueue(redis_url=settings.redis_url or None)
    return _queue

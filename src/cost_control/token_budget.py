"""
Token budget enforcement per session and globally.

Budgets:
- Per-session daily limit (default: 100k tokens)
- Global daily limit (default: 1M tokens)
- Per-request limit (default: 10k tokens)

Storage: in-memory (single instance). For multi-instance deployments,
swap in a Redis-backed implementation — the interface is identical.

Why token budgets?
- Prevent a single session from draining API credits
- Cap total daily spend before it becomes a surprise bill
- Surface abnormal usage patterns early via the warning log
"""

from datetime import datetime
from typing import Optional

from src.config.settings import settings
from src.observability.logging_config import get_logger

log = get_logger(__name__)


class TokenBudget:
    """
    In-memory token budget tracker.

    Counters reset at midnight (date-based, not sliding window).
    Thread safety is not guaranteed; for true concurrency use Redis with
    INCR + EXPIRE.
    """

    def __init__(
        self,
        session_daily_limit: int = 100_000,
        global_daily_limit: int = 1_000_000,
        per_request_limit: int = 10_000,
    ) -> None:
        self.session_daily_limit = session_daily_limit
        self.global_daily_limit = global_daily_limit
        self.per_request_limit = per_request_limit

        self._session_usage: dict[str, int] = {}
        self._global_usage: int = 0
        self._last_reset: datetime = datetime.now()

    def _maybe_reset_daily(self) -> None:
        now = datetime.now()
        if now.date() > self._last_reset.date():
            log.info(
                "token_budget_daily_reset",
                previous_global_usage=self._global_usage,
                session_count=len(self._session_usage),
            )
            self._session_usage.clear()
            self._global_usage = 0
            self._last_reset = now

    def check_and_reserve(
        self,
        session_id: str,
        estimated_tokens: int,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if the request is within all budgets and optimistically reserve tokens.

        Args:
            session_id: Session identifier
            estimated_tokens: Upper-bound token estimate for this request

        Returns:
            (allowed, reason) — reason is None when allowed is True
        """
        self._maybe_reset_daily()

        if estimated_tokens > self.per_request_limit:
            reason = f"Request exceeds per-request limit ({self.per_request_limit:,} tokens)"
            log.warning(
                "token_budget_exceeded",
                session_id=session_id,
                limit_type="per_request",
                requested=estimated_tokens,
                limit=self.per_request_limit,
            )
            return False, reason

        session_used = self._session_usage.get(session_id, 0)
        if session_used + estimated_tokens > self.session_daily_limit:
            reason = f"Session daily limit exceeded ({self.session_daily_limit:,} tokens)"
            log.warning(
                "token_budget_exceeded",
                session_id=session_id,
                limit_type="session_daily",
                used=session_used,
                requested=estimated_tokens,
                limit=self.session_daily_limit,
            )
            return False, reason

        if self._global_usage + estimated_tokens > self.global_daily_limit:
            reason = f"Global daily limit exceeded ({self.global_daily_limit:,} tokens)"
            log.error(
                "token_budget_exceeded",
                session_id=session_id,
                limit_type="global_daily",
                used=self._global_usage,
                requested=estimated_tokens,
                limit=self.global_daily_limit,
            )
            return False, reason

        self._session_usage[session_id] = session_used + estimated_tokens
        self._global_usage += estimated_tokens

        log.info(
            "token_budget_reserved",
            session_id=session_id,
            tokens=estimated_tokens,
            session_total=self._session_usage[session_id],
            global_total=self._global_usage,
        )

        return True, None

    def get_usage(self, session_id: str) -> dict:
        """Return current usage stats for a session."""
        self._maybe_reset_daily()
        session_used = self._session_usage.get(session_id, 0)
        return {
            "session_used": session_used,
            "session_limit": self.session_daily_limit,
            "session_remaining": self.session_daily_limit - session_used,
            "global_used": self._global_usage,
            "global_limit": self.global_daily_limit,
            "global_remaining": self.global_daily_limit - self._global_usage,
        }


# ── Lazy singleton ────────────────────────────────────────────────────────────

_budget: Optional[TokenBudget] = None


def get_token_budget() -> TokenBudget:
    """Get or create the global token budget tracker."""
    global _budget
    if _budget is None:
        _budget = TokenBudget(
            session_daily_limit=settings.token_budget_session_daily,
            global_daily_limit=settings.token_budget_global_daily,
            per_request_limit=settings.token_budget_per_request,
        )
    return _budget

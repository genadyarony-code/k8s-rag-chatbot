"""
Cost tracking and alerting for OpenAI API usage.

Tracks:
- Cumulative daily cost
- Cost per session
- Cost per model

Alert thresholds:
- Warning (default: $10/day)  — logs a warning, no action
- Hard limit (default: $50/day) — enables degraded mode, blocks OpenAI calls

Pricing constants reflect March 2026 OpenAI pricing.
Update PRICING if models or rates change.
"""

from datetime import datetime
from typing import Optional

from src.observability.logging_config import get_logger
from src.observability.metrics import chat_cost_usd_total

log = get_logger(__name__)


class CostTracker:
    """
    Daily cost tracker with warning and hard-limit thresholds.

    Counters reset at midnight (date-based). Not thread-safe; for multi-worker
    deployments replace with a Redis-backed implementation.
    """

    # USD per 1M tokens
    PRICING: dict[str, dict[str, float]] = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"embedding": 0.02},
    }

    def __init__(
        self,
        warning_threshold_usd: float = 10.0,
        hard_limit_usd: float = 50.0,
    ) -> None:
        self.warning_threshold = warning_threshold_usd
        self.hard_limit = hard_limit_usd

        self._daily_cost: float = 0.0
        self._session_costs: dict[str, float] = {}
        self._last_reset: datetime = datetime.now()

        self._warning_sent = False
        self._limit_exceeded = False

    def _maybe_reset_daily(self) -> None:
        now = datetime.now()
        if now.date() > self._last_reset.date():
            log.info(
                "cost_tracker_daily_reset",
                previous_cost=self._daily_cost,
                session_count=len(self._session_costs),
            )
            self._daily_cost = 0.0
            self._session_costs.clear()
            self._warning_sent = False
            self._limit_exceeded = False
            self._last_reset = now

    def calculate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        embedding_tokens: int = 0,
    ) -> float:
        """Return cost in USD for the given token counts."""
        if model not in self.PRICING:
            log.warning("unknown_model_pricing", model=model)
            return 0.0

        pricing = self.PRICING[model]
        cost = 0.0
        if "input" in pricing:
            cost += (input_tokens / 1_000_000) * pricing["input"]
        if "output" in pricing:
            cost += (output_tokens / 1_000_000) * pricing["output"]
        if "embedding" in pricing:
            cost += (embedding_tokens / 1_000_000) * pricing["embedding"]
        return cost

    def track_request(
        self,
        session_id: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        embedding_tokens: int = 0,
    ) -> tuple[float, bool]:
        """
        Record cost for a completed request and check alert thresholds.

        Returns:
            (cost_usd, allowed)
            allowed=False if the hard limit has been exceeded after this request.
        """
        self._maybe_reset_daily()

        cost = self.calculate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            embedding_tokens=embedding_tokens,
        )

        self._daily_cost += cost
        self._session_costs[session_id] = self._session_costs.get(session_id, 0) + cost

        # Update the Prometheus counter that was defined in Phase 1
        chat_cost_usd_total.inc(cost)

        log.info(
            "cost_tracked",
            session_id=session_id,
            model=model,
            cost_usd=round(cost, 8),
            daily_total=round(self._daily_cost, 6),
        )

        if self._daily_cost >= self.hard_limit and not self._limit_exceeded:
            self._limit_exceeded = True
            log.critical(
                "cost_hard_limit_exceeded",
                daily_cost=self._daily_cost,
                limit=self.hard_limit,
                action="DEGRADED MODE ENABLED - OpenAI calls blocked",
            )
            return cost, False

        if self._daily_cost >= self.warning_threshold and not self._warning_sent:
            self._warning_sent = True
            log.warning(
                "cost_warning_threshold_exceeded",
                daily_cost=self._daily_cost,
                threshold=self.warning_threshold,
                limit=self.hard_limit,
            )

        return cost, True

    def is_budget_available(self) -> bool:
        """Return False if the hard daily cost limit has been exceeded."""
        self._maybe_reset_daily()
        return not self._limit_exceeded

    def get_stats(self) -> dict:
        """Return a snapshot of today's cost statistics."""
        self._maybe_reset_daily()
        return {
            "daily_cost_usd": round(self._daily_cost, 6),
            "warning_threshold_usd": self.warning_threshold,
            "hard_limit_usd": self.hard_limit,
            "budget_remaining_usd": round(max(0.0, self.hard_limit - self._daily_cost), 6),
            "budget_exceeded": self._limit_exceeded,
            "session_count": len(self._session_costs),
        }


# ── Lazy singleton ────────────────────────────────────────────────────────────

_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _tracker
    if _tracker is None:
        from src.config.settings import settings

        _tracker = CostTracker(
            warning_threshold_usd=settings.cost_warning_threshold_usd,
            hard_limit_usd=settings.cost_hard_limit_usd,
        )
    return _tracker

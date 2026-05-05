"""
Circuit breaker for OpenAI API calls.

Pattern:
- CLOSED:    Normal operation — requests go through.
- OPEN:      Too many failures — requests blocked (fail fast).
- HALF_OPEN: Testing if service recovered — limited requests allowed.

Configuration (default):
- Failure threshold: 5 failures triggers OPEN
- Recovery timeout:  60 seconds before entering HALF_OPEN
- Excluded errors:   ValueError (input validation) does not count as a failure

Why a circuit breaker?
- Prevents hammering a failing upstream service
- Fails fast rather than waiting for network timeouts to pile up
- Automatically tests recovery so traffic resumes without manual intervention
"""

from functools import wraps
from typing import Callable

from pybreaker import CircuitBreaker, CircuitBreakerError, CircuitBreakerListener

from src.observability.logging_config import get_logger
from src.observability.metrics import circuit_breaker_state

log = get_logger(__name__)

# Re-export so callers can catch CircuitBreakerError from this module
__all__ = ["openai_breaker", "with_circuit_breaker", "CircuitBreakerError"]


class _OpenAIBreakerListener(CircuitBreakerListener):
    """Structured-log state transitions for the OpenAI circuit breaker."""

    def state_change(self, cb, old_state, new_state) -> None:  # type: ignore[override]
        state_name = str(new_state).lower()
        if "open" in state_name and "half" not in state_name:
            log.error(
                "circuit_breaker_opened",
                breaker_name=cb.name,
                fail_count=cb.fail_counter,
                message="OpenAI API circuit breaker OPENED - requests will be blocked",
            )
            circuit_breaker_state.labels(breaker=cb.name).set(2)
        elif "half" in state_name:
            log.warning(
                "circuit_breaker_half_open",
                breaker_name=cb.name,
                message="OpenAI API circuit breaker HALF-OPEN - testing recovery",
            )
            circuit_breaker_state.labels(breaker=cb.name).set(1)
        elif "closed" in state_name:
            log.info(
                "circuit_breaker_closed",
                breaker_name=cb.name,
                message="OpenAI API circuit breaker CLOSED - service recovered",
            )
            circuit_breaker_state.labels(breaker=cb.name).set(0)


openai_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=60,
    exclude=[ValueError],
    name="openai_api",
    listeners=[_OpenAIBreakerListener()],
)


def with_circuit_breaker(func: Callable) -> Callable:
    """
    Decorator that wraps a function with the OpenAI circuit breaker.

    When the breaker is OPEN, CircuitBreakerError is raised immediately
    without executing the underlying function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return openai_breaker.call(func, *args, **kwargs)
        except CircuitBreakerError:
            log.error(
                "circuit_breaker_blocked_request",
                function=func.__name__,
                breaker_state=str(openai_breaker.current_state),
            )
            raise

    return wrapper

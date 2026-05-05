"""
Unit tests for cost controls: token budgets, circuit breaker, cost tracking.

These tests instantiate fresh objects so they are fully isolated and do not
depend on the global singleton state used by the running application.
"""

import pytest
from pybreaker import CircuitBreaker, CircuitBreakerError

from src.cost_control.cost_tracker import CostTracker
from src.cost_control.token_budget import TokenBudget


# ── Token Budget Tests ────────────────────────────────────────────────────────


def test_token_budget_allows_within_limit():
    budget = TokenBudget(session_daily_limit=10_000)
    allowed, reason = budget.check_and_reserve("session1", 1_000)
    assert allowed
    assert reason is None


def test_token_budget_tracks_cumulative_usage():
    budget = TokenBudget(session_daily_limit=10_000)
    budget.check_and_reserve("session1", 4_000)
    budget.check_and_reserve("session1", 4_000)
    # Third request — only 2k remaining, requesting 3k
    allowed, reason = budget.check_and_reserve("session1", 3_000)
    assert not allowed
    assert reason is not None
    assert "exceeded" in reason.lower()


def test_token_budget_blocks_over_session_limit():
    budget = TokenBudget(session_daily_limit=10_000)
    budget.check_and_reserve("session1", 9_000)
    allowed, reason = budget.check_and_reserve("session1", 2_000)
    assert not allowed
    assert "session" in reason.lower()


def test_token_budget_per_request_limit():
    budget = TokenBudget(per_request_limit=5_000)
    allowed, reason = budget.check_and_reserve("session1", 10_000)
    assert not allowed
    assert "per-request" in reason.lower()


def test_token_budget_global_limit():
    budget = TokenBudget(global_daily_limit=20_000, session_daily_limit=100_000)
    budget.check_and_reserve("session1", 8_000)
    budget.check_and_reserve("session2", 8_000)
    allowed, reason = budget.check_and_reserve("session3", 5_000)
    assert not allowed
    assert "global" in reason.lower()


def test_token_budget_get_usage_returns_correct_fields():
    budget = TokenBudget(session_daily_limit=10_000, global_daily_limit=50_000)
    budget.check_and_reserve("s1", 3_000)
    usage = budget.get_usage("s1")
    assert usage["session_used"] == 3_000
    assert usage["session_remaining"] == 7_000
    assert usage["global_used"] == 3_000
    assert usage["global_limit"] == 50_000


def test_token_budget_different_sessions_are_isolated():
    budget = TokenBudget(session_daily_limit=5_000, global_daily_limit=100_000)
    budget.check_and_reserve("alice", 4_500)
    # Bob's budget is independent of Alice's
    allowed, _ = budget.check_and_reserve("bob", 4_500)
    assert allowed


# ── Cost Tracker Tests ────────────────────────────────────────────────────────


def test_cost_calculation_gpt4o_mini():
    tracker = CostTracker()
    cost = tracker.calculate_cost(
        model="gpt-4o-mini",
        input_tokens=1_000,
        output_tokens=500,
    )
    expected = (1_000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
    assert abs(cost - expected) < 1e-9


def test_cost_calculation_embedding():
    tracker = CostTracker()
    cost = tracker.calculate_cost(
        model="text-embedding-3-small",
        embedding_tokens=1_000_000,
    )
    assert abs(cost - 0.02) < 1e-9


def test_cost_calculation_unknown_model_returns_zero():
    tracker = CostTracker()
    cost = tracker.calculate_cost(model="gpt-99-ultra", input_tokens=1_000)
    assert cost == 0.0


def test_cost_warning_threshold_sets_flag():
    tracker = CostTracker(warning_threshold_usd=0.001, hard_limit_usd=1.0)
    tracker.track_request("test", "gpt-4o-mini", input_tokens=10_000, output_tokens=5_000)
    assert tracker._warning_sent


def test_cost_warning_does_not_block_requests():
    tracker = CostTracker(warning_threshold_usd=0.001, hard_limit_usd=1.0)
    _, allowed = tracker.track_request("test", "gpt-4o-mini", input_tokens=10_000, output_tokens=5_000)
    assert allowed


def test_cost_hard_limit_sets_exceeded_flag():
    tracker = CostTracker(warning_threshold_usd=0.0001, hard_limit_usd=0.0005)
    tracker.track_request("test", "gpt-4o-mini", input_tokens=100_000, output_tokens=50_000)
    assert tracker._limit_exceeded


def test_cost_is_budget_available_returns_false_after_limit():
    tracker = CostTracker(warning_threshold_usd=0.0001, hard_limit_usd=0.0005)
    tracker.track_request("test", "gpt-4o-mini", input_tokens=100_000, output_tokens=50_000)
    assert not tracker.is_budget_available()


def test_cost_get_stats_shape():
    tracker = CostTracker(warning_threshold_usd=5.0, hard_limit_usd=20.0)
    stats = tracker.get_stats()
    assert "daily_cost_usd" in stats
    assert "hard_limit_usd" in stats
    assert "budget_remaining_usd" in stats
    assert "budget_exceeded" in stats
    assert stats["budget_exceeded"] is False


# ── Circuit Breaker Tests ─────────────────────────────────────────────────────


def test_circuit_breaker_passes_successful_calls():
    breaker = CircuitBreaker(fail_max=3, reset_timeout=60)

    @breaker
    def ok_function():
        return "success"

    assert ok_function() == "success"


def test_circuit_breaker_opens_after_failures():
    """Circuit should open after fail_max consecutive failures."""
    breaker = CircuitBreaker(fail_max=3, reset_timeout=60, exclude=[])

    @breaker
    def failing_function():
        raise IOError("simulated failure")

    for _ in range(3):
        try:
            failing_function()
        except IOError:
            pass

    with pytest.raises(CircuitBreakerError):
        failing_function()


def test_circuit_breaker_excludes_value_errors():
    """ValueError should not count toward the failure threshold."""
    breaker = CircuitBreaker(fail_max=2, reset_timeout=60, exclude=[ValueError])

    @breaker
    def validation_error():
        raise ValueError("bad input")

    for _ in range(5):
        try:
            validation_error()
        except ValueError:
            pass

    # Circuit should still be closed — ValueError is excluded
    assert str(breaker.current_state).lower() != "open"

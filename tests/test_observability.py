"""
Tests for structured logging and Prometheus metrics.
"""

import pytest

from src.observability.logging_config import configure_logging, get_logger
from src.observability.metrics import (
    chat_latency_seconds,
    chat_requests_total,
    get_metrics,
)


def test_structured_logging_does_not_raise():
    """Test that structured logger emits events without errors."""
    configure_logging("INFO")
    log = get_logger("test")
    log.info("test_event", user_id="test123", count=42)


def test_structured_logging_accepts_kwargs():
    """Test that arbitrary keyword arguments are accepted as structured fields."""
    configure_logging("DEBUG")
    log = get_logger("test")
    log.debug("debug_event", session_id="abc", tokens=100, cost_usd=0.000015)
    log.warning("warn_event", reason="test_only")
    log.error("error_event", error="fake_error")


def test_metrics_counter_increments():
    """Test that Prometheus counters increment correctly."""
    before = chat_requests_total.labels(session_id="_unit_test")._value.get()
    chat_requests_total.labels(session_id="_unit_test").inc()
    after = chat_requests_total.labels(session_id="_unit_test")._value.get()
    assert after == before + 1


def test_metrics_histogram_records():
    """Test that Prometheus histograms accept observations."""
    chat_latency_seconds.observe(0.42)
    chat_latency_seconds.observe(1.5)


def test_metrics_export_contains_known_metrics():
    """Test that the /metrics exposition format includes expected metric names."""
    output = get_metrics()
    assert b"chat_requests_total" in output
    assert b"chat_latency_seconds" in output
    assert b"retrieval_latency_seconds" in output
    assert b"generation_latency_seconds" in output
    assert b"index_health" in output
    assert b"feature_flag_status" in output

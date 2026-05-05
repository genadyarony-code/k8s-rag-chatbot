"""
Unit tests for the HITL (Human-in-the-Loop) framework.

All tests are fast — no API calls, no external services:
- Confidence score calculation and level mapping
- Approval requirement policy
- Approval queue (in-memory backend)
- Authorization matrix: execute, approve, auto-approve
"""

import pytest

from src.hitl.confidence import calculate_confidence, get_confidence_level, requires_approval
from src.hitl.approval_queue import ApprovalQueue
from src.hitl.authorization import ActionType, Role, can_approve, can_execute, should_auto_approve


# ── Confidence Score Tests ────────────────────────────────────────────────────


def test_confidence_high_quality_retrieval():
    """High retrieval score + valid citations = high confidence."""
    score = calculate_confidence(
        retrieval_score=0.9,
        citation_valid=True,
        source_count=5,
    )
    assert score > 0.7
    assert get_confidence_level(score) in ("high", "medium")


def test_confidence_low_quality_retrieval():
    """Low retrieval score + failed citations = low confidence."""
    score = calculate_confidence(
        retrieval_score=0.2,
        citation_valid=False,
        source_count=0,
    )
    assert score < 0.5
    assert get_confidence_level(score) == "low"


def test_confidence_with_eval_score():
    """LLM judge score (1–5) is correctly normalised and applied."""
    with_perfect = calculate_confidence(
        retrieval_score=0.8,
        citation_valid=True,
        eval_score=5.0,
        source_count=3,
    )
    with_poor = calculate_confidence(
        retrieval_score=0.8,
        citation_valid=True,
        eval_score=1.0,
        source_count=3,
    )
    assert with_perfect > with_poor


def test_confidence_clamped_to_unit_interval():
    """Output must always be in [0, 1]."""
    score = calculate_confidence(
        retrieval_score=1.0,
        citation_valid=True,
        eval_score=5.0,
        source_count=10,
    )
    assert 0.0 <= score <= 1.0


def test_confidence_source_diversity_capped():
    """Source diversity tops out at 5 sources; more doesn't increase score."""
    s5 = calculate_confidence(0.7, True, source_count=5)
    s10 = calculate_confidence(0.7, True, source_count=10)
    assert s5 == s10


def test_confidence_level_thresholds():
    assert get_confidence_level(0.9) == "high"
    assert get_confidence_level(0.8) == "high"
    assert get_confidence_level(0.79) == "medium"
    assert get_confidence_level(0.5) == "medium"
    assert get_confidence_level(0.49) == "low"
    assert get_confidence_level(0.0) == "low"


# ── Approval Policy Tests ─────────────────────────────────────────────────────


def test_requires_approval_read_high_confidence():
    assert not requires_approval(0.8, "read")


def test_requires_approval_read_very_low():
    assert requires_approval(0.2, "read")


def test_requires_approval_write_medium():
    assert requires_approval(0.7, "write")


def test_requires_approval_write_very_high():
    assert not requires_approval(0.95, "write")


def test_requires_approval_delete_always():
    assert requires_approval(1.0, "delete")
    assert requires_approval(0.0, "delete")


def test_requires_approval_unknown_defaults_to_true():
    assert requires_approval(0.9, "unknown_action")


# ── Approval Queue Tests ──────────────────────────────────────────────────────


def test_queue_submit_and_get():
    queue = ApprovalQueue()  # in-memory backend
    rid = queue.submit(
        action_type="kubectl_apply",
        action_data={"manifest": "deployment.yaml"},
        confidence=0.65,
        session_id="sess_test",
        user_id="user_1",
    )
    assert rid.startswith("approval_")
    req = queue.get(rid)
    assert req is not None
    assert req["status"] == "pending"
    assert req["user_id"] == "user_1"
    assert req["action_type"] == "kubectl_apply"


def test_queue_approve():
    queue = ApprovalQueue()
    rid = queue.submit("kubectl_apply", {}, 0.7, "sess", "user_1")
    assert queue.approve(rid, "admin_1") is True
    req = queue.get(rid)
    assert req["status"] == "approved"
    assert req["approved_by"] == "admin_1"


def test_queue_reject_with_reason():
    queue = ApprovalQueue()
    rid = queue.submit("kubectl_delete", {}, 0.4, "sess", "user_1")
    assert queue.reject(rid, "admin_1", "Too risky at this confidence") is True
    req = queue.get(rid)
    assert req["status"] == "rejected"
    assert "risky" in req["rejection_reason"]


def test_queue_double_approve_fails():
    """Approving an already-processed request must return False."""
    queue = ApprovalQueue()
    rid = queue.submit("read", {}, 0.9, "sess", "user_1")
    queue.approve(rid, "approver_1")
    assert queue.approve(rid, "approver_2") is False


def test_queue_list_pending():
    queue = ApprovalQueue()
    rid1 = queue.submit("read", {}, 0.9, "sess", "u1")
    rid2 = queue.submit("write", {}, 0.6, "sess", "u2")
    pending_ids = {r["request_id"] for r in queue.list_pending()}
    assert rid1 in pending_ids
    assert rid2 in pending_ids


def test_queue_list_pending_filter_by_user():
    queue = ApprovalQueue()
    queue.submit("read", {}, 0.9, "sess", "user_alice")
    queue.submit("read", {}, 0.9, "sess", "user_bob")
    alice_pending = queue.list_pending(user_id="user_alice")
    assert all(r["user_id"] == "user_alice" for r in alice_pending)


def test_queue_get_unknown_returns_none():
    queue = ApprovalQueue()
    assert queue.get("approval_nonexistent_id") is None


# ── Authorization Matrix Tests ────────────────────────────────────────────────


def test_user_can_execute_read():
    assert can_execute(Role.USER, ActionType.READ)


def test_user_can_execute_write():
    assert can_execute(Role.USER, ActionType.WRITE)


def test_user_cannot_execute_delete():
    assert not can_execute(Role.USER, ActionType.DELETE)


def test_approver_cannot_execute_delete():
    assert not can_execute(Role.APPROVER, ActionType.DELETE)


def test_admin_can_execute_delete():
    assert can_execute(Role.ADMIN, ActionType.DELETE)


def test_approver_can_approve_write():
    assert can_approve(Role.APPROVER, ActionType.WRITE)


def test_approver_cannot_approve_delete():
    assert not can_approve(Role.APPROVER, ActionType.DELETE)


def test_admin_can_approve_delete():
    assert can_approve(Role.ADMIN, ActionType.DELETE)


def test_user_cannot_approve_anything():
    for action in ActionType:
        assert not can_approve(Role.USER, action)


def test_auto_approve_read_above_threshold():
    assert should_auto_approve(ActionType.READ, 0.4)


def test_auto_approve_read_below_threshold():
    assert not should_auto_approve(ActionType.READ, 0.2)


def test_auto_approve_write_requires_high_confidence():
    assert not should_auto_approve(ActionType.WRITE, 0.8)
    assert should_auto_approve(ActionType.WRITE, 0.95)


def test_auto_approve_delete_never():
    assert not should_auto_approve(ActionType.DELETE, 1.0)


def test_auto_approve_kubectl_delete_never():
    assert not should_auto_approve(ActionType.KUBECTL_DELETE, 1.0)

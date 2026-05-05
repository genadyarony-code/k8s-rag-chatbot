"""
Authorization matrix for HITL action approval.

Defines:
- Which roles can EXECUTE each action type (initiate the request).
- Which roles can APPROVE each action type (grant execution).
- At what confidence threshold an action is auto-approved (no human needed).

Role hierarchy (higher = more privileged):
    USER  (1) → APPROVER  (2) → ADMIN  (3)

Rationale for thresholds:
- READ / KUBECTL_GET: low-risk, auto-approve above 0.3 / 0.5.
- WRITE / KUBECTL_APPLY: high-risk, auto-approve only above 0.9 / 0.95.
- DELETE / KUBECTL_DELETE: irreversible — NEVER auto-approved.

This module is a policy skeleton for future tool-calling agents. No live
actions are taken by the chatbot today; confidence scoring and approval queue
are wired in so adding a first tool requires zero scaffolding changes.
"""

from enum import Enum
from typing import Optional

from src.observability.logging_config import get_logger

log = get_logger(__name__)


class ActionType(str, Enum):
    """Supported action types (current + planned)."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    KUBECTL_GET = "kubectl_get"
    KUBECTL_APPLY = "kubectl_apply"
    KUBECTL_DELETE = "kubectl_delete"
    API_CALL = "api_call"


class Role(str, Enum):
    """User roles, ordered from least to most privileged."""
    USER = "user"
    APPROVER = "approver"
    ADMIN = "admin"


_ROLE_LEVEL: dict[str, int] = {
    Role.USER: 1,
    Role.APPROVER: 2,
    Role.ADMIN: 3,
}

# Authorization matrix.
# auto_approve_threshold=None means the action is NEVER auto-approved.
_MATRIX: dict[str, dict] = {
    ActionType.READ: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.3,
        "approver_roles": [Role.APPROVER, Role.ADMIN],
    },
    ActionType.WRITE: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.9,
        "approver_roles": [Role.APPROVER, Role.ADMIN],
    },
    ActionType.DELETE: {
        "required_role": Role.ADMIN,
        "auto_approve_threshold": None,
        "approver_roles": [Role.ADMIN],
    },
    ActionType.KUBECTL_GET: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.5,
        "approver_roles": [Role.APPROVER, Role.ADMIN],
    },
    ActionType.KUBECTL_APPLY: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.95,
        "approver_roles": [Role.APPROVER, Role.ADMIN],
    },
    ActionType.KUBECTL_DELETE: {
        "required_role": Role.APPROVER,
        "auto_approve_threshold": None,
        "approver_roles": [Role.ADMIN],
    },
    ActionType.API_CALL: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.85,
        "approver_roles": [Role.APPROVER, Role.ADMIN],
    },
}


def can_execute(user_role: str, action_type: str) -> bool:
    """
    Return True if `user_role` has permission to initiate `action_type`.

    Note: passing the check here does NOT mean the action runs immediately —
    it may still require approval depending on confidence score.
    """
    config = _MATRIX.get(action_type, {})
    required = config.get("required_role", Role.ADMIN)
    return _ROLE_LEVEL.get(user_role, 0) >= _ROLE_LEVEL.get(required, 999)


def can_approve(user_role: str, action_type: str) -> bool:
    """Return True if `user_role` can approve a pending `action_type` request."""
    config = _MATRIX.get(action_type, {})
    approver_roles = config.get("approver_roles", [Role.ADMIN])
    return user_role in approver_roles


def should_auto_approve(action_type: str, confidence: float) -> bool:
    """
    Return True if `action_type` can be auto-approved at this confidence level.

    A return value of True means no human approval is required — the system
    may execute the action directly.
    """
    config = _MATRIX.get(action_type, {})
    threshold: Optional[float] = config.get("auto_approve_threshold")

    if threshold is None:
        return False  # Never auto-approve (e.g. delete)

    return confidence >= threshold

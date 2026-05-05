"""
Audit logging for authentication and authorization events.

Every auth decision — success or failure — is emitted as a structured JSON log
event tagged with `"audit_event"`. Downstream log aggregators (Loki, CloudWatch,
Datadog) can filter on this field to build an access audit trail without touching
application code.
"""

from datetime import datetime
from typing import Optional

from src.observability.logging_config import get_logger

log = get_logger("audit")


def log_auth_event(
    event_type: str,
    user_id: Optional[str],
    resource: Optional[str] = None,
    action: Optional[str] = None,
    result: str = "success",
    **kwargs,
) -> None:
    """
    Emit a structured audit log entry.

    Args:
        event_type: Logical event name (e.g. "authentication", "key_revoke").
        user_id:    Actor performing the action; None on unauthenticated failures.
        resource:   Object being acted on (e.g. key_id, endpoint path).
        action:     Verb describing the action (e.g. "create", "revoke").
        result:     "success" | "failure" | "denied"
        **kwargs:   Any additional structured context.
    """
    log.info(
        "audit_event",
        event_type=event_type,
        user_id=user_id,
        resource=resource,
        action=action,
        result=result,
        timestamp=datetime.now().isoformat(),
        **kwargs,
    )

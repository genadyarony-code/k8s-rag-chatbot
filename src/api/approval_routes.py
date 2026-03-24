"""
Approval workflow REST API.

Endpoints:
    GET  /approvals                        — list pending requests visible to the caller
    GET  /approvals/{request_id}           — get a single request
    POST /approvals/{request_id}/approve   — approve a pending request
    POST /approvals/{request_id}/reject    — reject a pending request

All endpoints require authentication (Bearer API key). The caller's role
determines which action types they can approve:
    approver role → can approve WRITE / KUBECTL_APPLY / API_CALL
    admin role    → can approve anything including DELETE / KUBECTL_DELETE

This router is a forward-looking skeleton for when the chatbot grows tools.
Today the approval queue is filled via unit tests or future tool-calling code;
no live action is taken by the chatbot itself.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.auth.dependencies import require_auth
from src.auth.storage import get_auth_storage
from src.hitl.approval_queue import get_approval_queue
from src.hitl.authorization import can_approve
from src.observability.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/approvals", tags=["approvals"])


# ── Request / Response models ─────────────────────────────────────────────────

class RejectBody(BaseModel):
    reason: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
async def list_pending_approvals(user_id: str = Depends(require_auth)):
    """
    Return all pending approval requests that the authenticated user can approve.

    Admins see everything; approvers see only their eligible action types.
    """
    queue = get_approval_queue()
    storage = get_auth_storage()

    user = storage.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    all_pending = queue.list_pending()
    visible = [r for r in all_pending if can_approve(user.role, r["action_type"])]

    log.info(
        "pending_approvals_listed",
        user_id=user_id,
        role=user.role,
        total_pending=len(all_pending),
        visible=len(visible),
    )

    return {"pending": visible, "total": len(visible)}


@router.get("/{request_id}")
async def get_approval_request(
    request_id: str,
    user_id: str = Depends(require_auth),
):
    """Return the full details of an approval request by ID."""
    queue = get_approval_queue()
    request = queue.get(request_id)

    if not request:
        raise HTTPException(status_code=404, detail="Approval request not found")

    return request


@router.post("/{request_id}/approve")
async def approve_request(
    request_id: str,
    user_id: str = Depends(require_auth),
):
    """Approve a pending action. Returns 403 if caller lacks the required role."""
    queue = get_approval_queue()
    storage = get_auth_storage()

    request = queue.get(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Approval request not found")

    user = storage.get_user(user_id)
    if not user or not can_approve(user.role, request["action_type"]):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Role '{getattr(user, 'role', 'unknown')}' cannot approve "
                f"'{request['action_type']}' actions"
            ),
        )

    success = queue.approve(request_id, user_id)
    if not success:
        raise HTTPException(
            status_code=409,
            detail="Request already processed or not found",
        )

    log.info("approval_api_granted", request_id=request_id, approver=user_id)
    return {"message": "Approved", "request_id": request_id}


@router.post("/{request_id}/reject")
async def reject_request(
    request_id: str,
    body: RejectBody,
    user_id: str = Depends(require_auth),
):
    """Reject a pending action with a mandatory reason."""
    queue = get_approval_queue()
    storage = get_auth_storage()

    request = queue.get(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Approval request not found")

    user = storage.get_user(user_id)
    if not user or not can_approve(user.role, request["action_type"]):
        raise HTTPException(status_code=403, detail="Not authorised to reject this action")

    success = queue.reject(request_id, user_id, body.reason)
    if not success:
        raise HTTPException(status_code=409, detail="Request already processed or not found")

    log.info("approval_api_rejected", request_id=request_id, rejector=user_id)
    return {"message": "Rejected", "request_id": request_id}

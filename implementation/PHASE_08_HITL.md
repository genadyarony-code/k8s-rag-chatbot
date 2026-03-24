# PHASE 8: Human-in-the-Loop Framework

> **Week:** 6-7  
> **Priority:** P1 (Production Readiness - for when tools are added)  
> **Duration:** 4-5 days  
> **Dependencies:** PHASE 7 (confidence scoring from eval)

---

## Objective

Implement confidence scoring, approval workflow skeleton, action authorization matrix, and approval audit trail.

**Why this matters:**  
From reference: *"When the agent has tools, HITL is not optional. It's a safety requirement."*  
Right now the chatbot is read-only, but when you add tools (kubectl commands, API calls), you need approval workflows.

---

## Pre-Flight Checklist

- [ ] PHASE 1-7 completed
- [ ] Evaluation framework working
- [ ] All tests passing

---

## Task 1: Install Dependencies

**Add to `requirements.txt`:**

```
# For approval queue (Redis-backed)
redis==5.0.1  # Already added in Phase 3

# For async task queue (optional, for background approvals)
celery==5.3.4
```

**Install:**
```bash
pip install celery==5.3.4
```

---

## Task 2: Create Confidence Scorer

**Create:** `src/hitl/__init__.py`

```python
# Empty init
```

**Create:** `src/hitl/confidence.py`

```python
"""
Confidence scoring for responses.

Confidence = f(retrieval_quality, generation_confidence, citation_validation)

High confidence (>0.8): Auto-approve
Medium confidence (0.5-0.8): Optional review
Low confidence (<0.5): Require approval
"""

from typing import Dict
import math

from src.observability.logging_config import get_logger

log = get_logger(__name__)


def calculate_confidence(
    retrieval_score: float,
    citation_valid: bool,
    eval_score: float = None,
    source_count: int = 0
) -> float:
    """
    Calculate confidence score for a response.
    
    Args:
        retrieval_score: Top retrieval score (0-1)
        citation_valid: Whether citations are grounded
        eval_score: Optional LLM judge score (1-5)
        source_count: Number of sources used
        
    Returns:
        Confidence score (0-1)
        
    Formula:
        confidence = 0.4 * retrieval_score + 
                     0.3 * citation_score + 
                     0.2 * eval_score_normalized + 
                     0.1 * source_diversity
    """
    
    # Retrieval component (40% weight)
    retrieval_component = retrieval_score * 0.4
    
    # Citation component (30% weight)
    citation_component = (1.0 if citation_valid else 0.3) * 0.3
    
    # Eval component (20% weight)
    if eval_score is not None:
        eval_normalized = (eval_score - 1) / 4.0  # 1-5 → 0-1
        eval_component = eval_normalized * 0.2
    else:
        eval_component = 0.5 * 0.2  # Neutral if no eval
    
    # Source diversity component (10% weight)
    # More sources = higher confidence (up to 5 sources)
    source_diversity = min(source_count / 5.0, 1.0)
    source_component = source_diversity * 0.1
    
    confidence = retrieval_component + citation_component + eval_component + source_component
    
    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    
    log.info(
        "confidence_calculated",
        confidence=confidence,
        retrieval_score=retrieval_score,
        citation_valid=citation_valid,
        source_count=source_count
    )
    
    return confidence


def get_confidence_level(confidence: float) -> str:
    """
    Map confidence score to level.
    
    Returns:
        "high" | "medium" | "low"
    """
    
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"


def requires_approval(confidence: float, action_type: str = "read") -> bool:
    """
    Determine if action requires approval based on confidence and type.
    
    Args:
        confidence: Confidence score (0-1)
        action_type: Type of action (read | write | delete)
        
    Returns:
        True if approval required
    """
    
    # Read-only actions (current chatbot)
    if action_type == "read":
        # Only require approval for very low confidence
        return confidence < 0.3
    
    # Write actions (future: kubectl apply, API updates)
    elif action_type == "write":
        # Require approval unless very high confidence
        return confidence < 0.9
    
    # Delete actions (future: kubectl delete)
    elif action_type == "delete":
        # Always require approval
        return True
    
    # Unknown action type - require approval
    return True
```

---

## Task 3: Create Approval Queue

**Create:** `src/hitl/approval_queue.py`

```python
"""
Approval queue for pending actions.

Storage: Redis (production) or in-memory (development)

Queue operations:
- Submit action for approval
- Approve/reject action
- List pending actions
- Get action status
"""

import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import secrets

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.observability.logging_config import get_logger

log = get_logger(__name__)


class ApprovalQueue:
    """
    Queue for pending approval actions.
    
    Uses Redis if available, otherwise in-memory storage.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize approval queue.
        
        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379/0")
                If None, uses in-memory storage
        """
        
        if redis_url and REDIS_AVAILABLE:
            self.redis = redis.from_url(redis_url)
            self.storage_type = "redis"
            log.info("approval_queue_initialized", storage="redis", url=redis_url)
        else:
            self.redis = None
            self.storage_type = "memory"
            self._memory_store: Dict[str, Dict] = {}
            log.info("approval_queue_initialized", storage="memory")
    
    def submit(
        self,
        action_type: str,
        action_data: Dict,
        confidence: float,
        session_id: str,
        user_id: str
    ) -> str:
        """
        Submit an action for approval.
        
        Args:
            action_type: Type of action (e.g., "kubectl_apply", "api_call")
            action_data: Action details
            confidence: Confidence score
            session_id: Session identifier
            user_id: User who initiated the action
            
        Returns:
            Approval request ID
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
            "rejection_reason": None
        }
        
        # Store
        if self.storage_type == "redis":
            self.redis.set(
                f"approval:{request_id}",
                json.dumps(request),
                ex=86400  # Expire after 24 hours
            )
            self.redis.sadd("approval:pending", request_id)
        else:
            self._memory_store[request_id] = request
        
        log.info(
            "approval_submitted",
            request_id=request_id,
            action_type=action_type,
            confidence=confidence,
            user_id=user_id
        )
        
        return request_id
    
    def get(self, request_id: str) -> Optional[Dict]:
        """Get approval request by ID."""
        
        if self.storage_type == "redis":
            data = self.redis.get(f"approval:{request_id}")
            return json.loads(data) if data else None
        else:
            return self._memory_store.get(request_id)
    
    def approve(self, request_id: str, approver_id: str) -> bool:
        """
        Approve a pending request.
        
        Args:
            request_id: Request ID
            approver_id: User ID of approver
            
        Returns:
            True if approved successfully
        """
        
        request = self.get(request_id)
        
        if not request:
            log.warning("approval_not_found", request_id=request_id)
            return False
        
        if request["status"] != "pending":
            log.warning("approval_already_processed", request_id=request_id, status=request["status"])
            return False
        
        request["status"] = "approved"
        request["approved_at"] = datetime.now().isoformat()
        request["approved_by"] = approver_id
        
        # Update storage
        if self.storage_type == "redis":
            self.redis.set(f"approval:{request_id}", json.dumps(request))
            self.redis.srem("approval:pending", request_id)
        else:
            self._memory_store[request_id] = request
        
        log.info(
            "approval_granted",
            request_id=request_id,
            approver_id=approver_id,
            action_type=request["action_type"]
        )
        
        return True
    
    def reject(self, request_id: str, rejector_id: str, reason: str) -> bool:
        """
        Reject a pending request.
        
        Args:
            request_id: Request ID
            rejector_id: User ID of rejector
            reason: Rejection reason
            
        Returns:
            True if rejected successfully
        """
        
        request = self.get(request_id)
        
        if not request:
            return False
        
        if request["status"] != "pending":
            return False
        
        request["status"] = "rejected"
        request["approved_at"] = datetime.now().isoformat()
        request["approved_by"] = rejector_id
        request["rejection_reason"] = reason
        
        # Update storage
        if self.storage_type == "redis":
            self.redis.set(f"approval:{request_id}", json.dumps(request))
            self.redis.srem("approval:pending", request_id)
        else:
            self._memory_store[request_id] = request
        
        log.info(
            "approval_rejected",
            request_id=request_id,
            rejector_id=rejector_id,
            reason=reason
        )
        
        return True
    
    def list_pending(self, user_id: Optional[str] = None) -> List[Dict]:
        """
        List pending approval requests.
        
        Args:
            user_id: Optional filter by user
            
        Returns:
            List of pending requests
        """
        
        if self.storage_type == "redis":
            request_ids = self.redis.smembers("approval:pending")
            requests = []
            for req_id in request_ids:
                req_id = req_id.decode() if isinstance(req_id, bytes) else req_id
                req = self.get(req_id)
                if req and (not user_id or req["user_id"] == user_id):
                    requests.append(req)
            return requests
        else:
            requests = [
                r for r in self._memory_store.values()
                if r["status"] == "pending" and (not user_id or r["user_id"] == user_id)
            ]
            return requests


# Global queue instance
_queue = None


def get_approval_queue() -> ApprovalQueue:
    """Get or create the global approval queue."""
    global _queue
    if _queue is None:
        # Try Redis, fallback to memory
        redis_url = None  # Set from settings in production
        _queue = ApprovalQueue(redis_url)
    return _queue
```

---

## Task 4: Create Authorization Matrix

**Create:** `src/hitl/authorization.py`

```python
"""
Authorization matrix for action approval.

Defines which actions require approval and who can approve them.
"""

from typing import Dict, List
from enum import Enum

from src.observability.logging_config import get_logger

log = get_logger(__name__)


class ActionType(str, Enum):
    """Supported action types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    KUBECTL_GET = "kubectl_get"
    KUBECTL_APPLY = "kubectl_apply"
    KUBECTL_DELETE = "kubectl_delete"
    API_CALL = "api_call"


class Role(str, Enum):
    """User roles."""
    USER = "user"
    ADMIN = "admin"
    APPROVER = "approver"


# Authorization matrix
# Format: {action_type: {required_role, auto_approve_threshold}}
AUTHORIZATION_MATRIX: Dict[str, Dict] = {
    ActionType.READ: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.3,
        "requires_approval": False
    },
    ActionType.WRITE: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.9,
        "requires_approval": True,
        "approver_roles": [Role.APPROVER, Role.ADMIN]
    },
    ActionType.DELETE: {
        "required_role": Role.ADMIN,
        "auto_approve_threshold": None,  # Never auto-approve
        "requires_approval": True,
        "approver_roles": [Role.ADMIN]
    },
    ActionType.KUBECTL_GET: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.5,
        "requires_approval": False
    },
    ActionType.KUBECTL_APPLY: {
        "required_role": Role.USER,
        "auto_approve_threshold": 0.95,
        "requires_approval": True,
        "approver_roles": [Role.APPROVER, Role.ADMIN]
    },
    ActionType.KUBECTL_DELETE: {
        "required_role": Role.APPROVER,
        "auto_approve_threshold": None,
        "requires_approval": True,
        "approver_roles": [Role.ADMIN]
    },
}


def can_execute(user_role: str, action_type: str) -> bool:
    """
    Check if user has permission to execute action.
    
    Args:
        user_role: User's role
        action_type: Action type
        
    Returns:
        True if user can execute (subject to approval)
    """
    
    config = AUTHORIZATION_MATRIX.get(action_type, {})
    required_role = config.get("required_role", Role.ADMIN)
    
    # Simple role hierarchy: ADMIN > APPROVER > USER
    role_levels = {Role.USER: 1, Role.APPROVER: 2, Role.ADMIN: 3}
    
    user_level = role_levels.get(user_role, 0)
    required_level = role_levels.get(required_role, 999)
    
    return user_level >= required_level


def can_approve(user_role: str, action_type: str) -> bool:
    """
    Check if user can approve this action type.
    
    Args:
        user_role: Approver's role
        action_type: Action type
        
    Returns:
        True if user can approve
    """
    
    config = AUTHORIZATION_MATRIX.get(action_type, {})
    approver_roles = config.get("approver_roles", [Role.ADMIN])
    
    return user_role in approver_roles


def should_auto_approve(action_type: str, confidence: float) -> bool:
    """
    Determine if action should be auto-approved based on confidence.
    
    Args:
        action_type: Action type
        confidence: Confidence score (0-1)
        
    Returns:
        True if should auto-approve
    """
    
    config = AUTHORIZATION_MATRIX.get(action_type, {})
    threshold = config.get("auto_approve_threshold")
    
    if threshold is None:
        return False  # Never auto-approve
    
    return confidence >= threshold
```

---

## Task 5: Create Approval API Endpoints

**Create:** `src/api/approval_routes.py`

```python
"""
Approval workflow API endpoints.

Endpoints:
- GET /approvals - List pending approvals
- POST /approvals/{request_id}/approve - Approve request
- POST /approvals/{request_id}/reject - Reject request
- GET /approvals/{request_id} - Get request details
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.hitl.approval_queue import get_approval_queue
from src.hitl.authorization import can_approve
from src.auth.dependencies import require_auth
from src.auth.storage import get_auth_storage
from src.observability.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/approvals", tags=["approvals"])


# ── Request/Response Models ─────────────────────────────────────────────

class ApproveRequest(BaseModel):
    pass  # Empty for now


class RejectRequest(BaseModel):
    reason: str


# ── Endpoints ───────────────────────────────────────────────────────────

@router.get("")
async def list_pending_approvals(user_id: str = Depends(require_auth)):
    """
    List pending approval requests.
    
    Filters to only show requests the user can approve.
    """
    
    queue = get_approval_queue()
    storage = get_auth_storage()
    
    user = storage.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get all pending
    all_pending = queue.list_pending()
    
    # Filter to approvalable by this user
    approvalable = [
        req for req in all_pending
        if can_approve(user.role, req["action_type"])
    ]
    
    log.info(
        "pending_approvals_listed",
        user_id=user_id,
        total_pending=len(all_pending),
        approvalable=len(approvalable)
    )
    
    return {"pending": approvalable}


@router.get("/{request_id}")
async def get_approval_request(
    request_id: str,
    user_id: str = Depends(require_auth)
):
    """Get details of an approval request."""
    
    queue = get_approval_queue()
    request = queue.get(request_id)
    
    if not request:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    return request


@router.post("/{request_id}/approve")
async def approve_request(
    request_id: str,
    user_id: str = Depends(require_auth)
):
    """Approve a pending request."""
    
    queue = get_approval_queue()
    storage = get_auth_storage()
    
    # Get request
    request = queue.get(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    # Check authorization
    user = storage.get_user(user_id)
    if not user or not can_approve(user.role, request["action_type"]):
        raise HTTPException(
            status_code=403,
            detail=f"User role '{user.role}' cannot approve '{request['action_type']}' actions"
        )
    
    # Approve
    success = queue.approve(request_id, user_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Could not approve request")
    
    return {"message": "Request approved", "request_id": request_id}


@router.post("/{request_id}/reject")
async def reject_request(
    request_id: str,
    body: RejectRequest,
    user_id: str = Depends(require_auth)
):
    """Reject a pending request."""
    
    queue = get_approval_queue()
    storage = get_auth_storage()
    
    # Get request
    request = queue.get(request_id)
    if not request:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    # Check authorization
    user = storage.get_user(user_id)
    if not user or not can_approve(user.role, request["action_type"]):
        raise HTTPException(status_code=403, detail="Not authorized to reject")
    
    # Reject
    success = queue.reject(request_id, user_id, body.reason)
    
    if not success:
        raise HTTPException(status_code=400, detail="Could not reject request")
    
    return {"message": "Request rejected", "request_id": request_id}
```

**Include in main app:**

In `src/api/main.py`:

```python
from src.api.approval_routes import router as approval_router

app.include_router(approval_router)
```

---

## Task 6: Integrate Confidence Scoring into Chat

**Modify:** `src/agent/nodes.py`

**In `generate_node`, add confidence calculation:**

```python
from src.hitl.confidence import calculate_confidence, get_confidence_level

def generate_node(state: dict) -> dict:
    # ... existing generation code ...
    
    # After citation validation
    validator = get_validator()
    is_valid, unsupported = validator.validate(answer, state["context"], threshold=0.5)
    
    # ── CONFIDENCE SCORING ──────────────────────────────────────────────
    
    # Get top retrieval score
    retrieval_score = state["context"][0]["score"] if state["context"] else 0.0
    
    # Calculate confidence
    confidence = calculate_confidence(
        retrieval_score=retrieval_score,
        citation_valid=is_valid,
        source_count=len(sources)
    )
    
    confidence_level = get_confidence_level(confidence)
    
    span.set_attribute("confidence_score", confidence)
    span.set_attribute("confidence_level", confidence_level)
    
    log.info(
        "response_confidence_calculated",
        session_id=state["session_id"],
        confidence=confidence,
        level=confidence_level
    )
    
    # ────────────────────────────────────────────────────────────────────
    
    # ... rest of code ...
    
    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,  # ← ADD THIS
        "confidence_level": confidence_level  # ← ADD THIS
    }
```

**Update ChatResponse model:**

In `src/api/main.py`:

```python
class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str
    confidence: Optional[float] = None  # ← ADD THIS
    confidence_level: Optional[str] = None  # ← ADD THIS
```

---

## Task 7: Write HITL Tests

**Create:** `tests/test_hitl.py`

```python
"""
Tests for HITL (Human-in-the-Loop) framework.
"""

import pytest

from src.hitl.confidence import calculate_confidence, get_confidence_level, requires_approval
from src.hitl.approval_queue import ApprovalQueue
from src.hitl.authorization import can_execute, can_approve, should_auto_approve, ActionType, Role


# ── Confidence Tests ────────────────────────────────────────────────────

def test_confidence_calculation():
    """Test confidence score calculation."""
    
    # High confidence
    conf1 = calculate_confidence(
        retrieval_score=0.9,
        citation_valid=True,
        source_count=5
    )
    assert conf1 > 0.7
    assert get_confidence_level(conf1) in ["high", "medium"]
    
    # Low confidence
    conf2 = calculate_confidence(
        retrieval_score=0.3,
        citation_valid=False,
        source_count=1
    )
    assert conf2 < 0.5
    assert get_confidence_level(conf2) == "low"


def test_requires_approval():
    """Test approval requirement logic."""
    
    # Read actions don't require approval (unless very low confidence)
    assert not requires_approval(0.8, "read")
    assert requires_approval(0.2, "read")
    
    # Write actions require approval unless very high confidence
    assert requires_approval(0.7, "write")
    assert not requires_approval(0.95, "write")
    
    # Delete actions always require approval
    assert requires_approval(1.0, "delete")


# ── Approval Queue Tests ────────────────────────────────────────────────

def test_approval_queue():
    """Test approval queue operations."""
    
    queue = ApprovalQueue()  # In-memory
    
    # Submit request
    request_id = queue.submit(
        action_type="kubectl_apply",
        action_data={"manifest": "..."},
        confidence=0.7,
        session_id="test",
        user_id="user123"
    )
    
    assert request_id is not None
    
    # Get request
    request = queue.get(request_id)
    assert request["status"] == "pending"
    assert request["user_id"] == "user123"
    
    # List pending
    pending = queue.list_pending()
    assert len(pending) >= 1
    
    # Approve
    success = queue.approve(request_id, "approver123")
    assert success
    
    # Check status
    request = queue.get(request_id)
    assert request["status"] == "approved"
    assert request["approved_by"] == "approver123"


# ── Authorization Tests ─────────────────────────────────────────────────

def test_authorization_matrix():
    """Test authorization checks."""
    
    # Users can execute reads
    assert can_execute(Role.USER, ActionType.READ)
    
    # Users can execute writes (subject to approval)
    assert can_execute(Role.USER, ActionType.WRITE)
    
    # Only admins can execute deletes
    assert not can_execute(Role.USER, ActionType.DELETE)
    assert can_execute(Role.ADMIN, ActionType.DELETE)


def test_approval_authorization():
    """Test who can approve what."""
    
    # Approvers can approve writes
    assert can_approve(Role.APPROVER, ActionType.WRITE)
    
    # Only admins can approve deletes
    assert not can_approve(Role.APPROVER, ActionType.DELETE)
    assert can_approve(Role.ADMIN, ActionType.DELETE)


def test_auto_approve_logic():
    """Test auto-approval based on confidence."""
    
    # Reads auto-approve at low threshold
    assert should_auto_approve(ActionType.READ, 0.4)
    
    # Writes need very high confidence
    assert not should_auto_approve(ActionType.WRITE, 0.7)
    assert should_auto_approve(ActionType.WRITE, 0.95)
    
    # Deletes never auto-approve
    assert not should_auto_approve(ActionType.DELETE, 1.0)
```

**Run tests:**
```bash
pytest tests/test_hitl.py -v
```

---

## Verification Steps

**1. Test confidence scoring:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a Pod?",
    "session_id": "confidence_test"
  }'

# Check response - should include confidence and confidence_level
```

**2. Test approval queue (simulated):**

```python
# test_approval_flow.py
from src.hitl.approval_queue import get_approval_queue

queue = get_approval_queue()

# Submit action
request_id = queue.submit(
    action_type="kubectl_apply",
    action_data={"command": "kubectl apply -f deployment.yaml"},
    confidence=0.6,
    session_id="test_session",
    user_id="user_test"
)

print(f"Submitted: {request_id}")

# List pending
pending = queue.list_pending()
print(f"Pending: {len(pending)}")

# Approve
queue.approve(request_id, "admin_user")
print("Approved!")

# Check status
request = queue.get(request_id)
print(f"Status: {request['status']}")
```

**3. Test approval API:**

```bash
# Create an admin user first (or use existing)
# Then submit and approve via API

curl http://localhost:8000/approvals \
  -H "Authorization: Bearer $ADMIN_API_KEY"

# Should list pending approvals
```

---

## Success Criteria

- [ ] Confidence scores calculated for all responses
- [ ] Approval queue stores requests
- [ ] Approval/rejection workflow works
- [ ] Authorization matrix enforced
- [ ] Low confidence responses flagged
- [ ] All tests pass: `pytest tests/test_hitl.py -v`

---

## Future Enhancements

When tools are added:

1. **Tool call interception**: Intercept tool calls, check confidence, submit for approval if needed
2. **Approval UI**: Build React UI for approval dashboard
3. **Notifications**: Send Slack/email when approval needed
4. **Audit trail**: Full audit log of all approvals/rejections

---

## Next Phase

Once verified:

**→ Proceed to `PHASE_09_ADVANCED.md`**

Advanced features (model fallback, caching) are optional enhancements.

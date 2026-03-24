# PHASE 4: Authentication & Authorization

> **Week:** 3  
> **Priority:** P1 (Production Readiness)  
> **Duration:** 4-5 days  
> **Dependencies:** PHASE 1 (logging), PHASE 3 (cost controls need user IDs)

---

## Objective

Implement API key authentication, user management, and audit logging. Prepare authorization framework for future RBAC.

**Why this matters:**  
Right now, anyone can use the API. No user tracking, no accountability, no way to tie costs to users.

---

## Pre-Flight Checklist

- [ ] PHASE 1-3 completed
- [ ] Cost tracking working (needs user_id field)
- [ ] All tests passing

---

## Task 1: Install Dependencies

**Add to `requirements.txt`:**

```
# Authentication
pyjwt==2.8.0
passlib==1.7.4
python-multipart==0.0.9

# API key generation
secrets  # Built-in Python module
```

**Install:**
```bash
pip install pyjwt==2.8.0 passlib==1.7.4 python-multipart==0.0.9
```

---

## Task 2: Create User Model & Storage

**Create:** `src/auth/__init__.py`

```python
# Empty init
```

**Create:** `src/auth/models.py`

```python
"""
User and API Key models.

For this phase, we use in-memory storage.
Production: Replace with PostgreSQL/MongoDB.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    """
    User account.
    
    Fields:
    - user_id: Unique identifier (UUID)
    - email: User email (unique)
    - name: Display name
    - created_at: Account creation timestamp
    - is_active: Account status
    - role: User role (for future RBAC)
    """
    
    user_id: str
    email: str
    name: str
    created_at: datetime
    is_active: bool = True
    role: str = "user"  # user | admin
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active,
            "role": self.role,
        }


@dataclass
class APIKey:
    """
    API key for authentication.
    
    Fields:
    - key_id: Key identifier (shown to user)
    - key_hash: Hashed key value (stored securely)
    - user_id: Owner of this key
    - name: Human-readable name (e.g., "Production Server")
    - created_at: Key creation timestamp
    - last_used_at: Last request timestamp
    - is_active: Key status
    """
    
    key_id: str
    key_hash: str
    user_id: str
    name: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    
    def to_dict(self) -> dict:
        """Serialize to dict (never expose key_hash)."""
        return {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
        }
```

**Create:** `src/auth/storage.py`

```python
"""
In-memory storage for users and API keys.

Production: Replace with database.
"""

from typing import Optional
from datetime import datetime

from src.auth.models import User, APIKey


class InMemoryAuthStorage:
    """
    Thread-safe in-memory storage for auth data.
    
    Warning: Data is lost on restart. Use database in production.
    """
    
    def __init__(self):
        self._users: dict[str, User] = {}           # user_id -> User
        self._users_by_email: dict[str, User] = {}  # email -> User
        self._api_keys: dict[str, APIKey] = {}      # key_hash -> APIKey
        self._keys_by_id: dict[str, APIKey] = {}    # key_id -> APIKey
    
    # ── Users ───────────────────────────────────────────────────────────
    
    def create_user(self, user: User) -> User:
        """Create a new user."""
        if user.user_id in self._users:
            raise ValueError(f"User {user.user_id} already exists")
        if user.email in self._users_by_email:
            raise ValueError(f"Email {user.email} already registered")
        
        self._users[user.user_id] = user
        self._users_by_email[user.email] = user
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self._users_by_email.get(email)
    
    def list_users(self) -> list[User]:
        """List all users."""
        return list(self._users.values())
    
    # ── API Keys ────────────────────────────────────────────────────────
    
    def create_api_key(self, api_key: APIKey) -> APIKey:
        """Create a new API key."""
        if api_key.key_hash in self._api_keys:
            raise ValueError(f"Key hash collision")
        
        self._api_keys[api_key.key_hash] = api_key
        self._keys_by_id[api_key.key_id] = api_key
        return api_key
    
    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        return self._api_keys.get(key_hash)
    
    def get_api_key_by_id(self, key_id: str) -> Optional[APIKey]:
        """Get API key by ID."""
        return self._keys_by_id.get(key_id)
    
    def list_user_keys(self, user_id: str) -> list[APIKey]:
        """List all keys for a user."""
        return [k for k in self._api_keys.values() if k.user_id == user_id]
    
    def update_key_last_used(self, key_hash: str):
        """Update last_used_at timestamp."""
        if key_hash in self._api_keys:
            self._api_keys[key_hash].last_used_at = datetime.now()
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys_by_id:
            self._keys_by_id[key_id].is_active = False
            return True
        return False


# Global storage instance
_storage: Optional[InMemoryAuthStorage] = None


def get_auth_storage() -> InMemoryAuthStorage:
    """Get or create the global auth storage."""
    global _storage
    if _storage is None:
        _storage = InMemoryAuthStorage()
    return _storage
```

---

## Task 3: Create API Key Manager

**Create:** `src/auth/api_keys.py`

```python
"""
API key generation and validation.

Key format: k8s_rag_[32 random chars]
Example: k8s_rag_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

Security:
- Keys are hashed with bcrypt before storage
- Only key_id and prefix are shown to user after creation
- Full key is shown ONCE during creation
"""

import secrets
import string
from datetime import datetime

from passlib.hash import bcrypt

from src.auth.models import APIKey
from src.auth.storage import get_auth_storage
from src.observability.logging_config import get_logger

log = get_logger(__name__)

# Key configuration
KEY_PREFIX = "k8s_rag_"
KEY_LENGTH = 32  # Random portion length
KEY_ALPHABET = string.ascii_letters + string.digits


def generate_api_key() -> str:
    """
    Generate a new API key.
    
    Format: k8s_rag_[32 random chars]
    
    Returns:
        Full API key (show this ONCE to user)
    """
    random_part = ''.join(secrets.choice(KEY_ALPHABET) for _ in range(KEY_LENGTH))
    return f"{KEY_PREFIX}{random_part}"


def hash_api_key(key: str) -> str:
    """
    Hash an API key for secure storage.
    
    Uses bcrypt with default rounds (12).
    """
    return bcrypt.hash(key)


def verify_api_key(key: str, key_hash: str) -> bool:
    """
    Verify an API key against its hash.
    
    Args:
        key: Plain API key
        key_hash: Stored hash
        
    Returns:
        True if key matches hash
    """
    return bcrypt.verify(key, key_hash)


def create_api_key(user_id: str, name: str) -> tuple[str, APIKey]:
    """
    Create a new API key for a user.
    
    Args:
        user_id: Owner of the key
        name: Human-readable key name
        
    Returns:
        (full_key, api_key_object)
        
    Important: full_key must be shown to user ONCE and never stored.
    """
    storage = get_auth_storage()
    
    # Generate key
    full_key = generate_api_key()
    key_hash = hash_api_key(full_key)
    
    # Create key object
    api_key = APIKey(
        key_id=f"key_{secrets.token_hex(8)}",
        key_hash=key_hash,
        user_id=user_id,
        name=name,
        created_at=datetime.now(),
    )
    
    # Store
    storage.create_api_key(api_key)
    
    log.info(
        "api_key_created",
        user_id=user_id,
        key_id=api_key.key_id,
        name=name
    )
    
    return full_key, api_key


def validate_api_key(key: str) -> tuple[bool, Optional[str]]:
    """
    Validate an API key and return user_id if valid.
    
    Args:
        key: API key from request header
        
    Returns:
        (is_valid, user_id)
    """
    storage = get_auth_storage()
    
    # Check all stored keys (in production, use indexed lookup by prefix)
    for api_key in storage._api_keys.values():
        if not api_key.is_active:
            continue
        
        if verify_api_key(key, api_key.key_hash):
            # Update last used timestamp
            storage.update_key_last_used(api_key.key_hash)
            
            log.info(
                "api_key_validated",
                user_id=api_key.user_id,
                key_id=api_key.key_id
            )
            
            return True, api_key.user_id
    
    log.warning("api_key_validation_failed", key_prefix=key[:15] if key else None)
    return False, None
```

---

## Task 4: Create Authentication Dependency

**Create:** `src/auth/dependencies.py`

```python
"""
FastAPI dependencies for authentication.

Usage in endpoints:
    @app.get("/protected")
    async def protected(user_id: str = Depends(require_auth)):
        return {"user_id": user_id}
"""

from typing import Optional
from fastapi import Header, HTTPException, Depends

from src.auth.api_keys import validate_api_key
from src.auth.storage import get_auth_storage
from src.observability.logging_config import get_logger

log = get_logger(__name__)


async def get_api_key(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Extract API key from Authorization header.
    
    Expected format: "Bearer k8s_rag_..."
    """
    if not authorization:
        return None
    
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    
    return parts[1]


async def require_auth(api_key: Optional[str] = Depends(get_api_key)) -> str:
    """
    Require authentication for an endpoint.
    
    Returns:
        user_id if authenticated
        
    Raises:
        HTTPException(401) if not authenticated
        HTTPException(403) if user is inactive
    """
    
    if not api_key:
        log.warning("auth_missing", reason="no_authorization_header")
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Include 'Authorization: Bearer <api_key>' header."
        )
    
    is_valid, user_id = validate_api_key(api_key)
    
    if not is_valid or not user_id:
        log.warning("auth_invalid", reason="invalid_api_key")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    # Check if user is active
    storage = get_auth_storage()
    user = storage.get_user(user_id)
    
    if not user or not user.is_active:
        log.warning("auth_user_inactive", user_id=user_id)
        raise HTTPException(
            status_code=403,
            detail="User account is inactive"
        )
    
    log.info("auth_success", user_id=user_id)
    return user_id


async def optional_auth(api_key: Optional[str] = Depends(get_api_key)) -> Optional[str]:
    """
    Optional authentication (for endpoints that work with or without auth).
    
    Returns:
        user_id if authenticated, None otherwise
    """
    
    if not api_key:
        return None
    
    is_valid, user_id = validate_api_key(api_key)
    return user_id if is_valid else None
```

---

## Task 5: Create User Management Endpoints

**Create:** `src/api/auth_routes.py`

```python
"""
Authentication and user management routes.

Endpoints:
- POST /auth/users - Create user (admin only in production)
- POST /auth/keys - Create API key
- GET /auth/keys - List user's keys
- DELETE /auth/keys/{key_id} - Revoke key
- GET /auth/me - Get current user info
"""

from datetime import datetime
import secrets

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr

from src.auth.models import User
from src.auth.storage import get_auth_storage
from src.auth.api_keys import create_api_key
from src.auth.dependencies import require_auth, optional_auth
from src.observability.logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


# ── Request/Response Models ─────────────────────────────────────────────

class CreateUserRequest(BaseModel):
    email: EmailStr
    name: str


class CreateUserResponse(BaseModel):
    user_id: str
    email: str
    name: str
    message: str


class CreateKeyRequest(BaseModel):
    name: str  # e.g., "Production Server", "Development"


class CreateKeyResponse(BaseModel):
    key_id: str
    api_key: str  # Full key, shown ONCE
    name: str
    message: str


class KeyInfo(BaseModel):
    key_id: str
    name: str
    created_at: str
    last_used_at: Optional[str]
    is_active: bool


class UserInfo(BaseModel):
    user_id: str
    email: str
    name: str
    role: str
    created_at: str


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/users", response_model=CreateUserResponse)
async def create_user(request: CreateUserRequest):
    """
    Create a new user account.
    
    For this demo, anyone can create users.
    In production: require admin role or use OAuth.
    """
    storage = get_auth_storage()
    
    # Check if email already exists
    existing = storage.get_user_by_email(request.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        user_id=f"user_{secrets.token_hex(16)}",
        email=request.email,
        name=request.name,
        created_at=datetime.now(),
    )
    
    storage.create_user(user)
    
    log.info("user_created", user_id=user.user_id, email=user.email)
    
    return CreateUserResponse(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        message="User created. Use POST /auth/keys to create an API key."
    )


@router.post("/keys", response_model=CreateKeyResponse)
async def create_key(
    request: CreateKeyRequest,
    user_id: str = Depends(require_auth)
):
    """
    Create a new API key for the authenticated user.
    
    Returns the full key ONCE. Save it securely.
    """
    
    full_key, api_key = create_api_key(user_id, request.name)
    
    return CreateKeyResponse(
        key_id=api_key.key_id,
        api_key=full_key,
        name=request.name,
        message="Save this key securely. It will not be shown again."
    )


@router.get("/keys", response_model=list[KeyInfo])
async def list_keys(user_id: str = Depends(require_auth)):
    """
    List all API keys for the authenticated user.
    """
    storage = get_auth_storage()
    keys = storage.list_user_keys(user_id)
    
    return [
        KeyInfo(
            key_id=k.key_id,
            name=k.name,
            created_at=k.created_at.isoformat(),
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            is_active=k.is_active
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: str, user_id: str = Depends(require_auth)):
    """
    Revoke an API key.
    
    Only the key owner can revoke it.
    """
    storage = get_auth_storage()
    key = storage.get_api_key_by_id(key_id)
    
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    
    if key.user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to revoke this key")
    
    storage.revoke_key(key_id)
    
    log.info("api_key_revoked", user_id=user_id, key_id=key_id)
    
    return {"message": "Key revoked successfully"}


@router.get("/me", response_model=UserInfo)
async def get_current_user(user_id: str = Depends(require_auth)):
    """
    Get information about the authenticated user.
    """
    storage = get_auth_storage()
    user = storage.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserInfo(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        role=user.role,
        created_at=user.created_at.isoformat()
    )
```

---

## Task 6: Protect Chat Endpoint

**Modify:** `src/api/main.py`

**Add imports:**
```python
from src.api.auth_routes import router as auth_router
from src.auth.dependencies import require_auth
```

**Include auth routes:**
```python
app = FastAPI(title="K8s RAG Chatbot", version="1.0.0", lifespan=lifespan)

# Add auth routes
app.include_router(auth_router)

# ... rest of app setup
```

**Update `/chat` endpoint:**
```python
@app.post("/chat")
@limiter.limit("10/minute")
async def chat(
    request: ChatRequest,
    req: Request,
    user_id: str = Depends(require_auth)  # ← ADD THIS
):
    """
    Main chat endpoint (now requires authentication).
    """
    
    # Start timer
    start_time = time.time()
    
    # Validate input (existing code)
    try:
        validated_question = validate_chat_input(
            request.question,
            request.session_id,
            max_length=5000
        )
    except HTTPException:
        raise
    
    # Log with user_id
    chat_requests_total.labels(session_id=request.session_id).inc()
    
    log.info(
        "chat_request_received",
        user_id=user_id,  # ← ADD THIS
        session_id=request.session_id,
        question_length=len(validated_question)
    )
    
    # ... rest of code (pass user_id where needed)
```

**Update cost tracking to include user_id:**

In `src/agent/nodes.py`, when calling cost tracker:

```python
cost_usd, _ = tracker.track_request(
    session_id=state["session_id"],
    model=settings.llm_model,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    user_id=state.get("user_id")  # ← ADD THIS
)
```

And pass user_id in state:

```python
# In main.py, when calling graph.invoke:
result = await asyncio.to_thread(
    graph.invoke,
    {
        "question": validated_question,
        "session_id": request.session_id,
        "user_id": user_id  # ← ADD THIS
    },
    {"configurable": {"thread_id": request.session_id}},
)
```

---

## Task 7: Create Audit Log

**Create:** `src/auth/audit.py`

```python
"""
Audit logging for authentication and authorization events.

All auth events are logged:
- User creation
- API key creation/revocation
- Authentication attempts (success/failure)
- Authorization decisions

Logs are structured JSON for easy querying.
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
    **kwargs
):
    """
    Log an authentication/authorization event.
    
    Args:
        event_type: Event type (login, key_create, access_denied, etc.)
        user_id: User performing the action
        resource: Resource being accessed
        action: Action being performed
        result: "success" | "failure" | "denied"
        **kwargs: Additional context
    """
    
    log.info(
        "audit_event",
        event_type=event_type,
        user_id=user_id,
        resource=resource,
        action=action,
        result=result,
        timestamp=datetime.now().isoformat(),
        **kwargs
    )
```

**Use in dependencies:**

In `src/auth/dependencies.py`:

```python
from src.auth.audit import log_auth_event

async def require_auth(api_key: Optional[str] = Depends(get_api_key)) -> str:
    # ... existing code ...
    
    if not is_valid or not user_id:
        log_auth_event(
            event_type="authentication",
            user_id=None,
            result="failure",
            reason="invalid_api_key"
        )
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # ... existing code ...
    
    log_auth_event(
        event_type="authentication",
        user_id=user_id,
        result="success"
    )
    
    return user_id
```

---

## Task 8: Write Auth Tests

**Create:** `tests/test_auth.py`

```python
"""
Tests for authentication and authorization.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.auth.storage import get_auth_storage
from src.auth.api_keys import create_api_key, generate_api_key, hash_api_key, verify_api_key


# ── API Key Tests ───────────────────────────────────────────────────────

def test_generate_api_key():
    """Test that generated keys have correct format."""
    key = generate_api_key()
    
    assert key.startswith("k8s_rag_")
    assert len(key) == len("k8s_rag_") + 32


def test_hash_and_verify_key():
    """Test key hashing and verification."""
    key = generate_api_key()
    key_hash = hash_api_key(key)
    
    # Correct key verifies
    assert verify_api_key(key, key_hash)
    
    # Wrong key doesn't verify
    wrong_key = generate_api_key()
    assert not verify_api_key(wrong_key, key_hash)


def test_create_api_key():
    """Test API key creation."""
    storage = get_auth_storage()
    
    # Create test user
    from src.auth.models import User
    from datetime import datetime
    user = User(
        user_id="test_user",
        email="test@example.com",
        name="Test User",
        created_at=datetime.now()
    )
    storage.create_user(user)
    
    # Create key
    full_key, api_key = create_api_key("test_user", "Test Key")
    
    assert full_key.startswith("k8s_rag_")
    assert api_key.user_id == "test_user"
    assert api_key.name == "Test Key"
    assert api_key.is_active


# ── Authentication Tests ────────────────────────────────────────────────

def test_auth_endpoints():
    """Test authentication flow."""
    client = TestClient(app)
    
    # 1. Create user
    response = client.post("/auth/users", json={
        "email": "newuser@example.com",
        "name": "New User"
    })
    assert response.status_code == 200
    user_data = response.json()
    user_id = user_data["user_id"]
    
    # 2. Try to access protected endpoint without auth
    response = client.get("/auth/me")
    assert response.status_code == 401
    
    # 3. Create API key (need to auth first - chicken/egg problem)
    # In real scenario, keys are created via admin panel or OAuth
    # For test, we'll create directly
    storage = get_auth_storage()
    user = storage.get_user(user_id)
    full_key, _ = create_api_key(user_id, "Test Key")
    
    # 4. Access protected endpoint with auth
    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {full_key}"}
    )
    assert response.status_code == 200
    assert response.json()["user_id"] == user_id
    
    # 5. List keys
    response = client.get(
        "/auth/keys",
        headers={"Authorization": f"Bearer {full_key}"}
    )
    assert response.status_code == 200
    keys = response.json()
    assert len(keys) >= 1


def test_invalid_auth():
    """Test that invalid auth is rejected."""
    client = TestClient(app)
    
    # Wrong format
    response = client.get(
        "/auth/me",
        headers={"Authorization": "InvalidFormat"}
    )
    assert response.status_code == 401
    
    # Invalid key
    response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer k8s_rag_invalid_key_here"}
    )
    assert response.status_code == 401
```

**Run tests:**
```bash
pytest tests/test_auth.py -v
```

---

## Verification Steps

**1. Create a user:**

```bash
curl -X POST http://localhost:8000/auth/users \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "name": "Test User"
  }'

# Save the user_id from response
```

**2. Create API key (bootstrap - in production use admin panel):**

```python
# bootstrap_key.py
from src.auth.storage import get_auth_storage
from src.auth.api_keys import create_api_key

storage = get_auth_storage()
users = storage.list_users()

if users:
    user = users[0]
    full_key, api_key = create_api_key(user.user_id, "Bootstrap Key")
    print(f"API Key: {full_key}")
    print(f"Key ID: {api_key.key_id}")
else:
    print("No users found. Create one first.")
```

Run: `python bootstrap_key.py`

**3. Test authenticated request:**

```bash
export API_KEY="k8s_rag_..." # From step 2

curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer $API_KEY"

# Should return user info
```

**4. Test chat with auth:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a Pod?",
    "session_id": "auth_test"
  }'

# Should work
```

**5. Test without auth:**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a Pod?",
    "session_id": "no_auth_test"
  }'

# Should return 401 Unauthorized
```

**6. Check audit logs:**

```bash
tail -f logs/app.log | grep audit_event
```

---

## Success Criteria

- [ ] Users can be created via `/auth/users`
- [ ] API keys can be created via `/auth/keys`
- [ ] Chat endpoint requires valid API key
- [ ] Invalid keys return 401
- [ ] All auth events are logged
- [ ] Tests pass: `pytest tests/test_auth.py -v`

---

## Migration from Phase 3

Update cost tracker to include user_id:

**In `src/cost_control/cost_tracker.py`:**

```python
def track_request(
    self,
    session_id: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    embedding_tokens: int = 0,
    user_id: Optional[str] = None  # ← ADD THIS
) -> tuple[float, bool]:
    # ... existing code ...
    
    log.info(
        "cost_tracked",
        user_id=user_id,  # ← ADD THIS
        session_id=session_id,
        model=model,
        cost_usd=cost,
        daily_total=self._daily_cost
    )
```

---

## Next Phase

Once verified:

**→ Proceed to `PHASE_05_OBSERVABILITY.md`**

Advanced observability (tracing, dashboards) builds on auth (need user_id in traces).

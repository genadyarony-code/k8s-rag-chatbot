"""
Unit and integration tests for the authentication system.

Unit tests cover:
- Key generation format and entropy
- SHA-256 hashing and constant-time verification
- In-memory storage CRUD

Integration tests use TestClient with the check_index_health lifespan
mocked out (same pattern as test_api.py) to test the full /auth/* flow.
"""

import json
from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.auth.api_keys import (
    create_api_key,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)
from src.auth.models import APIKey, User
from src.auth.storage import InMemoryAuthStorage


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def storage():
    """Fresh isolated storage instance for each test."""
    return InMemoryAuthStorage()


@pytest.fixture
def test_user(storage):
    """A pre-created user in the isolated storage."""
    user = User(
        user_id="user_test001",
        email="test@example.com",
        name="Test User",
        created_at=datetime.now(),
    )
    storage.create_user(user)
    return user


@pytest.fixture
def client():
    """
    TestClient with the app's lifespan health-check mocked out.
    Mirrors the pattern in tests/test_api.py.
    """
    with patch("src.api.main.check_index_health"):
        from src.api.main import app

        with TestClient(app) as c:
            yield c


# ── Key generation tests ──────────────────────────────────────────────────────


def test_generate_api_key_format():
    key = generate_api_key()
    assert key.startswith("k8s_rag_")
    assert len(key) == len("k8s_rag_") + 32


def test_generate_api_key_is_unique():
    keys = {generate_api_key() for _ in range(50)}
    assert len(keys) == 50  # no collisions in 50 draws


def test_hash_and_verify_round_trip():
    key = generate_api_key()
    stored_hash = hash_api_key(key)
    assert verify_api_key(key, stored_hash)


def test_verify_rejects_wrong_key():
    key = generate_api_key()
    stored_hash = hash_api_key(key)
    other_key = generate_api_key()
    assert not verify_api_key(other_key, stored_hash)


def test_hash_is_deterministic():
    key = generate_api_key()
    assert hash_api_key(key) == hash_api_key(key)


# ── Storage tests ─────────────────────────────────────────────────────────────


def test_storage_create_and_get_user(storage):
    user = User(
        user_id="u1",
        email="alice@example.com",
        name="Alice",
        created_at=datetime.now(),
    )
    storage.create_user(user)
    assert storage.get_user("u1") is user
    assert storage.get_user_by_email("alice@example.com") is user


def test_storage_duplicate_email_raises(storage):
    user = User("u1", "dup@test.com", "User1", datetime.now())
    storage.create_user(user)
    dup = User("u2", "dup@test.com", "User2", datetime.now())
    with pytest.raises(ValueError, match="already registered"):
        storage.create_user(dup)


def test_storage_duplicate_user_id_raises(storage):
    user = User("u1", "a@test.com", "A", datetime.now())
    storage.create_user(user)
    dup = User("u1", "b@test.com", "B", datetime.now())
    with pytest.raises(ValueError, match="already exists"):
        storage.create_user(dup)


def test_storage_revoke_key(storage, test_user):
    api_key = APIKey(
        key_id="key_abc",
        key_hash=hash_api_key("k8s_rag_" + "x" * 32),
        user_id=test_user.user_id,
        name="mykey",
        created_at=datetime.now(),
    )
    storage.create_api_key(api_key)
    assert storage.revoke_key("key_abc")
    assert not storage.get_api_key_by_id("key_abc").is_active


def test_storage_revoke_nonexistent_key(storage):
    assert not storage.revoke_key("does_not_exist")


def test_storage_list_user_keys(storage, test_user):
    for i in range(3):
        storage.create_api_key(APIKey(
            key_id=f"key_{i}",
            key_hash=hash_api_key(f"k8s_rag_{'a' * 32}{i}"),
            user_id=test_user.user_id,
            name=f"key{i}",
            created_at=datetime.now(),
        ))
    keys = storage.list_user_keys(test_user.user_id)
    assert len(keys) == 3


# ── create_api_key integration (uses global storage singleton) ────────────────


def test_create_api_key_returns_valid_key():
    from src.auth.storage import get_auth_storage

    storage = get_auth_storage()
    # Pre-create a user in the global storage (tests run against same process)
    user_id = "unit_test_user_" + generate_api_key()[-8:]
    user = User(user_id=user_id, email=f"{user_id}@test.com", name="Unit", created_at=datetime.now())
    storage.create_user(user)

    full_key, api_key = create_api_key(user_id, "CI Test Key")

    assert full_key.startswith("k8s_rag_")
    assert api_key.user_id == user_id
    assert api_key.is_active
    # Hash stored correctly
    assert verify_api_key(full_key, api_key.key_hash)


# ── HTTP integration tests ────────────────────────────────────────────────────


def test_create_user_endpoint(client):
    response = client.post("/auth/users", json={"email": "http_test@example.com", "name": "HTTP"})
    assert response.status_code == 200
    body = response.json()
    assert "user_id" in body
    assert body["email"] == "http_test@example.com"


def test_create_duplicate_user_returns_400(client):
    client.post("/auth/users", json={"email": "dup2@example.com", "name": "Dup"})
    resp = client.post("/auth/users", json={"email": "dup2@example.com", "name": "Dup"})
    assert resp.status_code == 400


def test_get_me_without_auth_returns_401(client):
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_get_me_with_invalid_key_returns_401(client):
    response = client.get("/auth/me", headers={"Authorization": "Bearer k8s_rag_" + "z" * 32})
    assert response.status_code == 401


def test_full_auth_flow(client):
    """
    End-to-end: create user → bootstrap key → /auth/me → /auth/keys → revoke.
    """
    # 1. Create user
    resp = client.post("/auth/users", json={"email": "flow@example.com", "name": "Flow"})
    assert resp.status_code == 200
    user_id = resp.json()["user_id"]

    # 2. Bootstrap key directly (no auth required to create the first key in dev mode)
    full_key, api_key = create_api_key(user_id, "Bootstrap")

    headers = {"Authorization": f"Bearer {full_key}"}

    # 3. /auth/me returns correct user
    resp = client.get("/auth/me", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["user_id"] == user_id

    # 4. /auth/keys lists the bootstrap key
    resp = client.get("/auth/keys", headers=headers)
    assert resp.status_code == 200
    key_ids = [k["key_id"] for k in resp.json()]
    assert api_key.key_id in key_ids

    # 5. Create a second key via the endpoint itself
    resp = client.post("/auth/keys", json={"name": "Second Key"}, headers=headers)
    assert resp.status_code == 200
    second_key_id = resp.json()["key_id"]

    # 6. Revoke the second key
    resp = client.delete(f"/auth/keys/{second_key_id}", headers=headers)
    assert resp.status_code == 200

    # 7. Verify second key is revoked in storage
    from src.auth.storage import get_auth_storage
    revoked = get_auth_storage().get_api_key_by_id(second_key_id)
    assert revoked is not None
    assert not revoked.is_active


def test_revoke_other_users_key_returns_403(client):
    """A user cannot revoke another user's key."""
    # Create two users and bootstrap a key for user2
    resp1 = client.post("/auth/users", json={"email": "user1_403@example.com", "name": "U1"})
    user1_id = resp1.json()["user_id"]
    key1, _ = create_api_key(user1_id, "U1 Key")

    resp2 = client.post("/auth/users", json={"email": "user2_403@example.com", "name": "U2"})
    user2_id = resp2.json()["user_id"]
    _, key2_obj = create_api_key(user2_id, "U2 Key")

    # User1 tries to revoke User2's key
    resp = client.delete(
        f"/auth/keys/{key2_obj.key_id}",
        headers={"Authorization": f"Bearer {key1}"},
    )
    assert resp.status_code == 403

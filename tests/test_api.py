import pytest
import json
import os
from unittest.mock import patch, MagicMock


@pytest.fixture
def client_with_mock_index(tmp_path):
    """
    יוצר TestClient עם index_meta.json mock —
    כדי שה-lifespan health check יעבור בלי index אמיתי
    """
    # יוצר manifest זמני
    manifest_dir = tmp_path / "data" / "processed"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "index_meta.json"
    manifest.write_text(json.dumps({
        "chunk_schema_version": "v1.0",
        "chunk_count": 100,
        "built_at": "2025-01-01T00:00:00",
        "chroma_ok": True,
        "bm25_ok": True
    }))

    with patch("src.api.main.check_index_health"):  # skip health check
        from fastapi.testclient import TestClient
        from src.api.main import app
        with TestClient(app) as client:
            yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, client_with_mock_index):
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_chroma.return_value.get_collection.return_value = MagicMock()
            with patch("pathlib.Path.exists", return_value=True):
                r = client_with_mock_index.get("/health")
                assert r.status_code == 200

    def test_health_has_required_fields(self, client_with_mock_index):
        with patch("chromadb.PersistentClient"):
            with patch("pathlib.Path.exists", return_value=True):
                r = client_with_mock_index.get("/health")
                data = r.json()
                assert "status" in data
                assert "chroma_ok" in data
                assert "bm25_ok" in data
                assert "feature_flags" in data


class TestResetEndpoint:
    def test_reset_returns_200(self, client_with_mock_index):
        r = client_with_mock_index.post("/reset/test-session-123")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == "test-session-123"


class TestCheckIndexHealth:
    def test_raises_when_manifest_missing(self, tmp_path):
        with patch("src.api.main.os.path.exists", return_value=False):
            from src.api.main import check_index_health
            with pytest.raises(RuntimeError, match="manifest not found"):
                check_index_health()

    def test_raises_when_chroma_not_ok(self, tmp_path):
        bad_meta = json.dumps({"chroma_ok": False, "bm25_ok": True})
        with patch("builtins.open", MagicMock(return_value=MagicMock(
            __enter__=lambda s: s,
            __exit__=MagicMock(return_value=False),
            read=MagicMock(return_value=bad_meta)
        ))):
            with patch("src.api.main.os.path.exists", return_value=True):
                with patch("src.api.main.json.load", return_value={"chroma_ok": False, "bm25_ok": True}):
                    from src.api.main import check_index_health
                    with pytest.raises(RuntimeError, match="inconsistent"):
                        check_index_health()

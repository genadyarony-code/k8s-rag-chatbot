import pytest
from unittest.mock import patch, MagicMock

from src.config.settings import settings


class TestChromaFlag:
    def test_bm25_used_when_chroma_disabled(self):
        with patch.object(settings, "ff_use_chroma", False):
            with patch.object(settings, "ff_use_session_memory", False):
                with patch("src.agent.nodes._bm25_search") as mock_bm25:
                    with patch("src.agent.nodes._chroma_search") as mock_chroma:
                        mock_bm25.return_value = []
                        from src.agent.nodes import retrieve_node
                        retrieve_node({"question": "test", "session_id": "abc"})
                        mock_bm25.assert_called_once()
                        mock_chroma.assert_not_called()

    def test_chroma_used_when_enabled(self):
        with patch.object(settings, "ff_use_chroma", True):
            with patch.object(settings, "ff_use_session_memory", False):
                with patch("src.agent.nodes._chroma_search") as mock_chroma:
                    with patch("src.agent.nodes._bm25_search") as mock_bm25:
                        mock_chroma.return_value = []
                        from src.agent.nodes import retrieve_node
                        retrieve_node({"question": "test", "session_id": "abc"})
                        mock_chroma.assert_called()  # called once or twice (fallback) — but always called
                        mock_bm25.assert_not_called()


class TestOpenAIFlag:
    def test_raw_chunks_when_openai_disabled(self, sample_context):
        with patch.object(settings, "ff_use_openai", False):
            from src.agent.nodes import generate_node
            state = {
                "question": "test", "session_id": "abc",
                "context": sample_context, "history": []
            }
            result = generate_node(state)
            assert "Pod Scheduling" in result["answer"] or "CrashLoopBackOff" in result["answer"]
            assert len(result["sources"]) > 0

    def test_openai_called_when_enabled(self, sample_context, mock_openai_response):
        with patch.object(settings, "ff_use_openai", True):
            with patch.object(settings, "ff_use_session_memory", False):
                with patch("src.agent.nodes._get_openai_client") as mock_get_client:
                    mock_client = MagicMock()
                    mock_client.chat.completions.create.return_value = mock_openai_response
                    mock_get_client.return_value = mock_client

                    from src.agent.nodes import generate_node
                    state = {
                        "question": "test", "session_id": "abc",
                        "context": sample_context, "history": []
                    }
                    result = generate_node(state)
                    assert result["answer"] == "This is a test answer about Kubernetes."


class TestMemoryFlag:
    def test_empty_history_when_memory_disabled(self):
        with patch.object(settings, "ff_use_chroma", True):
            with patch.object(settings, "ff_use_session_memory", False):
                with patch("src.agent.nodes._chroma_search", return_value=[]):
                    from src.agent.nodes import retrieve_node
                    result = retrieve_node({"question": "test", "session_id": "abc"})
                    assert result["history"] == []

    def test_history_loaded_when_memory_enabled(self):
        with patch.object(settings, "ff_use_chroma", True):
            with patch.object(settings, "ff_use_session_memory", True):
                with patch("src.agent.nodes._chroma_search", return_value=[]):
                    with patch("src.agent.nodes.session_memory") as mock_memory:
                        mock_memory.get.return_value = [{"role": "user", "content": "prev"}]
                        from src.agent.nodes import retrieve_node
                        result = retrieve_node({"question": "test", "session_id": "abc"})
                        assert len(result["history"]) == 1

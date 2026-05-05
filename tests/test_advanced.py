"""
Unit tests for Phase 9 advanced features.

Fast tests (no model downloads, no external API calls):
- Model fallback cascade logic
- RRF-based deduplication and ordering
- Semantic cache similarity threshold logic (mocked embeddings)
- Multi-tenant collection naming
- Advanced prompting utilities

Slow tests (require model API access) are not included here — the full
semantic cache behaviour is verified by integration tests or manual testing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.advanced.prompting import (
    add_chain_of_thought,
    build_few_shot_messages,
    detect_complex_query,
    self_consistency_prompt,
)


# ── Prompting utility tests ───────────────────────────────────────────────────


def test_add_chain_of_thought_appends_instruction():
    base = "You are a Kubernetes assistant."
    result = add_chain_of_thought(base)
    assert result.startswith(base)
    assert "step by step" in result.lower()
    assert len(result) > len(base)


def test_build_few_shot_messages_structure():
    msgs = build_few_shot_messages("What is a Service?")
    # Last message is the real question
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "What is a Service?"
    # Pairs of user/assistant before it
    for msg in msgs[:-1]:
        assert msg["role"] in ("user", "assistant")


def test_build_few_shot_custom_examples():
    examples = [{"question": "Q1", "answer": "A1"}]
    msgs = build_few_shot_messages("live question", examples=examples)
    assert msgs[0]["content"] == "Q1"
    assert msgs[1]["content"] == "A1"
    assert msgs[2]["content"] == "live question"


def test_self_consistency_prompt_contains_question():
    prompt = self_consistency_prompt("What is a Pod?", n_samples=2)
    assert "What is a Pod?" in prompt
    assert "Draft 1" in prompt
    assert "Draft 2" in prompt
    assert "consolidated" in prompt.lower()


def test_detect_complex_query_true():
    assert detect_complex_query("What is the difference between Deployments and StatefulSets?")
    assert detect_complex_query("Why does CrashLoopBackOff happen?")
    assert detect_complex_query("How does Kubernetes networking work?")
    assert detect_complex_query("Debug my pod restart issue")


def test_detect_complex_query_false():
    assert not detect_complex_query("List all namespaces")
    assert not detect_complex_query("What is a ConfigMap?")


# ── Multi-tenant collection naming ────────────────────────────────────────────


def test_tenant_collection_naming():
    """Collection names must follow the k8s_docs_{tenant_id} convention."""
    from src.advanced.multi_tenant import TenantManager

    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.get_collection.return_value = mock_collection

    tm = TenantManager(mock_client)
    col = tm.get_collection("acme_corp")

    mock_client.get_collection.assert_called_once_with("k8s_docs_acme_corp")
    assert col is mock_collection


def test_tenant_collection_creates_if_missing():
    """If the collection does not exist, TenantManager must create it."""
    from src.advanced.multi_tenant import TenantManager

    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Not found")
    mock_collection = MagicMock()
    mock_client.create_collection.return_value = mock_collection

    tm = TenantManager(mock_client)
    col = tm.get_collection("new_tenant")

    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args
    assert "k8s_docs_new_tenant" in str(call_kwargs)
    assert col is mock_collection


def test_tenant_collection_cached_on_second_call():
    """TenantManager must not hit ChromaDB twice for the same tenant."""
    from src.advanced.multi_tenant import TenantManager

    mock_client = MagicMock()
    mock_client.get_collection.return_value = MagicMock()

    tm = TenantManager(mock_client)
    tm.get_collection("tenant_a")
    tm.get_collection("tenant_a")

    assert mock_client.get_collection.call_count == 1


# ── Model fallback logic ──────────────────────────────────────────────────────


def test_fallback_handler_returns_first_model_on_success():
    """When the primary model succeeds, no fallback should be tried."""
    from src.advanced.model_fallback import ModelFallbackHandler

    with patch("src.advanced.model_fallback.OpenAI") as MockOAI, \
         patch("src.advanced.model_fallback.settings") as mock_settings:
        mock_settings.openai_api_key = "test"
        mock_settings.anthropic_api_key = ""

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Answer text"
        mock_resp.usage.prompt_tokens = 100
        mock_resp.usage.completion_tokens = 50
        mock_resp.usage.total_tokens = 150
        MockOAI.return_value.chat.completions.create.return_value = mock_resp

        handler = ModelFallbackHandler()
        result = handler.call_with_fallback(
            messages=[{"role": "user", "content": "test"}],
            session_id="sess1",
        )

    assert result["content"] == "Answer text"
    assert result["provider"] == "openai"
    assert result["total_tokens"] == 150


def test_fallback_handler_falls_to_second_on_rate_limit():
    """A RateLimitError on model 1 must cause a retry with model 2."""
    from openai import RateLimitError
    from src.advanced.model_fallback import ModelFallbackHandler

    responses: list = []

    def side_effect(**kwargs):
        model = kwargs["model"]
        if model == "gpt-4o-mini":
            raise RateLimitError("rate limited", response=MagicMock(status_code=429), body={})
        # gpt-3.5-turbo succeeds
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Fallback answer"
        mock_resp.usage.prompt_tokens = 80
        mock_resp.usage.completion_tokens = 40
        mock_resp.usage.total_tokens = 120
        return mock_resp

    with patch("src.advanced.model_fallback.OpenAI") as MockOAI, \
         patch("src.advanced.model_fallback.settings") as mock_settings:
        mock_settings.openai_api_key = "test"
        mock_settings.anthropic_api_key = ""
        MockOAI.return_value.chat.completions.create.side_effect = side_effect

        handler = ModelFallbackHandler()
        result = handler.call_with_fallback(
            messages=[{"role": "user", "content": "test"}],
            session_id="sess2",
        )

    assert result["content"] == "Fallback answer"
    assert result["model"] == "gpt-3.5-turbo"


def test_fallback_handler_raises_when_all_fail():
    """RuntimeError must be raised when every model in the cascade fails."""
    from openai import RateLimitError
    from src.advanced.model_fallback import ModelFallbackHandler

    def always_fail(**kwargs):
        raise RateLimitError("always fail", response=MagicMock(status_code=429), body={})

    with patch("src.advanced.model_fallback.OpenAI") as MockOAI, \
         patch("src.advanced.model_fallback.settings") as mock_settings:
        mock_settings.openai_api_key = "test"
        mock_settings.anthropic_api_key = ""
        MockOAI.return_value.chat.completions.create.side_effect = always_fail

        handler = ModelFallbackHandler()
        with pytest.raises(RuntimeError, match="All models in cascade failed"):
            handler.call_with_fallback(
                messages=[{"role": "user", "content": "test"}],
                session_id="sess3",
            )


# ── Semantic cache logic (mocked embeddings) ──────────────────────────────────


def test_semantic_cache_miss_on_empty():
    """An empty cache must always return None."""
    try:
        import faiss  # noqa: F401
        import numpy as np
    except ImportError:
        pytest.skip("faiss-cpu not installed")

    from src.advanced.semantic_cache import SemanticCache

    fake_vec = np.ones(1536, dtype=np.float32)
    fake_vec /= np.linalg.norm(fake_vec)

    with patch("src.advanced.semantic_cache.OpenAI") as MockOAI, \
         patch("src.advanced.semantic_cache.settings") as mock_settings:
        mock_settings.openai_api_key = "test"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_embed_resp = MagicMock()
        mock_embed_resp.data[0].embedding = fake_vec.tolist()
        MockOAI.return_value.embeddings.create.return_value = mock_embed_resp

        cache = SemanticCache(similarity_threshold=0.95)
        assert cache.get("What is a Pod?") is None
        assert cache.size == 0


def test_semantic_cache_hit_above_threshold():
    """Identical question must return a cache hit (similarity = 1.0)."""
    try:
        import faiss  # noqa: F401
        import numpy as np
    except ImportError:
        pytest.skip("faiss-cpu not installed")

    from src.advanced.semantic_cache import SemanticCache

    fake_vec = np.ones(1536, dtype=np.float32)
    fake_vec /= np.linalg.norm(fake_vec)

    with patch("src.advanced.semantic_cache.OpenAI") as MockOAI, \
         patch("src.advanced.semantic_cache.settings") as mock_settings:
        mock_settings.openai_api_key = "test"
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_embed_resp = MagicMock()
        mock_embed_resp.data[0].embedding = fake_vec.tolist()
        MockOAI.return_value.embeddings.create.return_value = mock_embed_resp

        cache = SemanticCache(similarity_threshold=0.95)
        cache.set("What is a Pod?", "A Pod is ...", ["k8s.pdf"])
        result = cache.get("What is a Pod?")

    assert result is not None
    assert result["answer"] == "A Pod is ..."
    assert result["sources"] == ["k8s.pdf"]

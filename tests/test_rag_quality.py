"""
Unit tests for RAG quality components.

Fast tests (no model downloads):
- RRF fusion correctness
- should_decompose heuristic
- decompose_query passthrough for simple queries

Slow tests (require model download on first run, ~80 MB each):
- ChunkReranker (FlashRank ms-marco-MiniLM)
- CitationValidator (sentence-transformers all-MiniLM-L6-v2)

Run slow tests explicitly:  pytest tests/test_rag_quality.py -m slow
Run all fast tests:          pytest tests/test_rag_quality.py -m "not slow"
"""

import pytest

from src.rag.hybrid_search import reciprocal_rank_fusion
from src.rag.query_decomposition import should_decompose, decompose_query


# ── RRF Fusion Tests (fast, no model) ────────────────────────────────────────


def test_rrf_doc_appearing_in_both_lists_ranks_highest():
    """Document in both lists should outscore documents in only one list."""
    list1 = [
        {"content": "alpha", "score": 0.9},
        {"content": "beta", "score": 0.8},
    ]
    list2 = [
        {"content": "beta", "score": 0.95},  # "beta" appears in both
        {"content": "gamma", "score": 0.7},
    ]
    fused = reciprocal_rank_fusion([list1, list2])
    assert fused[0]["content"] == "beta"


def test_rrf_score_key_present():
    docs = [{"content": "x", "score": 1.0}]
    fused = reciprocal_rank_fusion([docs])
    assert "rrf_score" in fused[0]
    assert fused[0]["rrf_score"] > 0


def test_rrf_empty_list_is_safe():
    fused = reciprocal_rank_fusion([[]])
    assert fused == []


def test_rrf_single_list_preserves_original_order():
    docs = [
        {"content": "first", "score": 0.9},
        {"content": "second", "score": 0.7},
    ]
    fused = reciprocal_rank_fusion([docs])
    assert fused[0]["content"] == "first"
    assert fused[1]["content"] == "second"


def test_rrf_deduplicates_identical_content():
    docs = [{"content": "dup", "score": 0.9}]
    fused = reciprocal_rank_fusion([docs, docs])
    assert len([d for d in fused if d["content"] == "dup"]) == 1


def test_rrf_three_lists_union():
    a = [{"content": "a", "score": 1.0}]
    b = [{"content": "b", "score": 1.0}]
    c = [{"content": "c", "score": 1.0}]
    fused = reciprocal_rank_fusion([a, b, c])
    contents = {d["content"] for d in fused}
    assert contents == {"a", "b", "c"}


# ── Query Decomposition Heuristic Tests (fast) ───────────────────────────────


def test_should_decompose_comparison_queries():
    assert should_decompose("What is the difference between X and Y?")
    assert should_decompose("Compare Deployments vs StatefulSets")
    assert should_decompose("X versus Y: which is better?")


def test_should_decompose_multiple_question_marks():
    assert should_decompose("What is X? And what is Y?")


def test_should_decompose_long_and_query():
    long_query = "How do I configure resource limits and set up autoscaling for pods?"
    assert should_decompose(long_query)


def test_should_not_decompose_simple_queries():
    assert not should_decompose("What is a Pod?")
    assert not should_decompose("How do I debug CrashLoopBackOff?")
    assert not should_decompose("List Kubernetes namespaces")


def test_decompose_returns_original_for_simple_query():
    """Simple queries must pass through unchanged without an LLM call."""
    query = "What is a Kubernetes Pod?"
    result = decompose_query(query)
    assert len(result) == 1
    assert result[0] == query


# ── Reranker Tests (slow — downloads FlashRank model ~20 MB) ─────────────────


@pytest.mark.slow
def test_reranker_returns_correct_top_k():
    from src.rag.reranker import ChunkReranker

    ranker = ChunkReranker()
    chunks = [
        {"content": "Kubernetes is a container orchestration platform.", "score": 0.9},
        {"content": "The weather in Paris is typically mild in spring.", "score": 0.85},
        {"content": "Pods are the smallest deployable units in Kubernetes.", "score": 0.8},
        {"content": "StatefulSets manage stateful applications in Kubernetes.", "score": 0.7},
    ]
    reranked = ranker.rerank("What is Kubernetes?", chunks, top_k=2)

    assert len(reranked) == 2
    # Both top results should mention Kubernetes (not the weather chunk)
    assert all("Kubernetes" in c["content"] or "Pods" in c["content"] for c in reranked)


@pytest.mark.slow
def test_reranker_adds_rerank_score():
    from src.rag.reranker import ChunkReranker

    ranker = ChunkReranker()
    chunks = [{"content": "Kubernetes orchestrates containers.", "score": 0.5}]
    reranked = ranker.rerank("What is Kubernetes?", chunks, top_k=1)

    assert len(reranked) == 1
    assert "rerank_score" in reranked[0]
    assert "original_score" in reranked[0]


@pytest.mark.slow
def test_reranker_empty_input():
    from src.rag.reranker import ChunkReranker

    ranker = ChunkReranker()
    assert ranker.rerank("any query", [], top_k=5) == []


# ── Citation Validator Tests (slow — downloads ~80 MB model) ─────────────────


@pytest.mark.slow
def test_citation_validator_grounded_answer():
    from src.rag.citation_validator import CitationValidator

    v = CitationValidator()
    chunks = [
        {"content": "Kubernetes is a container orchestration platform developed by Google."},
        {"content": "Pods are the smallest deployable units in Kubernetes."},
    ]
    answer = "Kubernetes is an orchestration platform. Pods are deployable units."
    is_valid, unsupported = v.validate(answer, chunks, threshold=0.5)
    assert is_valid
    assert len(unsupported) == 0


@pytest.mark.slow
def test_citation_validator_ungrounded_answer():
    from src.rag.citation_validator import CitationValidator

    v = CitationValidator()
    chunks = [
        {"content": "Kubernetes manages containerised workloads."},
    ]
    answer = "Kubernetes was invented by aliens on planet Zorbax in the year 2087."
    is_valid, unsupported = v.validate(answer, chunks, threshold=0.5)
    assert not is_valid
    assert len(unsupported) > 0


@pytest.mark.slow
def test_citation_validator_empty_chunks():
    from src.rag.citation_validator import CitationValidator

    v = CitationValidator()
    is_valid, unsupported = v.validate("Any answer.", [], threshold=0.5)
    assert not is_valid

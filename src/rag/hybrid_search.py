"""
Hybrid search using Reciprocal Rank Fusion (RRF).

Strategy:
1. Run ChromaDB (dense vector) and BM25 (sparse keyword) searches in parallel.
2. Fuse their ranked lists using the RRF algorithm.
3. Return the merged, deduplicated list for downstream reranking.

Why hybrid?
- Vector search excels at semantic / paraphrase queries.
- BM25 excels at exact keyword and entity queries.
- RRF combines both without requiring score normalisation — each retriever
  only contributes rank positions, not raw scores.

Why k=60?
The RRF constant k was set to 60 in the original Cormack et al. paper.
Values between 40 and 80 produce nearly identical results in practice; 60
is a safe default that avoids over-weighting top-1 results.
"""

from typing import Callable

from src.observability.logging_config import get_logger
from src.observability.metrics import hybrid_search_fusion_score

log = get_logger(__name__)


def reciprocal_rank_fusion(
    results_list: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Fuse multiple ranked lists using the RRF algorithm.

    RRF formula: score(doc) = Σ  1 / (k + rank_i)
                              all lists i where doc appears

    Args:
        results_list: Each element is a ranked list of chunk dicts.
                      Chunks must contain at least "content".
        k:            RRF constant (default: 60).

    Returns:
        Merged list sorted by RRF score (descending). Each chunk gains
        an "rrf_score" key.
    """
    doc_scores: dict[int, dict] = {}

    for result_list in results_list:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = hash(doc["content"])
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "rrf_score": 0.0}
            doc_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)

    fused = sorted(doc_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    results: list[dict] = []
    for item in fused:
        chunk = dict(item["doc"])
        chunk["rrf_score"] = item["rrf_score"]
        chunk["score"] = item["rrf_score"]
        results.append(chunk)

    if results:
        hybrid_search_fusion_score.observe(results[0]["rrf_score"])

    log.info(
        "rrf_fusion_completed",
        input_lists=len(results_list),
        unique_docs=len(results),
        top_rrf_score=round(results[0]["rrf_score"], 6) if results else 0,
    )

    return results


def hybrid_search(
    query: str,
    chroma_search_fn: Callable,
    bm25_search_fn: Callable,
    top_k: int = 20,
) -> list[dict]:
    """
    Combine dense and sparse retrieval via RRF.

    Args:
        query:           User question.
        chroma_search_fn: Callable(query, top_k) → list[dict]
        bm25_search_fn:   Callable(query, top_k) → list[dict]
        top_k:           Number of candidates to pull from each retriever.

    Returns:
        RRF-fused results (may be up to 2 × top_k unique docs before reranking).
    """
    vector_results = chroma_search_fn(query, top_k)
    bm25_results = bm25_search_fn(query, top_k)

    log.info(
        "hybrid_search_results",
        vector_count=len(vector_results),
        bm25_count=len(bm25_results),
    )

    return reciprocal_rank_fusion([vector_results, bm25_results])

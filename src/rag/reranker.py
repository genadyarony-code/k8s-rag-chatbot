"""
Reranking layer using FlashRank.

Why rerank?
- Vector search optimises for semantic similarity in embedding space.
- Semantic similarity ≠ relevance for the specific query intent.
- A cross-encoder reranker scores each (query, document) pair jointly,
  giving far more accurate relevance judgements.

Strategy:
1. Vector / hybrid search retrieves top 20 candidates (wide net).
2. Reranker scores each candidate against the query.
3. Return the top-K by rerank score.

Performance:
- FlashRank's ms-marco-MiniLM-L-12-v2 runs on CPU in ~10 ms for 20 docs.
- No GPU required.
"""

import time
from typing import Optional

from flashrank import RerankRequest, Ranker

from src.observability.logging_config import get_logger
from src.observability.metrics import reranking_improvement, retrieval_latency_seconds

log = get_logger(__name__)


class ChunkReranker:
    """
    Thin wrapper around a FlashRank cross-encoder model.

    The model is loaded once on first instantiation (lazy singleton pattern).
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        self.model_name = model_name
        self.ranker = Ranker(model_name=model_name)
        log.info("reranker_initialized", model=model_name)

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        Rerank `chunks` by relevance to `query` and return the top-K.

        Each input chunk must have at least a "content" key.
        Each returned chunk gains two extra keys:
        - rerank_score:    cross-encoder score (replaces vector/RRF score)
        - original_score:  the score the chunk carried before reranking

        Args:
            query:   User question.
            chunks:  Candidate chunks from vector / hybrid search.
            top_k:   Number of top results to return after reranking.

        Returns:
            At most top_k chunks, sorted by rerank score (descending).
        """
        if not chunks:
            return []

        start_time = time.time()

        # FlashRank preserves all dict fields; we just need to ensure
        # the query-facing text is in the "text" key.
        passages = [
            {"id": i, "text": chunk["content"], **chunk}
            for i, chunk in enumerate(chunks)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        reranked: list[dict] = []
        for result in results[:top_k]:
            # FlashRank returns the original passage dict with "score" added
            chunk = {k: v for k, v in result.items() if k not in ("id", "text")}
            chunk["rerank_score"] = result["score"]
            chunk["original_score"] = chunk.get("score", 0.0)
            chunk["score"] = result["score"]
            reranked.append(chunk)

        elapsed = time.time() - start_time

        # Record score improvement for the top result (original → reranked)
        if reranked and reranked[0].get("original_score") is not None:
            improvement = reranked[0]["rerank_score"] - reranked[0]["original_score"]
            reranking_improvement.observe(improvement)

        log.info(
            "reranking_completed",
            model=self.model_name,
            input_chunks=len(chunks),
            output_chunks=len(reranked),
            latency_seconds=round(elapsed, 3),
        )

        return reranked


# ── Lazy singleton ────────────────────────────────────────────────────────────

_reranker: Optional[ChunkReranker] = None


def get_reranker() -> ChunkReranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = ChunkReranker()
    return _reranker

"""
Semantic caching using embedding similarity + FAISS.

Strategy:
1. On every query, embed the question with text-embedding-3-small.
2. Search the FAISS index for the nearest cached question.
3. If cosine similarity ≥ threshold (default 0.95), return the cached answer.
4. On cache miss, the caller generates normally and calls cache.set().

Why 0.95?
The threshold is intentionally high. A similarity of 0.95 means the two
questions are semantically nearly identical — "What is a Pod?" vs
"Can you explain what a Kubernetes Pod is?". At lower thresholds we risk
returning incorrect answers for related but distinct questions.

Cost/latency profile:
- Cache hit:  one embedding call (~5 ms, ~$0.000002) → no LLM call
- Cache miss: one embedding call + normal LLM call

Storage:
- FAISS flat index (in-memory): stores unit-normalised 1536-d embeddings.
- Answer storage: Redis (production) or dict (dev/test).

Feature flag: FF_USE_SEMANTIC_CACHE (default False).
Requires: faiss-cpu — disabled by default to avoid hard dependency.
"""

import hashlib
import json
from typing import Optional

from openai import OpenAI

from src.config.settings import settings
from src.observability.logging_config import get_logger
from src.observability.metrics import cache_hit_rate, cache_hits, cache_misses

log = get_logger(__name__)

_EMBED_DIM = 1536   # text-embedding-3-small output dimension
_CACHE_TTL = 86_400 * 7  # 7-day TTL for Redis entries


class SemanticCache:
    """
    Embedding-based answer cache backed by FAISS + Redis (or in-memory).

    Thread note: FAISS IndexFlatIP is not thread-safe for concurrent writes.
    For production use behind a single-worker API server this is acceptable.
    A RWLock should be added if using multi-process workers.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        similarity_threshold: float = 0.95,
    ) -> None:
        import numpy as np
        import faiss  # type: ignore[import]

        self._np = np
        self._faiss = faiss

        self._openai = OpenAI(api_key=settings.openai_api_key)
        self._threshold = similarity_threshold

        # FAISS flat inner-product index (operates on unit vectors → cosine sim)
        self._index = faiss.IndexFlatIP(_EMBED_DIM)
        self._hash_to_idx: dict[str, int] = {}
        self._idx_to_hash: dict[int, str] = {}

        # Answer storage
        if redis_url:
            try:
                import redis as _redis  # type: ignore[import]
                self._redis = _redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                self._store = "redis"
                log.info("semantic_cache_initialized", storage="redis")
            except Exception as exc:
                log.warning("redis_unavailable_for_cache", error=str(exc))
                self._redis = None
                self._store = "memory"
                self._mem: dict[str, dict] = {}
        else:
            self._redis = None
            self._store = "memory"
            self._mem = {}
            log.info("semantic_cache_initialized", storage="memory")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> "np.ndarray":
        response = self._openai.embeddings.create(
            input=text, model=settings.embedding_model
        )
        vec = self._np.array(response.data[0].embedding, dtype=self._np.float32)
        vec /= self._np.linalg.norm(vec)   # unit-normalise for cosine sim
        return vec

    @staticmethod
    def _hash(question: str) -> str:
        return hashlib.sha256(question.encode()).hexdigest()[:16]

    def _load(self, question_hash: str) -> Optional[dict]:
        if self._store == "redis":
            raw = self._redis.get(f"scache:{question_hash}")
            return json.loads(raw) if raw else None
        return self._mem.get(question_hash)

    def _save(self, question_hash: str, data: dict) -> None:
        if self._store == "redis":
            self._redis.set(f"scache:{question_hash}", json.dumps(data), ex=_CACHE_TTL)
        else:
            self._mem[question_hash] = data

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of entries currently cached."""
        return self._index.ntotal

    def get(self, question: str) -> Optional[dict]:
        """
        Return a cached answer dict or None on miss.

        The dict has keys: question, answer, sources, metadata.
        """
        if self._index.ntotal == 0:
            cache_misses.labels(cache_type="semantic").inc()
            return None

        query_vec = self._embed(question).reshape(1, -1)
        distances, indices = self._index.search(query_vec, k=1)
        similarity = float(distances[0][0])

        if similarity < self._threshold:
            cache_misses.labels(cache_type="semantic").inc()
            log.info(
                "cache_miss",
                question_preview=question[:60],
                best_similarity=round(similarity, 4),
                threshold=self._threshold,
            )
            self._update_hit_rate()
            return None

        idx = int(indices[0][0])
        qhash = self._idx_to_hash.get(idx)
        if not qhash:
            cache_misses.labels(cache_type="semantic").inc()
            return None

        result = self._load(qhash)
        if result:
            cache_hits.labels(cache_type="semantic").inc()
            log.info(
                "cache_hit",
                question_preview=question[:60],
                similarity=round(similarity, 4),
            )
        else:
            cache_misses.labels(cache_type="semantic").inc()

        self._update_hit_rate()
        return result

    def set(
        self,
        question: str,
        answer: str,
        sources: list[str],
        metadata: Optional[dict] = None,
    ) -> None:
        """Cache a question-answer pair with its sources."""
        qhash = self._hash(question)
        if qhash in self._hash_to_idx:
            return  # Already cached

        vec = self._embed(question)
        idx = self._index.ntotal
        self._index.add(vec.reshape(1, -1))
        self._hash_to_idx[qhash] = idx
        self._idx_to_hash[idx] = qhash

        self._save(qhash, {
            "question": question,
            "answer": answer,
            "sources": sources,
            "metadata": metadata or {},
        })

        log.info(
            "cache_set",
            question_preview=question[:60],
            cache_size=self._index.ntotal,
        )

    def _update_hit_rate(self) -> None:
        """Recompute the rolling hit-rate gauge."""
        hits = cache_hits.labels(cache_type="semantic")._value.get()
        misses = cache_misses.labels(cache_type="semantic")._value.get()
        total = hits + misses
        if total > 0:
            cache_hit_rate.labels(cache_type="semantic").set(hits / total)


# ── Lazy singleton ────────────────────────────────────────────────────────────

_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create the global semantic cache."""
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            redis_url=settings.redis_url or None,
            similarity_threshold=settings.semantic_cache_threshold,
        )
    return _cache

# PHASE 9: Advanced Features

> **Week:** 7-8  
> **Priority:** P3 (Nice-to-Have)  
> **Duration:** 4-5 days  
> **Dependencies:** PHASE 1-8 completed

---

## Objective

Implement model fallback cascade, semantic caching, multi-tenant isolation, and advanced prompt engineering techniques.

**Why this matters:**  
These features improve reliability, reduce costs, and enable production scaling.

---

## Pre-Flight Checklist

- [ ] PHASE 1-8 completed
- [ ] All core features working
- [ ] All tests passing

---

## Task 1: Install Dependencies

**Add to `requirements.txt`:**

```
# For semantic caching
redis==5.0.1  # Already added
faiss-cpu==1.7.4  # Vector similarity for cache

# For advanced prompting
# No new deps needed
```

**Install:**
```bash
pip install faiss-cpu==1.7.4
```

---

## Task 2: Implement Model Fallback Cascade

**Create:** `src/advanced/__init__.py`

```python
# Empty init
```

**Create:** `src/advanced/model_fallback.py`

```python
"""
Model fallback cascade for resilience.

Cascade:
1. gpt-4o-mini (primary, fast, cheap)
2. gpt-3.5-turbo (fallback 1, faster, cheaper)
3. claude-3-haiku (fallback 2, different provider)

Triggers:
- Rate limit errors (429)
- Service unavailable (503)
- Circuit breaker open
"""

from typing import Callable, Any, Optional
from openai import OpenAI, RateLimitError
from anthropic import Anthropic
import time

from src.config.settings import settings
from src.observability.logging_config import get_logger
from src.observability.metrics import chat_tokens_total

log = get_logger(__name__)


class ModelFallbackHandler:
    """
    Handle model fallback with cascade strategy.
    """
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
        # Optional: Anthropic API key for Claude fallback
        self.anthropic_client = None
        if hasattr(settings, 'anthropic_api_key') and settings.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=settings.anthropic_api_key)
        
        # Fallback order
        self.cascade = [
            ("gpt-4o-mini", self._call_openai),
            ("gpt-3.5-turbo", self._call_openai),
        ]
        
        if self.anthropic_client:
            self.cascade.append(("claude-3-haiku-20240307", self._call_anthropic))
        
        log.info(
            "model_fallback_initialized",
            cascade=[m[0] for m in self.cascade]
        )
    
    def _call_openai(self, model: str, messages: list, **kwargs) -> dict:
        """Call OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": model,
            "tokens": response.usage.total_tokens,
            "provider": "openai"
        }
    
    def _call_anthropic(self, model: str, messages: list, **kwargs) -> dict:
        """Call Anthropic API (Claude)."""
        # Convert OpenAI message format to Anthropic format
        system = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = self.anthropic_client.messages.create(
            model=model,
            max_tokens=kwargs.get("max_tokens", 1000),
            system=system,
            messages=anthropic_messages
        )
        
        return {
            "content": response.content[0].text,
            "model": model,
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
            "provider": "anthropic"
        }
    
    def call_with_fallback(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        session_id: str = None
    ) -> dict:
        """
        Call LLM with fallback cascade.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max completion tokens
            session_id: Session identifier for logging
            
        Returns:
            Response dict with content, model, tokens, provider
        """
        
        last_error = None
        
        for model, call_fn in self.cascade:
            try:
                log.info(
                    "model_fallback_attempting",
                    model=model,
                    session_id=session_id
                )
                
                result = call_fn(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Track metrics
                chat_tokens_total.labels(model=model).inc(result["tokens"])
                
                log.info(
                    "model_fallback_success",
                    model=model,
                    provider=result["provider"],
                    tokens=result["tokens"],
                    session_id=session_id
                )
                
                return result
            
            except RateLimitError as e:
                log.warning(
                    "model_rate_limited",
                    model=model,
                    session_id=session_id
                )
                last_error = e
                continue
            
            except Exception as e:
                log.error(
                    "model_call_failed",
                    model=model,
                    error=str(e),
                    session_id=session_id
                )
                last_error = e
                continue
        
        # All models failed
        log.error(
            "model_fallback_exhausted",
            session_id=session_id,
            last_error=str(last_error)
        )
        
        raise Exception(f"All models failed. Last error: {last_error}")


# Global handler
_handler = None


def get_fallback_handler() -> ModelFallbackHandler:
    """Get or create the global fallback handler."""
    global _handler
    if _handler is None:
        _handler = ModelFallbackHandler()
    return _handler
```

---

## Task 3: Implement Semantic Caching

**Create:** `src/advanced/semantic_cache.py`

```python
"""
Semantic caching using vector similarity.

Strategy:
1. Embed incoming question
2. Search cache for similar questions (cosine similarity > 0.95)
3. If hit, return cached answer
4. If miss, generate answer and cache it

Benefits:
- Reduce latency (cache hit: ~10ms vs generation: ~2s)
- Reduce costs (no LLM call needed)
- Improve consistency (same question → same answer)
"""

import json
import hashlib
from typing import Optional, Tuple
import numpy as np
import faiss

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from openai import OpenAI

from src.config.settings import settings
from src.observability.logging_config import get_logger
from src.observability.metrics import Counter

log = get_logger(__name__)


# Metrics
cache_hits = Counter("cache_hits_total", "Cache hit count", ["cache_type"])
cache_misses = Counter("cache_misses_total", "Cache miss count", ["cache_type"])


class SemanticCache:
    """
    Semantic cache using embeddings + FAISS.
    
    Storage:
    - Embeddings: FAISS index (in-memory)
    - Answers: Redis (production) or dict (development)
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        similarity_threshold: float = 0.95,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize semantic cache.
        
        Args:
            redis_url: Redis URL for answer storage
            similarity_threshold: Min cosine similarity for cache hit
            embedding_model: OpenAI embedding model
        """
        
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
        # FAISS index (384 dims for text-embedding-3-small)
        self.dimension = 1536  # text-embedding-3-small dimension
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine)
        
        # Question hash -> embedding index mapping
        self.hash_to_idx = {}
        self.idx_to_hash = {}
        
        # Answer storage
        if redis_url and REDIS_AVAILABLE:
            self.redis = redis.from_url(redis_url)
            self.storage_type = "redis"
        else:
            self.redis = None
            self.storage_type = "memory"
            self._memory_store = {}
        
        log.info(
            "semantic_cache_initialized",
            storage=self.storage_type,
            threshold=similarity_threshold,
            embedding_model=embedding_model
        )
    
    def _embed(self, text: str) -> np.ndarray:
        """Embed text using OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        
        # Normalize for cosine similarity
        embedding /= np.linalg.norm(embedding)
        
        return embedding
    
    def _hash_question(self, question: str) -> str:
        """Hash question for storage key."""
        return hashlib.sha256(question.encode()).hexdigest()[:16]
    
    def get(self, question: str) -> Optional[dict]:
        """
        Get cached answer for question.
        
        Args:
            question: User question
            
        Returns:
            Cached result dict or None
        """
        
        # Embed question
        query_embedding = self._embed(question)
        
        if self.index.ntotal == 0:
            # Empty cache
            cache_misses.labels(cache_type="semantic").inc()
            return None
        
        # Search FAISS
        query_vector = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_vector, k=1)
        
        similarity = float(distances[0][0])
        
        if similarity < self.similarity_threshold:
            # No similar question found
            cache_misses.labels(cache_type="semantic").inc()
            log.info(
                "cache_miss",
                question_preview=question[:50],
                best_similarity=similarity,
                threshold=self.similarity_threshold
            )
            return None
        
        # Cache hit!
        idx = int(indices[0][0])
        question_hash = self.idx_to_hash[idx]
        
        # Retrieve answer
        if self.storage_type == "redis":
            cached_data = self.redis.get(f"cache:{question_hash}")
            if cached_data:
                result = json.loads(cached_data)
            else:
                return None
        else:
            result = self._memory_store.get(question_hash)
        
        if result:
            cache_hits.labels(cache_type="semantic").inc()
            log.info(
                "cache_hit",
                question_preview=question[:50],
                similarity=similarity
            )
        
        return result
    
    def set(
        self,
        question: str,
        answer: str,
        sources: list,
        metadata: dict = None
    ):
        """
        Cache a question-answer pair.
        
        Args:
            question: User question
            answer: Generated answer
            sources: Sources used
            metadata: Optional metadata
        """
        
        question_hash = self._hash_question(question)
        
        # Check if already cached
        if question_hash in self.hash_to_idx:
            log.info("cache_update_skipped", reason="already_cached")
            return
        
        # Embed and add to FAISS
        embedding = self._embed(question)
        idx = self.index.ntotal
        self.index.add(embedding.reshape(1, -1))
        
        # Update mappings
        self.hash_to_idx[question_hash] = idx
        self.idx_to_hash[idx] = question_hash
        
        # Store answer
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "metadata": metadata or {}
        }
        
        if self.storage_type == "redis":
            self.redis.set(
                f"cache:{question_hash}",
                json.dumps(result),
                ex=86400 * 7  # 7 days TTL
            )
        else:
            self._memory_store[question_hash] = result
        
        log.info(
            "cache_set",
            question_preview=question[:50],
            cache_size=self.index.ntotal
        )


# Global cache
_cache = None


def get_semantic_cache() -> SemanticCache:
    """Get or create the global semantic cache."""
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            redis_url=None,  # Set from settings in production
            similarity_threshold=0.95
        )
    return _cache
```

---

## Task 4: Implement Multi-Tenant Isolation

**Create:** `src/advanced/multi_tenant.py`

```python
"""
Multi-tenant isolation for SaaS deployments.

Isolation strategy:
- Tenant ID in all operations
- Separate ChromaDB collections per tenant
- Separate cost budgets per tenant
- Tenant-level rate limits

For now, this is a skeleton. Full implementation requires:
- Tenant management API
- Dynamic collection creation
- Tenant-scoped queries
"""

from typing import Optional
import chromadb

from src.config.settings import settings
from src.observability.logging_config import get_logger

log = get_logger(__name__)


class TenantManager:
    """
    Manage multi-tenant ChromaDB collections.
    """
    
    def __init__(self, chroma_client: chromadb.PersistentClient):
        self.client = chroma_client
        log.info("tenant_manager_initialized")
    
    def get_collection(self, tenant_id: str) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection for a tenant.
        
        Args:
            tenant_id: Tenant identifier
            
        Returns:
            ChromaDB collection
        """
        
        collection_name = f"k8s_docs_{tenant_id}"
        
        try:
            collection = self.client.get_collection(collection_name)
            log.info("tenant_collection_retrieved", tenant_id=tenant_id)
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"tenant_id": tenant_id}
            )
            log.info("tenant_collection_created", tenant_id=tenant_id)
        
        return collection
    
    def isolate_query(
        self,
        tenant_id: str,
        query_embedding: list,
        n_results: int = 5
    ) -> list:
        """
        Run a tenant-isolated query.
        
        Args:
            tenant_id: Tenant identifier
            query_embedding: Query embedding
            n_results: Number of results
            
        Returns:
            Query results
        """
        
        collection = self.get_collection(tenant_id)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        log.info(
            "tenant_query_executed",
            tenant_id=tenant_id,
            results_count=len(results["documents"][0])
        )
        
        return results


# Note: Full multi-tenancy also requires:
# - Tenant-scoped cost budgets
# - Tenant-scoped rate limits
# - Tenant management API (create/update/delete tenants)
# - Tenant authentication (separate API keys per tenant)
```

---

## Task 5: Advanced Prompt Engineering

**Create:** `src/advanced/prompting.py`

```python
"""
Advanced prompt engineering techniques.

Techniques:
- Chain-of-thought (CoT) prompting
- Few-shot examples
- Self-consistency (multiple samples)
- Reflection (critique and revise)
"""

from typing import List, Dict

from src.observability.logging_config import get_logger

log = get_logger(__name__)


# ── Chain-of-Thought ────────────────────────────────────────────────────

COT_INSTRUCTION = """Before answering, think through the problem step by step:

1. What is the user really asking?
2. What concepts or components are involved?
3. What's the relationship between them?
4. What common pitfalls or misconceptions should I address?

Then provide a clear, structured answer."""


def add_chain_of_thought(system_prompt: str) -> str:
    """Add CoT instruction to system prompt."""
    return f"{system_prompt}\n\n{COT_INSTRUCTION}"


# ── Few-Shot Examples ───────────────────────────────────────────────────

K8S_FEW_SHOT_EXAMPLES = [
    {
        "question": "What is a Pod?",
        "answer": "A Pod is the smallest deployable unit in Kubernetes. It represents a single instance of a running process and can contain one or more containers that share storage and network resources. Pods are ephemeral - when they die, they're not resurrected."
    },
    {
        "question": "How do I debug a CrashLoopBackOff?",
        "answer": "CrashLoopBackOff means your pod is crashing repeatedly. Debug steps:\n1. Check logs: `kubectl logs <pod> --previous`\n2. Check the exit code in pod status\n3. Verify the image exists and is accessible\n4. Check resource limits - OOMKilled is common\n5. Review the container command/args"
    }
]


def build_few_shot_messages(question: str, examples: List[Dict] = None) -> List[Dict]:
    """
    Build messages with few-shot examples.
    
    Args:
        question: Current question
        examples: Example Q&A pairs
        
    Returns:
        Messages list with examples
    """
    
    if not examples:
        examples = K8S_FEW_SHOT_EXAMPLES
    
    messages = []
    
    for example in examples:
        messages.append({"role": "user", "content": example["question"]})
        messages.append({"role": "assistant", "content": example["answer"]})
    
    # Add current question
    messages.append({"role": "user", "content": question})
    
    return messages


# ── Self-Consistency ────────────────────────────────────────────────────

def self_consistency_prompt(question: str, n_samples: int = 3) -> str:
    """
    Generate prompt for self-consistency.
    
    Strategy:
    1. Generate N answers with temperature > 0
    2. Have the model vote on the best one
    3. Return the consensus answer
    
    Args:
        question: User question
        n_samples: Number of samples to generate
        
    Returns:
        Prompt for self-consistency
    """
    
    return f"""Generate {n_samples} different answers to this question, then analyze them and provide the best consolidated answer.

Question: {question}

Format your response as:
Answer 1: ...
Answer 2: ...
Answer 3: ...

Best Answer (consolidating the above): ..."""
```

---

## Task 6: Integrate Advanced Features

**Modify:** `src/agent/nodes.py`

**Use model fallback:**

```python
from src.advanced.model_fallback import get_fallback_handler

def generate_node(state: dict) -> dict:
    # ... existing code ...
    
    # Replace OpenAI call with fallback handler
    fallback_handler = get_fallback_handler()
    
    try:
        result = fallback_handler.call_with_fallback(
            messages=messages,
            temperature=0.1,
            max_tokens=1000,
            session_id=state["session_id"]
        )
        
        answer = result["content"]
        total_tokens = result["tokens"]
        model_used = result["model"]
        
        log.info(
            "generation_completed",
            session_id=state["session_id"],
            model=model_used,
            provider=result["provider"],
            total_tokens=total_tokens
        )
    
    except Exception as e:
        log.error("all_models_failed", session_id=state["session_id"], error=str(e))
        # Fallback to degraded mode
        return {...}
```

**Use semantic cache:**

```python
from src.advanced.semantic_cache import get_semantic_cache

def retrieve_node(state: dict) -> dict:
    # ... existing code ...
    
    # Check cache first
    cache = get_semantic_cache()
    cached = cache.get(question)
    
    if cached:
        log.info("using_cached_answer", session_id=session_id)
        return {
            **state,
            "context": [],  # No retrieval needed
            "answer": cached["answer"],  # Pre-generated answer
            "sources": cached["sources"],
            "from_cache": True
        }
    
    # Cache miss - proceed with normal retrieval
    # ... existing retrieval code ...
```

**And in generate_node, cache the result:**

```python
def generate_node(state: dict) -> dict:
    # Skip if from cache
    if state.get("from_cache"):
        return state
    
    # ... normal generation ...
    
    # Cache the result
    cache = get_semantic_cache()
    cache.set(
        question=state["question"],
        answer=answer,
        sources=sources,
        metadata={"confidence": confidence}
    )
    
    return {...}
```

---

## Task 7: Add Advanced Feature Metrics

**Modify:** `src/observability/metrics.py`

**Add:**
```python
# ── Advanced Feature Metrics ────────────────────────────────────────────

model_fallback_attempts = Counter(
    "model_fallback_attempts_total",
    "Model fallback attempts",
    ["from_model", "to_model", "reason"]
)

cache_hit_rate = Gauge(
    "cache_hit_rate",
    "Cache hit rate",
    ["cache_type"]
)
```

---

## Verification Steps

**1. Test model fallback:**

```python
# Simulate rate limit by invalidating API key
import os
os.environ["OPENAI_API_KEY"] = "invalid_key"

# Send request - should fallback to next model
```

**2. Test semantic cache:**

```bash
# Send same question twice
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Pod?", "session_id": "cache_test1"}'

# Second request should be instant (cache hit)
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Pod?", "session_id": "cache_test2"}'

# Check logs for cache_hit event
```

**3. Test cache hit rate:**

```bash
# Check metrics
curl http://localhost:8000/metrics | grep cache_hits
curl http://localhost:8000/metrics | grep cache_misses

# Calculate hit rate
```

---

## Success Criteria

- [ ] Model fallback works on failures
- [ ] Semantic cache reduces latency
- [ ] Cache hit rate > 20% in testing
- [ ] Advanced prompting improves quality
- [ ] All features optional (feature flags)

---

## Next Phase

Final phase!

**→ Proceed to `PHASE_10_DEPLOYMENT.md`**

Deployment readiness - Docker, K8s, CI/CD.

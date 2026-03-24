# PHASE 6: RAG Quality Improvements

> **Week:** 4-5  
> **Priority:** P1 (Production Readiness)  
> **Duration:** 5-6 days  
> **Dependencies:** PHASE 3 (RAG), PHASE 5 (metrics to measure improvement)

---

## Objective

Improve RAG quality with reranking, hybrid search (RRF), query decomposition, and citation validation.

**Why this matters:**  
From reference: *"Quality isn't a nice-to-have. It's the difference between users trusting vs abandoning the system."*  
Right now, retrieval is basic. No reranking, no complex query handling.

---

## Pre-Flight Checklist

- [ ] PHASE 1-5 completed
- [ ] ChromaDB and BM25 working
- [ ] Metrics tracking retrieval quality
- [ ] All tests passing

---

## Task 1: Install Dependencies

**Add to `requirements.txt`:**

```
# Reranking
flashrank==0.2.4

# For query decomposition (already have OpenAI)
# No new deps needed

# For citation validation
sentence-transformers==2.3.1  # For semantic similarity
```

**Install:**
```bash
pip install flashrank==0.2.4 sentence-transformers==2.3.1
```

---

## Task 2: Create Reranker Module

**Create:** `src/rag/reranker.py`

```python
"""
Reranking layer using FlashRank.

Why rerank?
- Vector search optimizes for semantic similarity
- Semantic similarity ≠ relevance for the specific query
- Reranker reorders results based on actual relevance

Strategy:
1. Vector search retrieves top 20 candidates
2. Reranker scores each candidate against query
3. Return top 5 reranked results

Performance:
- FlashRank is fast (~10ms for 20 docs)
- Runs on CPU (no GPU needed)
"""

from typing import List, Dict
from flashrank import Ranker, RerankRequest

from src.observability.logging_config import get_logger
from src.observability.metrics import retrieval_latency_seconds
import time

log = get_logger(__name__)


class ChunkReranker:
    """
    Reranker for retrieved chunks.
    
    Uses FlashRank's cross-encoder model to score chunk relevance.
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initialize reranker.
        
        Args:
            model_name: FlashRank model to use
                - "ms-marco-MiniLM-L-12-v2" (default, fast)
                - "ms-marco-MultiBERT-L-12" (slower, higher quality)
        """
        self.ranker = Ranker(model_name=model_name)
        log.info("reranker_initialized", model=model_name)
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Rerank retrieved chunks.
        
        Args:
            query: User question
            chunks: List of chunks from vector search
                Each chunk: {"content": str, "source": str, "score": float, ...}
            top_k: Number of top results to return
            
        Returns:
            Reranked chunks (top_k, sorted by rerank score)
        """
        
        if not chunks:
            return []
        
        start_time = time.time()
        
        # Prepare passages for reranking
        passages = [
            {
                "id": i,
                "text": chunk["content"],
                "meta": chunk  # Keep original chunk data
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Rerank
        rerank_request = RerankRequest(
            query=query,
            passages=passages
        )
        
        results = self.ranker.rerank(rerank_request)
        
        # Convert back to chunk format with new scores
        reranked_chunks = []
        for result in results[:top_k]:
            chunk = result["meta"]
            chunk["rerank_score"] = result["score"]
            chunk["original_score"] = chunk.get("score", 0.0)
            chunk["score"] = result["score"]  # Replace vector score with rerank score
            reranked_chunks.append(chunk)
        
        elapsed = time.time() - start_time
        
        log.info(
            "reranking_completed",
            input_chunks=len(chunks),
            output_chunks=len(reranked_chunks),
            latency_seconds=elapsed
        )
        
        return reranked_chunks


# Global reranker instance
_reranker = None


def get_reranker() -> ChunkReranker:
    """Get or create the global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = ChunkReranker()
    return _reranker
```

---

## Task 3: Implement Hybrid Search (RRF)

**Create:** `src/rag/hybrid_search.py`

```python
"""
Hybrid search using Reciprocal Rank Fusion (RRF).

Strategy:
1. Search with ChromaDB (vector)
2. Search with BM25 (keyword)
3. Fuse results using RRF algorithm
4. Rerank fused results

Why hybrid?
- Vector search: good for semantic similarity
- BM25: good for exact keyword matches
- RRF: combines both without needing score normalization
"""

from typing import List, Dict
import math

from src.observability.logging_config import get_logger

log = get_logger(__name__)


def reciprocal_rank_fusion(
    results_list: List[List[Dict]],
    k: int = 60
) -> List[Dict]:
    """
    Fuse multiple ranked lists using RRF.
    
    RRF formula: score(doc) = sum(1 / (k + rank))
    where rank is the position in each list.
    
    Args:
        results_list: List of ranked result lists
            Each result: {"content": str, "source": str, "score": float, ...}
        k: RRF constant (default: 60, from original paper)
        
    Returns:
        Fused results, sorted by RRF score
        
    Example:
        >>> vector_results = [chunk1, chunk2, chunk3]
        >>> bm25_results = [chunk3, chunk1, chunk4]
        >>> fused = reciprocal_rank_fusion([vector_results, bm25_results])
        # chunk1 and chunk3 appear in both, so they get higher scores
    """
    
    # Map doc ID -> doc data + RRF score
    doc_scores = {}
    
    for result_list in results_list:
        for rank, doc in enumerate(result_list, start=1):
            # Use content hash as doc ID (assumes identical content = same doc)
            doc_id = hash(doc["content"])
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "rrf_score": 0.0
                }
            
            # Add RRF score contribution
            doc_scores[doc_id]["rrf_score"] += 1.0 / (k + rank)
    
    # Sort by RRF score
    fused = sorted(
        doc_scores.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )
    
    # Extract docs and add RRF score
    results = []
    for item in fused:
        doc = item["doc"]
        doc["rrf_score"] = item["rrf_score"]
        doc["score"] = item["rrf_score"]  # Use RRF as primary score
        results.append(doc)
    
    log.info(
        "rrf_fusion_completed",
        input_lists=len(results_list),
        unique_docs=len(results),
        top_score=results[0]["rrf_score"] if results else 0
    )
    
    return results


def hybrid_search(
    query: str,
    chroma_search_fn,
    bm25_search_fn,
    top_k: int = 20
) -> List[Dict]:
    """
    Perform hybrid search using both vector and keyword search.
    
    Args:
        query: User question
        chroma_search_fn: Function to call for vector search
        bm25_search_fn: Function to call for BM25 search
        top_k: Number of candidates to retrieve from each
        
    Returns:
        Fused results
    """
    
    # Get results from both
    vector_results = chroma_search_fn(query, top_k=top_k)
    bm25_results = bm25_search_fn(query, top_k=top_k)
    
    log.info(
        "hybrid_search_started",
        vector_results=len(vector_results),
        bm25_results=len(bm25_results)
    )
    
    # Fuse with RRF
    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    
    return fused
```

---

## Task 4: Implement Query Decomposition

**Create:** `src/rag/query_decomposition.py`

```python
"""
Query decomposition for complex multi-hop questions.

Why decompose?
- "What's the difference between Deployments and StatefulSets?" needs to:
  1. Retrieve docs about Deployments
  2. Retrieve docs about StatefulSets
  3. Compare them
- Single query embedding can't capture this complexity

Strategy:
1. Detect if query is complex (comparison, multi-part)
2. Decompose into sub-queries using LLM
3. Retrieve for each sub-query
4. Merge results
"""

from typing import List, Dict
from openai import OpenAI

from src.config.settings import settings
from src.observability.logging_config import get_logger

log = get_logger(__name__)


DECOMPOSITION_PROMPT = """You are a query decomposition assistant.

Given a complex question, break it down into simpler sub-questions that can be answered independently.

Rules:
- Only decompose if the question has multiple parts or requires comparison
- Each sub-question should be self-contained
- Maximum 3 sub-questions
- If the question is already simple, return it as-is

Examples:

Question: "What's the difference between Deployments and StatefulSets?"
Sub-questions:
1. What are Kubernetes Deployments?
2. What are Kubernetes StatefulSets?

Question: "How do I debug a CrashLoopBackOff?"
Sub-questions:
1. How do I debug a CrashLoopBackOff?
(No decomposition needed - question is already simple)

Question: "What causes high memory usage in pods and how do I monitor it?"
Sub-questions:
1. What causes high memory usage in Kubernetes pods?
2. How do I monitor memory usage in Kubernetes pods?

Now decompose this question:
{question}

Return ONLY the sub-questions, one per line, numbered."""


def should_decompose(query: str) -> bool:
    """
    Heuristic to detect if query needs decomposition.
    
    Triggers:
    - Contains "difference between", "compare"
    - Contains "and" with multiple clauses
    - Contains multiple questions (?)
    """
    
    query_lower = query.lower()
    
    # Comparison indicators
    if any(phrase in query_lower for phrase in [
        "difference between",
        "compare",
        "vs",
        "versus",
        "or"
    ]):
        return True
    
    # Multiple questions
    if query.count("?") > 1:
        return True
    
    # Complex "and" structure (heuristic: long query with "and")
    if " and " in query_lower and len(query) > 50:
        return True
    
    return False


def decompose_query(query: str) -> List[str]:
    """
    Decompose a complex query into sub-queries using LLM.
    
    Args:
        query: Complex user question
        
    Returns:
        List of sub-queries (or [query] if no decomposition needed)
    """
    
    if not should_decompose(query):
        log.info("query_decomposition_skipped", reason="simple_query")
        return [query]
    
    client = OpenAI(api_key=settings.openai_api_key)
    
    prompt = DECOMPOSITION_PROMPT.format(question=query)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        
        output = response.choices[0].message.content.strip()
        
        # Parse sub-questions (numbered lines)
        sub_queries = []
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering (e.g., "1. ", "2. ")
            if line[0].isdigit() and ". " in line:
                line = line.split(". ", 1)[1]
            sub_queries.append(line)
        
        log.info(
            "query_decomposed",
            original=query,
            sub_queries=sub_queries,
            count=len(sub_queries)
        )
        
        return sub_queries if sub_queries else [query]
    
    except Exception as e:
        log.error("query_decomposition_failed", error=str(e))
        return [query]  # Fallback to original query
```

---

## Task 5: Implement Citation Validation

**Create:** `src/rag/citation_validator.py`

```python
"""
Citation validation using semantic similarity.

Why validate citations?
- LLM sometimes hallucinates or misattributes facts
- We need to verify that answer is grounded in retrieved chunks

Strategy:
1. Extract claims from answer
2. For each claim, check if it's supported by any chunk
3. Flag unsupported claims
"""

from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util

from src.observability.logging_config import get_logger

log = get_logger(__name__)


class CitationValidator:
    """
    Validate that answer is grounded in retrieved chunks.
    
    Uses sentence similarity to check if each sentence in the answer
    is semantically similar to at least one chunk.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize validator.
        
        Args:
            model_name: Sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        log.info("citation_validator_initialized", model=model_name)
    
    def validate(
        self,
        answer: str,
        chunks: List[Dict],
        threshold: float = 0.5
    ) -> Tuple[bool, List[str]]:
        """
        Validate that answer is grounded in chunks.
        
        Args:
            answer: Generated answer
            chunks: Retrieved chunks
            threshold: Minimum similarity score to consider grounded
            
        Returns:
            (is_valid, unsupported_sentences)
            is_valid: True if all sentences are grounded
            unsupported_sentences: List of sentences without support
        """
        
        if not chunks:
            return False, [answer]
        
        # Split answer into sentences
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        
        # Get chunk contents
        chunk_texts = [c["content"] for c in chunks]
        
        # Encode
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)
        
        # Check each sentence
        unsupported = []
        for i, sentence in enumerate(sentences):
            # Compute similarity with all chunks
            similarities = util.cos_sim(sentence_embeddings[i], chunk_embeddings)[0]
            max_similarity = similarities.max().item()
            
            if max_similarity < threshold:
                unsupported.append(sentence)
                log.warning(
                    "unsupported_claim_detected",
                    sentence=sentence[:100],
                    max_similarity=max_similarity,
                    threshold=threshold
                )
        
        is_valid = len(unsupported) == 0
        
        log.info(
            "citation_validation_completed",
            total_sentences=len(sentences),
            unsupported_count=len(unsupported),
            is_valid=is_valid
        )
        
        return is_valid, unsupported


# Global validator instance
_validator = None


def get_validator() -> CitationValidator:
    """Get or create the global citation validator."""
    global _validator
    if _validator is None:
        _validator = CitationValidator()
    return _validator
```

---

## Task 6: Integrate into Agent

**Modify:** `src/agent/nodes.py`

**Add imports:**
```python
from src.rag.reranker import get_reranker
from src.rag.hybrid_search import hybrid_search
from src.rag.query_decomposition import decompose_query
from src.rag.citation_validator import get_validator
```

**Update `retrieve_node` with hybrid search + reranking:**

```python
def retrieve_node(state: dict) -> dict:
    with tracer.start_as_current_span("retrieve_node") as span:
        start_time = time.time()
        
        question = state["question"]
        session_id = state["session_id"]
        
        span.set_attribute("session_id", session_id)
        span.set_attribute("question_length", len(question))
        
        # ── QUERY DECOMPOSITION ────────────────────────────────────────
        
        sub_queries = decompose_query(question)
        span.set_attribute("sub_queries_count", len(sub_queries))
        
        all_chunks = []
        
        for sub_query in sub_queries:
            # ── HYBRID SEARCH ───────────────────────────────────────────
            
            if settings.ff_use_chroma:
                # Hybrid search (vector + BM25)
                with tracer.start_as_current_span("hybrid_search"):
                    chunks = hybrid_search(
                        query=sub_query,
                        chroma_search_fn=lambda q, top_k: _chroma_search(q, top_k=top_k),
                        bm25_search_fn=lambda q, top_k: _bm25_search(q, top_k=top_k),
                        top_k=20  # Get 20 candidates from each
                    )
            else:
                chunks = _bm25_search(sub_query, top_k=20)
            
            # ── RERANKING ───────────────────────────────────────────────
            
            with tracer.start_as_current_span("reranking"):
                reranker = get_reranker()
                chunks = reranker.rerank(sub_query, chunks, top_k=5)
                span.set_attribute("chunks_after_rerank", len(chunks))
            
            all_chunks.extend(chunks)
        
        # Deduplicate chunks (same content from different sub-queries)
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            content_hash = hash(chunk["content"])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_chunks.append(chunk)
        
        context = unique_chunks[:10]  # Top 10 after dedup
        
        # ... existing history code ...
        
        elapsed = time.time() - start_time
        retrieval_latency_seconds.observe(elapsed)
        
        log.info(
            "retrieval_completed",
            session_id=session_id,
            sub_queries=len(sub_queries),
            total_chunks_retrieved=len(all_chunks),
            unique_chunks=len(unique_chunks),
            final_chunks=len(context),
            latency_seconds=elapsed
        )
        
        return {**state, "context": context, "history": history}
```

**Update `generate_node` with citation validation:**

```python
def generate_node(state: dict) -> dict:
    with tracer.start_as_current_span("generate_node") as span:
        # ... existing generation code ...
        
        answer = response.choices[0].message.content
        
        # ── CITATION VALIDATION ────────────────────────────────────────
        
        with tracer.start_as_current_span("citation_validation"):
            validator = get_validator()
            is_valid, unsupported = validator.validate(
                answer,
                state["context"],
                threshold=0.5
            )
            
            span.set_attribute("citations_valid", is_valid)
            span.set_attribute("unsupported_claims", len(unsupported))
            
            if not is_valid:
                log.warning(
                    "answer_contains_unsupported_claims",
                    session_id=state["session_id"],
                    unsupported_count=len(unsupported),
                    unsupported_preview=[s[:50] for s in unsupported[:3]]
                )
                
                # Optionally: append warning to answer
                # answer += f"\n\n⚠️ Note: This answer may contain {len(unsupported)} unsupported claim(s)."
        
        # ... rest of code ...
```

---

## Task 7: Add Quality Metrics

**Modify:** `src/observability/metrics.py`

**Add:**
```python
# ── RAG Quality Metrics ─────────────────────────────────────────────────

reranking_improvement = Histogram(
    "reranking_improvement",
    "Score improvement from reranking",
    buckets=[0, 0.1, 0.2, 0.5, 1.0, 2.0]
)

hybrid_search_fusion_score = Histogram(
    "hybrid_search_fusion_score",
    "RRF fusion scores",
    buckets=[0, 0.5, 1.0, 2.0, 5.0, 10.0]
)

citation_validation_failures = Counter(
    "citation_validation_failures_total",
    "Answers with unsupported claims"
)

query_decomposition_count = Counter(
    "query_decomposition_total",
    "Queries that were decomposed",
    ["sub_query_count"]
)
```

**Use metrics in RAG modules:**

In `reranker.py`:
```python
from src.observability.metrics import reranking_improvement

# In rerank() method:
if reranked_chunks:
    improvement = reranked_chunks[0]["rerank_score"] - reranked_chunks[0]["original_score"]
    reranking_improvement.observe(improvement)
```

In `citation_validator.py`:
```python
from src.observability.metrics import citation_validation_failures

# In validate() method:
if not is_valid:
    citation_validation_failures.inc()
```

---

## Task 8: Write RAG Quality Tests

**Create:** `tests/test_rag_quality.py`

```python
"""
Tests for RAG quality improvements.
"""

import pytest

from src.rag.reranker import ChunkReranker
from src.rag.hybrid_search import reciprocal_rank_fusion
from src.rag.query_decomposition import should_decompose, decompose_query
from src.rag.citation_validator import CitationValidator


# ── Reranker Tests ──────────────────────────────────────────────────────

def test_reranker():
    """Test that reranking reorders chunks."""
    reranker = ChunkReranker()
    
    chunks = [
        {"content": "Kubernetes is an orchestration system", "score": 0.9},
        {"content": "The weather is nice today", "score": 0.8},
        {"content": "Pods are the smallest deployable units", "score": 0.7},
    ]
    
    reranked = reranker.rerank("What is Kubernetes?", chunks, top_k=2)
    
    # Should return 2 chunks
    assert len(reranked) == 2
    
    # Should prioritize Kubernetes-related chunks
    assert "Kubernetes" in reranked[0]["content"] or "Pods" in reranked[0]["content"]
    assert "weather" not in reranked[0]["content"]


# ── Hybrid Search Tests ─────────────────────────────────────────────────

def test_rrf_fusion():
    """Test RRF fusion combines rankings."""
    list1 = [
        {"content": "doc1", "score": 0.9},
        {"content": "doc2", "score": 0.8},
    ]
    
    list2 = [
        {"content": "doc2", "score": 0.95},  # doc2 in both lists
        {"content": "doc3", "score": 0.7},
    ]
    
    fused = reciprocal_rank_fusion([list1, list2])
    
    # doc2 should rank highest (appears in both)
    assert fused[0]["content"] == "doc2"
    assert "rrf_score" in fused[0]


# ── Query Decomposition Tests ───────────────────────────────────────────

def test_should_decompose():
    """Test decomposition detection."""
    
    # Should decompose
    assert should_decompose("What's the difference between X and Y?")
    assert should_decompose("Compare deployments vs statefulsets")
    
    # Should NOT decompose
    assert not should_decompose("What is a Pod?")
    assert not should_decompose("How do I debug this?")


def test_decompose_simple_query():
    """Test that simple queries are not decomposed."""
    query = "What is a Kubernetes Pod?"
    result = decompose_query(query)
    
    # Should return original query
    assert len(result) == 1
    assert result[0] == query


# ── Citation Validation Tests ───────────────────────────────────────────

def test_citation_validator():
    """Test citation validation."""
    validator = CitationValidator()
    
    chunks = [
        {"content": "Kubernetes is a container orchestration platform."},
        {"content": "Pods are the smallest deployable units in Kubernetes."},
    ]
    
    # Grounded answer
    answer1 = "Kubernetes is a platform for orchestration. Pods are units in Kubernetes."
    is_valid1, unsupported1 = validator.validate(answer1, chunks, threshold=0.5)
    assert is_valid1
    assert len(unsupported1) == 0
    
    # Ungrounded answer
    answer2 = "Kubernetes was invented by aliens in 2050."
    is_valid2, unsupported2 = validator.validate(answer2, chunks, threshold=0.5)
    assert not is_valid2
    assert len(unsupported2) > 0
```

**Run tests:**
```bash
pytest tests/test_rag_quality.py -v
```

---

## Verification Steps

**1. Test reranking improvement:**

```bash
# Send query and check logs
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a Pod?",
    "session_id": "rerank_test"
  }'

# Check logs for reranking events
tail -f logs/app.log | grep reranking_completed
```

**2. Test hybrid search:**

```bash
# Complex query that benefits from both vector and keyword
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "CrashLoopBackOff troubleshooting",
    "session_id": "hybrid_test"
  }'

# Check logs for hybrid search
tail -f logs/app.log | grep hybrid_search
```

**3. Test query decomposition:**

```bash
# Comparison query
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Whats the difference between Deployments and StatefulSets?",
    "session_id": "decompose_test"
  }'

# Check logs
tail -f logs/app.log | grep query_decomposed
```

**4. Test citation validation:**

```bash
# Check metrics
curl http://localhost:8000/metrics | grep citation_validation_failures

# Send query, check traces in Jaeger for citation_validation span
```

**5. Measure quality improvement:**

```python
# eval_before_after.py
"""
Compare retrieval quality before/after reranking.
"""

from src.agent.nodes import retrieve_node, _chroma_search
from src.rag.reranker import get_reranker

test_queries = [
    "What is a Kubernetes Pod?",
    "How do I debug CrashLoopBackOff?",
    "What causes ImagePullBackOff?",
]

for query in test_queries:
    # Without reranking
    chunks_raw = _chroma_search(query, top_k=5)
    
    # With reranking
    chunks_candidates = _chroma_search(query, top_k=20)
    reranker = get_reranker()
    chunks_reranked = reranker.rerank(query, chunks_candidates, top_k=5)
    
    print(f"\n=== {query} ===")
    print(f"Top without reranking: {chunks_raw[0]['section_title']}")
    print(f"Top with reranking: {chunks_reranked[0]['section_title']}")
    print(f"Score improvement: {chunks_reranked[0]['rerank_score'] - chunks_raw[0]['score']:.3f}")
```

---

## Success Criteria

- [ ] Reranking reorders results (check top result changes)
- [ ] Hybrid search combines vector + BM25
- [ ] Query decomposition works on comparison queries
- [ ] Citation validation detects unsupported claims
- [ ] Quality metrics show improvement
- [ ] All tests pass: `pytest tests/test_rag_quality.py -v`

---

## Expected Improvements

Based on typical results:

- **Reranking**: +10-15% accuracy on retrieval
- **Hybrid search**: +5-10% on keyword-heavy queries
- **Query decomposition**: Handles complex questions that previously failed
- **Citation validation**: Reduces hallucination rate

---

## Next Phase

Once verified:

**→ Proceed to `PHASE_07_EVALUATION.md`**

Continuous evaluation builds on quality improvements (need baseline to measure against).

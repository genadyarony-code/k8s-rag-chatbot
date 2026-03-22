# Project Map — K8s RAG Chatbot

Full architecture documentation, design decisions, and technical rationale.
For the quick-start, see [README.md](README.md).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        docker compose up                             │
│                                                                     │
│  ┌──────────────────┐   HTTP/SSE    ┌──────────────────────────┐   │
│  │  Streamlit UI    │ ────────────► │  FastAPI  :8000          │   │
│  │  :8501           │               │                          │   │
│  │  app.py          │               │  POST /chat              │   │
│  │                  │               │  GET  /health            │   │
│  │  - Streams SSE   │               │  POST /reset/{id}        │   │
│  │  - Health check  │               └────────────┬─────────────┘   │
│  │  - Session reset │                            │                 │
│  └──────────────────┘                            ▼                 │
│                                       ┌──────────────────────┐     │
│                                       │   LangGraph Agent    │     │
│                                       │                      │     │
│                                       │  retrieve ──► generate│     │
│                                       └──────┬───────────┬───┘     │
│                                              │           │         │
│                              ┌───────────────┘           └──────┐  │
│                              ▼                                   ▼  │
│                   ┌─────────────────┐              ┌────────────────┐│
│                   │  ChromaDB       │              │  BM25 Index    ││
│                   │  (cosine HNSW)  │              │  (bm25.pkl)    ││
│                   │                 │              │                ││
│                   │  ~3,600 chunks  │  fallback    │  ~3,600 chunks ││
│                   │  1536-dim vecs  │ ◄──────────  │  no embeddings ││
│                   └─────────────────┘              └────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Ingestion (one-time, before starting the API)

```
data/raw/                  scripts/ingest.py
  k8s_concepts.html   ──►  load_html()
  k8s_in_action.pdf   ──►  load_pdf_smart()        ─── Docling first
  k8s_troubleshoot.html──►  load_html()                  └─ PyMuPDF fallback

        │
        ▼ RawDocument (content, source, doc_type)
  preprocessor.py
    clean_text()           collapse whitespace, strip artifacts
    RecursiveCharText…     1000 chars / 200 overlap
    extract_section_title  last ## heading ## before chunk
    extract_page_number    last [PAGE N] marker before chunk
        │
        ▼ list[Chunk] (chunk_id, content, source, doc_type, section_title, page_number)
  indexer.py
    embed_chunks()         text-embedding-3-small, batches of 100
    load_to_chroma()       PersistentClient, cosine HNSW space
    build_bm25_index()     BM25Okapi, pickle to bm25_index/bm25.pkl
        │
        ▼
  data/processed/index_meta.json   ← startup gate for FastAPI
```

### Request (per query)

```
User question
    │
    ▼ POST /chat  {question, session_id}
FastAPI main.py
    │
    ├── ff_use_streaming=true ──► _stream_response() [async generator]
    │                               │
    │                               ├── asyncio.to_thread(retrieve_node)
    │                               │       │
    │                               │       ├── _detect_doc_type()    ← regex routing
    │                               │       ├── _chroma_search()      ← embed + query
    │                               │       │       score < 0.3 → retry unfiltered
    │                               │       └── session_memory.get()
    │                               │
    │                               └── AsyncOpenAI streaming
    │                                       async for chunk → SSE token events
    │                                       → session_memory.add()
    │                                       → SSE done event with sources
    │
    └── ff_use_streaming=false ──► asyncio.to_thread(graph.invoke)
                                    retrieve_node → generate_node
                                    → ChatResponse JSON
```

---

## Design Decisions

### 1. Regex Routing vs. LLM Classifier

**Decision:** Use a compiled regex of 18 patterns to detect troubleshooting intent.

**Rationale:** The troubleshooting corpus is ~38 chunks (1% of total). Without routing,
vector similarity returns concepts/book results for queries like "why is my pod in CrashLoopBackOff"
because there are simply 100× more chunks with similar vocabulary in the other two corpora.

An LLM classifier call would add ~500ms and ~$0.0001 to every request. The regex
covers ~95% of cases with sub-millisecond execution. The 5% miss rate (edge cases
with unusual phrasing) is acceptable given the fallback: if routing produces
low-score results (< 0.3), the system retries with the full corpus automatically.

**Why 0.3 as the threshold?** ChromaDB always returns exactly `n_results` regardless
of relevance. A correctly routed troubleshooting query returns chunks with scores > 0.4.
A false-positive route (conceptual question mislabeled as troubleshooting) returns
the "best" troubleshooting chunks with scores of 0.2–0.28 — below the threshold.

### 2. BM25 as Fallback (Not Re-ranking)

**Decision:** BM25 is a complete fallback for when ChromaDB is disabled, not a
re-ranking step over vector results.

**Rationale:** Re-ranking (e.g., CrossEncoder) adds 200–500ms latency and another
model to maintain. At this corpus size (~3,600 chunks), the quality delta doesn't
justify the operational cost. BM25 as a fallback catches exact-match technical
queries (`CrashLoopBackOff`, `imagePullBackOff`) that sometimes score poorly in
semantic space because they're rare tokens with little training signal.

**Production path:** `flashrank` re-ranking after top-10 vector retrieval, narrowed
to top-5 for the prompt. Worth adding if corpus grows to >20 documents.

### 3. Manifest-Based Startup Gate

**Decision:** `ingest.py` writes `data/processed/index_meta.json`. FastAPI reads it
in its `lifespan` hook and raises `RuntimeError` if either index is unhealthy.

**Rationale:** A server that starts with partial indexes will produce wrong answers
silently — the worst failure mode for a RAG system. The manifest guarantees that
every running instance has a healthy index, or it never starts at all.

**Schema versioning:** `chunk_schema_version` is written into both the manifest and
the ChromaDB collection metadata. Any mismatch between stored and configured version
causes `load_to_chroma()` to raise. The operator must run `ingest.py --force` to
rebuild. This prevents stale indexes from surviving chunking strategy changes.

### 4. In-Process Session Memory

**Decision:** `SessionMemory` is a thread-safe dict of deques, not Redis.

**Rationale:** Demo scope. Redis adds an operational dependency (connection string,
eviction policy, serialization). The in-process store has one important constraint:
`--workers 1` is required, enforced in `Dockerfile.api`. Memory resets on restart.

**Production path:** Replace `session_memory.add()/get()` calls with Redis LPUSH/LRANGE.
The interface is already isolated to two call sites, making the swap a 20-line change.

### 5. Lazy Client Singletons

**Decision:** OpenAI and ChromaDB clients are initialized on first use via module-level
`_get_openai_client()` / `_get_chroma_collection()` helpers, not on import.

**Rationale:** Eager initialization at module import time would make every test that
imports `src.agent.nodes` require a valid `OPENAI_API_KEY` and a live ChromaDB index.
Lazy initialization means tests that mock `_chroma_search` or `_bm25_search` never
trigger the actual client creation, keeping the test suite fast and dependency-free.

**Performance:** Both clients are created at most once per process lifetime.
ChromaDB `PersistentClient` involves disk I/O; OpenAI `Client` allocates an HTTP
connection pool. Recreating these on every request (the previous behavior) added
measurable overhead on hot paths.

### 6. Loud Degradation in PDF Parsing

**Decision:** The Docling → PyMuPDF fallback prints a colored box to the terminal
with the file name, error reason, and quality impact.

**Rationale:** PDF parsing quality directly affects RAG answer quality. A silent
fallback would produce subtly worse answers with no indication of why. The loud
warning ensures that anyone running `ingest.py` knows they're not getting best
quality, and can investigate (install Docling, fix the PDF) before building the index.

### 7. Async Streaming Architecture

**Decision:** The streaming path in `main.py` uses `AsyncOpenAI` with `async for`,
while the batch path wraps the synchronous `graph.invoke()` with `asyncio.to_thread()`.

**Rationale:** FastAPI's async event loop should not be blocked by I/O-bound calls.
Without `to_thread()`, a synchronous OpenAI call would block all other concurrent
requests for the duration of the LLM response (~1–3s). The streaming path uses
`AsyncOpenAI` directly to yield each token without blocking.

---

## File Reference

```
src/
  config/
    settings.py       Pydantic BaseSettings — all config + feature flags in one place
  ingestion/
    loaders.py        HTML (BeautifulSoup) and PDF (Docling/PyMuPDF) loaders
    preprocessor.py   Cleaning, splitting, section-title extraction
    indexer.py        Embedding batching, ChromaDB write, BM25 pickle
  agent/
    graph.py          LangGraph StateGraph definition (retrieve → generate)
    nodes.py          retrieve_node, generate_node, query routing, client singletons
    memory.py         SessionMemory — thread-safe in-process conversation store
    prompts.py        System prompt + build_prompt() context injection
  api/
    main.py           FastAPI app, lifespan startup gate, /chat /health /reset
    schemas.py        Pydantic request/response models
  ui/
    app.py            Streamlit chat UI with SSE streaming

scripts/
  ingest.py           8-step ingestion pipeline CLI (load → chunk → embed → index → manifest)

tests/
  conftest.py         Shared fixtures (sample docs, mock responses)
  test_agent.py       SessionMemory, build_prompt
  test_api.py         /health, /reset, check_index_health
  test_feature_flags.py  Flag behavior via settings.ff_* patching
  test_ingestion.py   clean_text, section extraction, process_document
  eval/
    eval_questions.json  15 questions across concepts / troubleshooting / how-to
    run_eval.py          Live evaluation against running API (keyword + source scoring)
```

---

## Configuration Reference

All values come from environment variables or `.env`. See `.env.example` for the
full list. Key settings:

| Variable | Default | Effect |
|----------|---------|--------|
| `OPENAI_API_KEY` | required | OpenAI authentication |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for indexing and queries |
| `LLM_MODEL` | `gpt-4o-mini` | Generation model |
| `TOP_K_RESULTS` | `5` | Chunks returned per query |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `CHUNK_SCHEMA_VERSION` | `v1.0` | Bumping this forces index rebuild |
| `FF_USE_CHROMA` | `true` | `false` → BM25 only |
| `FF_USE_OPENAI` | `true` | `false` → return raw chunks |
| `FF_USE_SESSION_MEMORY` | `true` | `false` → stateless mode |
| `FF_USE_STREAMING` | `true` | `false` → batch JSON response |

# ☸️ K8s RAG Chatbot

A production-minded RAG chatbot for Kubernetes knowledge — built with LangGraph, ChromaDB, FastAPI, and Streamlit.

> Assignment context: Built as a technical interview exercise. Architecture decisions and engineering choices are


## What it does

Ask any Kubernetes question in natural language. The system retrieves relevant documentation chunks, generates a grounded answer via GPT-4o-mini, cites its sources, and maintains conversation history across turns.

```
"Why is my Pod stuck in Pending state?"
→ Retrieves: 5 relevant chunks from 3 curated K8s documents
→ Generates: Grounded answer with citations
→ Streams: Token-by-token via SSE
→ Remembers: Last 3 conversation turns
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT UI  :8501                   │
└─────────────────────┬───────────────────────────────────┘
                      │ HTTP / SSE
┌─────────────────────▼───────────────────────────────────┐
│               FASTAPI BACKEND  :8000                     │
│   POST /chat   GET /health   POST /reset/{session_id}    │
│   lifespan: index health check on startup                │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│               LANGGRAPH AGENT                            │
│   [retrieve_node] ──────────► [generate_node]            │
│   query routing + ChromaDB    GPT-4o-mini + streaming   │
│   / BM25 fallback             + cost monitoring          │
│   + session memory                                       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│   ChromaDB (vectors)   BM25 (fallback)   Session Memory  │
└─────────────────────────────────────────────────────────┘
```

**Design principle:** Each layer has one responsibility and knows nothing about the layers above it.

Full architecture documentation → [PROJECT_MAP.md](PROJECT_MAP.md)

---

## Knowledge Base

Three documents chosen to cover three distinct user intents:

| Document | Type | Intent |
|----------|------|--------|
| [kubernetes.io/docs/concepts](https://kubernetes.io/docs/concepts/) | HTML | "Explain what X is" |
| [Kubernetes in Action, 2nd Ed — Ch. 1–10](https://github.com/luksa/kubernetes-in-action-2nd-edition) | PDF | "How do I do X" |
| [learnk8s.io/troubleshooting-deployments](https://learnk8s.io/troubleshooting-deployments) | HTML | "Why isn't X working" |

RAG retrieval quality depends on embedding diversity across intents — not just keyword overlap.



## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Embeddings | `text-embedding-3-small` | ~$0.01 total for 3 docs, high quality on technical text |
| Vector DB | ChromaDB | Local-first, persistent, zero ops |
| Fallback Search | BM25 (`rank_bm25`) | Zero infrastructure — for exact terms like `CrashLoopBackOff` |
| PDF Parsing | Docling → PyMuPDF+pdfplumber | Docling preserves YAML indentation + tables. Graceful fallback if unavailable |
| LLM | `gpt-4o-mini` | Fast, cheap, sufficient for grounded RAG |
| Orchestration | LangGraph | Explicit graph, built-in checkpointer, extensible |
| Backend | FastAPI | Native async, SSE, Pydantic validation |
| Frontend | Streamlit | Fast to build, sufficient for demo |
| Memory | In-process dict | Demo scope — Redis is the obvious next step |
| Feature Flags | `.env` based | Zero dependency, restart = deploy |


## Engineering Decisions Worth Noting

### Graceful PDF Fallback
The ingestion pipeline tries Docling first for the PDF (preserves YAML indentation and tables). If Docling fails — Docker environment, missing models, memory constraints — it falls back to a PyMuPDF + pdfplumber hybrid loader. The fallback is **loud on purpose**: a colored terminal box is printed so the degradation is visible, not silent.

```
╔══════════════════════════════════════════════════════╗
║  ⚠️  DOCLING FALLBACK TRIGGERED                      ║
║  File   : k8s_in_action_ch1-10.pdf                  ║
║  Reason : ...                                        ║
║  Action : Falling back to PyMuPDF + pdfplumber       ║
║  Impact : YAML indentation & tables may be degraded  ║
╚══════════════════════════════════════════════════════╝
```

### Index Sync via Manifest
`ingest.py` writes `data/processed/index_meta.json` after building both indexes. FastAPI checks this file in its `lifespan` startup — if either `chroma_ok` or `bm25_ok` is false, the server **refuses to start** with a clear error message. No silent partial failures.

### Feature Flags as Mitigation Switches
Four operational killswitches, designed at calm time, deployable under pressure:

| Flag | Default | Fallback Behavior |
|------|---------|-------------------|
| `FF_USE_CHROMA` | `true` | BM25 keyword search |
| `FF_USE_OPENAI` | `true` | Return raw chunks, no generation |
| `FF_USE_SESSION_MEMORY` | `true` | Stateless mode |
| `FF_USE_STREAMING` | `true` | Batch response |

```bash
# Example: OpenAI API down
echo "FF_USE_OPENAI=false" >> .env && docker compose restart api
```

### Query Routing for Corpus Imbalance
The three source documents are not equal in size: concepts + book account for ~3,600 chunks while the troubleshooting document accounts for only ~38. Without intervention, vector similarity search almost always returns results from the larger corpora — even for clear troubleshooting questions like "Why is my pod in CrashLoopBackOff?".

The solution is lightweight keyword-based routing **before** the vector search:
1. The question is matched against a compiled regex of ~18 troubleshooting signals (known pod states, diagnostic verbs, "why is/isn't" patterns).
2. If matched, a `doc_type_filter="troubleshooting"` is added to the ChromaDB query to force retrieval from the troubleshooting corpus.
3. If the filtered query returns zero results, it falls back to an unfiltered search.

Why regex instead of an LLM router? An LLM classification call would double the cost of every request. The regex patterns cover ~95% of Kubernetes troubleshooting queries and add zero latency.

### Chunk Schema Versioning
`settings.chunk_schema_version` is stored in ChromaDB collection metadata and in `index_meta.json`. On startup, the indexer compares stored vs current version — mismatches fail fast with instructions to rebuild. Prevents silent retrieval degradation after chunking strategy changes.

### Cost Monitoring
Every LLM call logs token count before and after, with approximate cost. Simple `tiktoken` + `logging` — no Prometheus overhead needed at this scale.

---

## Requirements

- Python 3.13+
- OpenAI API key
- Docker + Docker Compose (for containerized run)

---

## Setup & Run

### Option A: Docker (recommended)

```bash
# 1. Clone and configure
git clone <repo>
cd k8s-rag-chatbot
cp .env.example .env
# Edit .env → add OPENAI_API_KEY

# 2. Place source documents in data/raw/
#    k8s_concepts.html       ← scraped from kubernetes.io/docs/concepts/
#    k8s_in_action_ch1-10.pdf ← Kubernetes in Action 2nd Ed, chapters 1-10
#    k8s_troubleshooting.html ← learnk8s.io/troubleshooting-deployments

# 3. Run ingestion (outside Docker — one time)
pip install -r requirements.txt
python scripts/ingest.py

# If chunking strategy changes, rebuild:
python scripts/ingest.py --force

# 4. Start services
docker compose up --build

# UI  → http://localhost:8501
# API → http://localhost:8000/docs
```

### Option B: Local (no Docker)

```bash
pip install -r requirements.txt
python scripts/ingest.py

# Terminal 1
uvicorn src.api.main:app --reload --port 8000

# Terminal 2
streamlit run src/ui/app.py
```

---

## Environment Variables

```bash
# .env.example

# Required
OPENAI_API_KEY=sk-...

# Optional — override defaults
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
TOP_K_RESULTS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNK_SCHEMA_VERSION=v1.0

# Feature Flags — operational killswitches
FF_USE_CHROMA=true
FF_USE_OPENAI=true
FF_USE_SESSION_MEMORY=true
FF_USE_STREAMING=true
```

---

## Using the UI

The Streamlit interface includes a sidebar with operational controls:

### 🔍 Health Check Button
Displays real-time system status:
- **ChromaDB status**: Vector search availability
- **BM25 status**: Keyword fallback availability  
- **Feature flags**: Current runtime configuration

**Why it's there:** In production, you need visibility into which components are working. If ChromaDB fails, the system automatically falls back to BM25 — but you want to know that's happening. The health check makes degraded operation visible, not silent.

### Feature Flags Display
Shows the current state of all operational killswitches:
- `use_chroma`: Vector search enabled/disabled
- `use_openai`: LLM generation enabled/disabled
- `use_session_memory`: Conversation history enabled/disabled
- `use_streaming`: Token-by-token streaming enabled/disabled

**Why it's there:** Feature flags are mitigation switches designed at calm time, deployable under pressure. The UI exposes them so you can verify the system is running in the expected mode. If OpenAI goes down at 2am, you flip `FF_USE_OPENAI=false` and the system returns raw chunks instead of failing completely — and the UI confirms the flag took effect.

### 🔄 Reset Session Button
Clears conversation history and starts a fresh session.

**Why it's there:** Session memory keeps the last 3 turns for context. Sometimes you want to start over without that history influencing the next answer.

---

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# Evaluation set — tests RAG quality, not just code correctness
python tests/eval/run_eval.py
```

The eval set contains questions with expected keywords and expected source doc types. It scores keyword coverage and source relevance — a RAG system that answers confidently but wrongly is worse than one that says "I don't know."

---

## Project Structure

```
k8s-rag-chatbot/
├── data/
│   ├── raw/                    # Source documents (gitignored)
│   └── processed/
│       ├── chunks.json         # Preprocessed chunks
│       └── index_meta.json     # Index sync manifest
├── src/
│   ├── config/                 # Settings + feature flags
│   ├── ingestion/              # loaders → preprocessor → indexer
│   ├── agent/                  # LangGraph graph + nodes + memory + prompts
│   ├── api/                    # FastAPI backend
│   └── ui/                     # Streamlit frontend
├── tests/
│   ├── test_*.py               # Unit tests
│   └── eval/                   # RAG quality evaluation
├── scripts/
│   └── ingest.py               # One-time ingestion CLI
├── PROJECT_MAP.md              # Full architecture + pseudocode + decisions
├── docker-compose.yml
└── requirements.txt
```

---

## What's Not Here (and Why)

| Feature | Why Not | Next Step |
|---------|---------|-----------|
| Re-ranking | Adds latency, overkill for 3 docs | `flashrank` after top-5 retrieval |
| Redis session store | Demo scope, single process | Replace `SessionMemory` with `redis.Redis` |
| Semantic chunking | LLM-based chunking = cost + slowness for small corpus | `semantic-chunker` if scale grows |
| Authentication | Out of scope for this exercise | FastAPI OAuth2 middleware |
| CI/CD | Out of scope | GitHub Actions: lint + test + docker build |

---

## Full Documentation

All architecture decisions, pseudocode, and technical rationale are in [PROJECT_MAP.md](PROJECT_MAP.md).

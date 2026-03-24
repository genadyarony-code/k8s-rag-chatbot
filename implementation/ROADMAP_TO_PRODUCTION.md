# K8s RAG Chatbot - Production Transformation Roadmap

> **Goal:** Transform the chatbot from 55% → 90% production maturity  
> **Timeline:** 7-8 weeks  
> **Methodology:** Implement all 5 layers of the Production Architecture Ideology

---

## Execution Strategy

This roadmap is divided into **sequential implementation phases**.  
Each phase has a corresponding `PHASE_N_*.md` file with detailed instructions for Claude Code.

**How to use:**
1. Start with `PHASE_01_FOUNDATION.md`
2. Run all tasks in Claude Code
3. Verify tests pass
4. Move to next phase
5. Repeat until complete

**Critical Rules:**
- ✅ Each phase builds on previous phases
- ✅ Tests must pass before moving forward
- ✅ No skipping phases
- ✅ All changes are additive (no breaking changes)

---

## Current State Assessment

| Layer | Current Score | Target Score | Gap |
|-------|--------------|--------------|-----|
| Layer 1: Runtime & Orchestration | 75/100 | 85/100 | +10 |
| Layer 2: Tool Layer | 60/100 | 75/100 | +15 |
| Layer 3: Data & RAG | 80/100 | 90/100 | +10 |
| Layer 4: Guardrails & Safety | 15/100 | 85/100 | **+70** |
| Layer 5: Evaluation & Telemetry | 25/100 | 90/100 | **+65** |

**Overall:** 55/100 → 90/100

---

## Phase Breakdown

### 🔴 PHASE 1: Foundation & Infrastructure (Week 1)
**Priority:** P0 - Deploy Blocking  
**File:** `PHASE_01_FOUNDATION.md`

**Objectives:**
- Set up structured logging (structlog)
- Add Prometheus metrics endpoints
- Implement health checks with readiness/liveness
- Create basic monitoring infrastructure

**Deliverables:**
- ✅ Structured JSON logging throughout codebase
- ✅ `/metrics` endpoint with Prometheus exposition
- ✅ Enhanced `/health` with readiness vs liveness
- ✅ Docker healthcheck working

**Test Criteria:**
- All logs are valid JSON
- Prometheus scrape works
- Health endpoint returns correct status codes

---

### 🔴 PHASE 2: Security & Input Validation (Week 1-2)
**Priority:** P0 - Deploy Blocking  
**File:** `PHASE_02_SECURITY.md`

**Objectives:**
- Implement prompt injection detection
- Add PII redaction (Microsoft Presidio)
- Rate limiting per session
- Input sanitization & validation

**Deliverables:**
- ✅ Prompt injection classifier (regex + LLM fallback)
- ✅ PII detection & redaction pipeline
- ✅ Rate limiter middleware (100 requests/hour per session)
- ✅ Request size limits (10KB input)

**Test Criteria:**
- Prompt injection attempts are blocked
- SSN/credit cards are redacted
- Rate limit returns 429 after threshold
- Oversized requests are rejected

---

### 🔴 PHASE 3: Cost Controls & Circuit Breakers (Week 2)
**Priority:** P0 - Deploy Blocking  
**File:** `PHASE_03_COST_CONTROLS.md`

**Objectives:**
- Token budget enforcement
- Circuit breaker for OpenAI
- Cost tracking & alerts
- Spending limits

**Deliverables:**
- ✅ Per-session token budgets (100k/day default)
- ✅ Circuit breaker with exponential backoff
- ✅ Cost alert thresholds ($10/day warning, $50/day hard limit)
- ✅ Budget exhaustion handling (graceful degradation)

**Test Criteria:**
- Session budget blocks after limit
- Circuit breaker opens after 5 failures
- Cost alerts trigger correctly
- Degraded mode works without OpenAI

---

### 🟡 PHASE 4: Authentication & Authorization (Week 3)
**Priority:** P1 - Production Readiness  
**File:** `PHASE_04_AUTH.md`

**Objectives:**
- API key authentication
- User session management
- Role-based permissions (read-only for now)
- Audit logging

**Deliverables:**
- ✅ API key auth via `Authorization: Bearer <key>`
- ✅ User context in all logs
- ✅ Permission system (ready for expansion)
- ✅ Audit trail for all requests

**Test Criteria:**
- Requests without auth return 401
- Invalid keys return 403
- Audit log captures user_id + action

---

### 🟡 PHASE 5: Advanced Observability (Week 3-4)
**Priority:** P1 - Production Readiness  
**File:** `PHASE_05_OBSERVABILITY.md`

**Objectives:**
- Distributed tracing (OpenTelemetry)
- Grafana dashboards
- Alert rules
- Log aggregation

**Deliverables:**
- ✅ OpenTelemetry instrumentation
- ✅ 3 Grafana dashboards (Health, Quality, Cost)
- ✅ Prometheus alert rules
- ✅ Centralized logging (Loki or stdout for aggregation)

**Test Criteria:**
- Traces show full request lifecycle
- Dashboards display real-time metrics
- Alerts fire on test conditions

---

### 🟡 PHASE 6: RAG Quality Improvements (Week 4-5)
**Priority:** P1 - Production Readiness  
**File:** `PHASE_06_RAG_QUALITY.md`

**Objectives:**
- Reranking layer (flashrank)
- Query decomposition for complex questions
- Citation validation
- Hybrid search (vector + BM25 in parallel)

**Deliverables:**
- ✅ Reranker after vector retrieval
- ✅ Question decomposition for multi-hop queries
- ✅ Citation grounding check
- ✅ RRF (Reciprocal Rank Fusion) for hybrid search

**Test Criteria:**
- Reranking improves eval scores by >5%
- Complex questions are decomposed correctly
- Citations are validated against chunks

---

### 🟢 PHASE 7: Continuous Evaluation (Week 5-6)
**Priority:** P2 - Quality Improvement  
**File:** `PHASE_07_EVALUATION.md`

**Objectives:**
- LLM-as-judge for live responses
- Automated regression detection
- A/B testing framework
- Feedback collection UI

**Deliverables:**
- ✅ LLM-as-judge scoring 10% of responses
- ✅ Nightly eval runs with baseline comparison
- ✅ A/B test infrastructure (prompt variants)
- ✅ Thumbs up/down in Streamlit UI

**Test Criteria:**
- Judge scores correlate with manual review
- Regression alerts trigger on quality drop
- A/B tests assign variants deterministically

---

### 🟢 PHASE 8: Human-in-the-Loop Framework (Week 6-7)
**Priority:** P1 - Production Readiness (for when tools are added)  
**File:** `PHASE_08_HITL.md`

**Objectives:**
- Confidence scoring
- Approval workflow skeleton
- Action authorization matrix
- Audit trail for approvals

**Deliverables:**
- ✅ Confidence score (0-1) for every response
- ✅ Approval queue (Redis-backed)
- ✅ Authorization matrix config
- ✅ Approval logging

**Test Criteria:**
- Low-confidence responses are flagged
- Approval workflow blocks until confirmed
- All actions are logged with approver

---

### 🟢 PHASE 9: Advanced Features (Week 7-8)
**Priority:** P3 - Nice-to-Have  
**File:** `PHASE_09_ADVANCED.md`

**Objectives:**
- Model fallback cascade
- Semantic caching (Redis)
- Multi-tenant isolation
- Advanced prompt engineering

**Deliverables:**
- ✅ gpt-4o-mini → gpt-3.5-turbo → claude fallback
- ✅ Semantic cache for common queries
- ✅ Tenant ID in all operations
- ✅ Chain-of-thought for complex questions

**Test Criteria:**
- Fallback cascade works on rate limit
- Cache hit rate > 20% in testing
- Tenant isolation is enforced

---

### 🟢 PHASE 10: Deployment & Operations (Week 8)
**Priority:** P1 - Production Readiness  
**File:** `PHASE_10_DEPLOYMENT.md`

**Objectives:**
- Production Dockerfile optimization
- Kubernetes manifests
- CI/CD pipeline
- Runbook documentation

**Deliverables:**
- ✅ Multi-stage Docker build
- ✅ K8s deployment + service + ingress
- ✅ GitHub Actions workflow
- ✅ Runbook for common scenarios

**Test Criteria:**
- Docker image builds in <2min
- K8s deployment is stable
- CI runs tests + builds + pushes

---

## Success Metrics

### Before (Current State)
- ❌ No input validation
- ❌ No cost controls
- ❌ No observability
- ❌ No security
- ⚠️ Manual testing only
- ⚠️ Local development only

### After (Target State)
- ✅ Prompt injection blocked
- ✅ PII redacted
- ✅ Cost limits enforced
- ✅ Full observability (logs, metrics, traces)
- ✅ API key auth
- ✅ Automated quality monitoring
- ✅ Production-ready deployment

---

## Non-Goals (Out of Scope)

These are intentionally NOT included to keep scope manageable:

❌ Multi-agent orchestration (current single-agent is sufficient)  
❌ Semantic chunking (current fixed-size chunking works well)  
❌ Custom embedding models (OpenAI embeddings are fine)  
❌ Real-time streaming analytics  
❌ GraphQL API (REST is sufficient)  
❌ Mobile app  
❌ Multi-language support  

---

## Validation Checkpoints

After each phase:

1. **Run all tests:** `pytest tests/ -v`
2. **Check code coverage:** `pytest --cov=src tests/`
3. **Run eval set:** `python tests/eval/run_eval.py`
4. **Manual smoke test:** Start services, send 5 queries
5. **Review logs:** Check for errors/warnings

**Only proceed if all pass.**

---

## Rollback Plan

If any phase fails:

1. **Git revert** to last working commit
2. **Review error logs**
3. **Fix root cause**
4. **Re-run phase**

**Never skip a failing phase.**

---

## File Inventory

| File | Phase | Purpose |
|------|-------|---------|
| `PHASE_01_FOUNDATION.md` | 1 | Structured logging + metrics |
| `PHASE_02_SECURITY.md` | 2 | Input validation + PII |
| `PHASE_03_COST_CONTROLS.md` | 3 | Budgets + circuit breakers |
| `PHASE_04_AUTH.md` | 4 | Authentication + audit |
| `PHASE_05_OBSERVABILITY.md` | 5 | Tracing + dashboards |
| `PHASE_06_RAG_QUALITY.md` | 6 | Reranking + hybrid search |
| `PHASE_07_EVALUATION.md` | 7 | LLM-as-judge + A/B tests |
| `PHASE_08_HITL.md` | 8 | Approval workflows |
| `PHASE_09_ADVANCED.md` | 9 | Model fallback + caching |
| `PHASE_10_DEPLOYMENT.md` | 10 | K8s + CI/CD |

---

## Timeline

```
Week 1:  PHASE 1 ━━━━━ PHASE 2 ━━━━━━━━━
Week 2:  PHASE 3 ━━━━━━━━━━━━━━━━━━━━━━━
Week 3:  PHASE 4 ━━━━━ PHASE 5 ━━━━━━━━━
Week 4:  PHASE 5 ━━━━━ PHASE 6 ━━━━━━━━━
Week 5:  PHASE 6 ━━━━━ PHASE 7 ━━━━━━━━━
Week 6:  PHASE 7 ━━━━━ PHASE 8 ━━━━━━━━━
Week 7:  PHASE 8 ━━━━━ PHASE 9 ━━━━━━━━━
Week 8:  PHASE 9 ━━━━━ PHASE 10 ━━━━━━━━
```

**Total:** 8 weeks, 10 phases, ~60 tasks

---

## Getting Started

**Run in order:**

```bash
# Phase 1
cat PHASE_01_FOUNDATION.md | claude-code

# Wait for completion, verify tests pass

# Phase 2
cat PHASE_02_SECURITY.md | claude-code

# ... and so on
```

**Alternatively:** Give all 10 files to Claude Code sequentially in a conversation.

---

## Maintenance After Completion

Once all phases are done:

**Weekly:**
- Review eval results
- Check cost trends
- Review low-scoring responses

**Monthly:**
- Update dependencies
- Review security alerts
- Analyze usage patterns

**Quarterly:**
- Model performance review
- Consider reranker/embedding upgrades
- Capacity planning

---

## Questions?

If any phase is unclear:
1. Read the detailed `PHASE_*.md` file
2. Check the reference architecture docs
3. Ask for clarification before proceeding

**Never guess. Always verify.**

---

**Ready to begin? Start with `PHASE_01_FOUNDATION.md`**

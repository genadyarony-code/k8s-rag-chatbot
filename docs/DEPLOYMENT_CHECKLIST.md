# Production Deployment Checklist

> Complete every section in order. Sign off at the bottom before going live.

---

## Pre-Deployment

### Code Quality
- [ ] All CI checks passing (`lint`, `test`, `security`)
- [ ] Test coverage ≥ 80%
- [ ] No open `CRITICAL` or `HIGH` Trivy findings
- [ ] No `ruff` lint errors in `src/` and `tests/`

### Configuration
- [ ] `k8s/secret.yaml` updated with real API keys (never commit real keys)
- [ ] `k8s/configmap.yaml` reviewed — feature flags match intended behaviour
- [ ] Cost limits set appropriately for expected traffic
- [ ] Rate limits calibrated (not too tight for load tests)
- [ ] `ENVIRONMENT=production` set in ConfigMap

### Infrastructure
- [ ] Kubernetes namespace exists (`k8s-rag-chatbot`)
- [ ] PVCs created and bound (`chroma-pvc`, `bm25-pvc`)
- [ ] ChromaDB index freshly ingested and verified
- [ ] BM25 index present at `/app/bm25_index/`
- [ ] Ingress controller installed (nginx) and DNS configured
- [ ] TLS certificate provisioned (cert-manager / manual)
- [ ] Monitoring stack deployed (Prometheus, Grafana, Jaeger, Alertmanager)

### Security
- [ ] `ENABLE_AUTH=true` in ConfigMap
- [ ] At least one admin API key provisioned
- [ ] Prompt injection detection enabled (`FF_USE_PROMPT_INJECTION_DETECTION=true`)
- [ ] PII redaction enabled (`FF_USE_PII_REDACTION=true`)
- [ ] Rate limiting enabled (`FF_USE_RATE_LIMITING=true`)
- [ ] No secrets in ConfigMap (only in Secret)

### Observability
- [ ] Structured JSON logs flowing to aggregator
- [ ] `/metrics` endpoint reachable and scraped by Prometheus
- [ ] Traces appearing in Jaeger
- [ ] Grafana dashboards imported and showing data
- [ ] Alert rules loaded in Prometheus (`monitoring/alert_rules.yml`)

### Cost Controls
- [ ] `TOKEN_BUDGET_GLOBAL_DAILY` set to a sensible cap
- [ ] `COST_HARD_LIMIT_USD` set
- [ ] Circuit breaker enabled (always on — `pybreaker`)
- [ ] Model fallback cascade enabled (`FF_USE_MODEL_FALLBACK=true`)

---

## Deployment

- [ ] Docker image built with `Dockerfile.prod` and tagged with semver
- [ ] Image pushed to GHCR (`ghcr.io/genadyarony-code/k8s-rag-chatbot:<version>`)
- [ ] `deployment.yaml` image tag updated to new version
- [ ] `kubectl apply -k k8s/` succeeded (no errors)
- [ ] Rollout completed: `kubectl rollout status deployment/chatbot-api -n k8s-rag-chatbot`
- [ ] All pods `Running` and `Ready` (e.g. `2/2`): `kubectl get pods -n k8s-rag-chatbot`
- [ ] Liveness probe passing: `curl http://localhost:8000/healthz/live`
- [ ] Readiness probe passing: `curl http://localhost:8000/healthz/ready`

---

## Post-Deployment Smoke Tests

Run these within 15 minutes of rollout.

### Basic functionality
- [ ] Health endpoints respond: `GET /healthz/live` → 200, `GET /healthz/ready` → 200
- [ ] Unauthenticated `POST /chat` returns `401` (auth is enforced)
- [ ] Authenticated `POST /chat` with "What is a Pod?" returns a coherent answer
- [ ] Sources are included in the response
- [ ] Confidence score is present in the response

### Security
- [ ] Prompt injection attempt is blocked (e.g. "Ignore previous instructions and...")
- [ ] PII in query is redacted in logs (test with a fake email/phone)
- [ ] Rate limiting activates after N rapid requests (check `429` response)
- [ ] Revoked API key returns `401`

### Observability
- [ ] New request appears in Jaeger traces
- [ ] `chat_requests_total` counter incremented in Prometheus
- [ ] No new alerts firing in Alertmanager

### Performance
- [ ] P95 latency < 5 s under light load
- [ ] Error rate < 1% over first 50 requests
- [ ] Memory and CPU within resource limits: `kubectl top pods -n k8s-rag-chatbot`

---

## Rollback Plan

If any critical issue is found after deployment:

1. **Immediate rollback:**
   ```bash
   kubectl rollout undo deployment/chatbot-api -n k8s-rag-chatbot
   ```
2. Verify rollback completed: `kubectl rollout status deployment/chatbot-api -n k8s-rag-chatbot`
3. Re-run smoke tests against the rolled-back version
4. File an incident report and schedule a post-mortem within 24 hours

---

## Sign-Off

| Role | Name | Date | Signature |
|---|---|---|---|
| Tech Lead | | | |
| DevOps Engineer | | | |
| Security Reviewer | | | |
| Product Manager (notified) | | | |

**Deployed version:** ________________  
**Deployed by:** ________________  
**Date / time (UTC):** ________________  
**Rollback deadline (UTC):** ________________ *(24 h post-deploy)*

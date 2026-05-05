# Production Runbook — k8s-rag-chatbot

> **Audience:** On-call engineers  
> **Last updated:** 2026-03-24  
> **Support escalation:** @platform-team-lead

---

## Quick Reference

| Symptom | First check | Quick fix |
|---|---|---|
| High error rate | `kubectl logs` | Check circuit breaker / OpenAI status |
| High latency | Jaeger traces | Scale pods or restart ChromaDB |
| Cost spike | `/budget` endpoint | Tighten rate limits, block abusive session |
| Pod crashes | `kubectl describe pod` | Check OOM / missing secret |
| Cache misses | Prometheus metrics | Expected until cache warms up |

---

## Scenario 1: High Error Rate

**Alert:** `HighErrorRate` (> 5% in 5 min window)

**Diagnose:**

```bash
# Tail recent errors
kubectl logs -n k8s-rag-chatbot -l app=chatbot-api --tail=200 \
  | python3 -c "import sys,json; [print(json.dumps(json.loads(l))) for l in sys.stdin if '\"level\":\"error\"' in l]"

# Check error counters by status
curl http://localhost:8000/metrics | grep chat_requests_total

# Check circuit breaker state (1 = open)
curl http://localhost:8000/metrics | grep circuit_breaker
```

**Resolution tree:**

1. **OpenAI 429 / rate limit** — Circuit breaker trips automatically; model fallback should kick in. Check `model_fallback_attempts_total`.
2. **OpenAI 5xx / outage** — Wait for OpenAI status (https://status.openai.com). Circuit breaker prevents thundering-herd.
3. **ChromaDB error** — Check PVC health: `kubectl describe pvc chroma-pvc -n k8s-rag-chatbot`. Restart pods: `kubectl rollout restart deployment/chatbot-api -n k8s-rag-chatbot`.
4. **Prompt injection spike** — Check `prompt_injection_blocked_total` metric. No action needed (working as designed). Review rate-limit settings if abusive.
5. **Validation errors** — Check recent requests for malformed payloads.

---

## Scenario 2: High Latency

**Alert:** `HighLatency` (P95 > 10 s)

**Diagnose:**

```bash
# Latency percentiles
curl http://localhost:8000/metrics | grep generation_latency_seconds

# Find slow traces in Jaeger
kubectl port-forward -n monitoring svc/jaeger-query 16686:16686
# Open http://localhost:16686 → search service=k8s-rag-chatbot, min duration 5s

# Resource pressure
kubectl top pods -n k8s-rag-chatbot
```

**Resolution tree:**

1. **Retrieval slow** — Check ChromaDB index fragmentation. Consider running `ingest.py` to rebuild.
2. **LLM slow** — Verify OpenAI status. Confirm `FF_USE_MODEL_FALLBACK=true` so gpt-3.5-turbo is tried on timeout.
3. **Semantic cache cold** — Normal on startup. Cache warms within ~100 requests; no action required.
4. **Memory pressure / OOM** — `kubectl top pods`; if memory near limit, increase pod `resources.limits.memory` or add replicas via HPA.

---

## Scenario 3: Cost Spike

**Alert:** `CostHardLimit` (daily spend > $50)

**Diagnose:**

```bash
# Check current budget status
curl http://localhost:8000/budget | jq .

# Find high-cost sessions in logs
kubectl logs -n k8s-rag-chatbot -l app=chatbot-api \
  | python3 -c "
import sys, json, collections
costs = collections.defaultdict(float)
for line in sys.stdin:
  try:
    rec = json.loads(line)
    if rec.get('event') == 'cost_tracked':
      costs[rec.get('session_id','unknown')] += rec.get('cost_usd', 0)
  except: pass
for sid, total in sorted(costs.items(), key=lambda x:-x[1])[:10]:
  print(f'{total:.4f} USD  {sid}')
"
```

**Resolution:**

1. If usage is legitimate: Increase `COST_HARD_LIMIT_USD` in configmap and roll out.
2. If usage looks like abuse / scraping: Identify the session/user from auth logs and revoke the API key.
3. Tighten rate limits: Reduce `RATE_LIMIT_PER_MINUTE` in configmap.

---

## Scenario 4: Pod Crashes / CrashLoopBackOff

**Diagnose:**

```bash
kubectl get pods -n k8s-rag-chatbot
kubectl logs -n k8s-rag-chatbot <pod-name> --previous
kubectl describe pod -n k8s-rag-chatbot <pod-name>
```

**Common causes:**

| Exit code | Cause | Fix |
|---|---|---|
| 137 | OOMKilled | Increase `resources.limits.memory` |
| 1 | Application error | Check logs for Python traceback |
| 127 | Missing binary | Verify image is built correctly |

**Missing secret:**
```bash
kubectl get secret chatbot-secrets -n k8s-rag-chatbot
# If missing: kubectl apply -f k8s/secret.yaml (fill in real values first)
```

**Index corruption (ChromaDB):**
```bash
# Nuclear option — re-run ingestion
kubectl delete pvc chroma-pvc bm25-pvc -n k8s-rag-chatbot
# Re-run ingest job, then re-apply PVCs
```

---

## Scenario 5: Evaluation Quality Drop

**Alert:** `EvalQualityDrop` (average LLM judge score < 3.0 / 5.0)

**Diagnose:**

```bash
# Check eval score histogram
curl http://localhost:8000/metrics | grep eval_score

# Run offline eval manually
cd /app && python tests/eval/run_eval.py
```

**Resolution:**

1. Check if a recent deployment changed the prompt (A/B test rollout).
2. Review `ab_test_assignments_total` — check if a new variant is underperforming.
3. If regression confirmed: roll back the deployment.

---

## Deployment Procedures

### Standard rollout

```bash
# 1. Build and tag
docker build -f Dockerfile.prod -t ghcr.io/genadyarony-code/k8s-rag-chatbot:v1.2.3 .

# 2. Push to registry
docker push ghcr.io/genadyarony-code/k8s-rag-chatbot:v1.2.3

# 3. Update image in the cluster
kubectl set image deployment/chatbot-api \
  api=ghcr.io/genadyarony-code/k8s-rag-chatbot:v1.2.3 \
  -n k8s-rag-chatbot

# 4. Watch rollout
kubectl rollout status deployment/chatbot-api -n k8s-rag-chatbot

# 5. Verify health
kubectl get pods -n k8s-rag-chatbot
curl http://localhost:8000/healthz/ready
```

### Rollback

```bash
# Instant rollback to previous revision
kubectl rollout undo deployment/chatbot-api -n k8s-rag-chatbot

# Rollback to a specific revision
kubectl rollout history deployment/chatbot-api -n k8s-rag-chatbot
kubectl rollout undo deployment/chatbot-api --to-revision=5 -n k8s-rag-chatbot
```

### Feature flag toggle (no redeploy)

```bash
# Disable model fallback temporarily
kubectl patch configmap chatbot-config -n k8s-rag-chatbot \
  --patch '{"data": {"FF_USE_MODEL_FALLBACK": "false"}}'

# Pods must be restarted to pick up ConfigMap changes
kubectl rollout restart deployment/chatbot-api -n k8s-rag-chatbot
```

---

## Monitoring Quick Links

| Tool | Access | Notes |
|---|---|---|
| Prometheus | `kubectl port-forward svc/prometheus 9090:9090 -n monitoring` | Raw metrics |
| Grafana | `kubectl port-forward svc/grafana 3000:3000 -n monitoring` | Dashboards (admin/admin) |
| Jaeger | `kubectl port-forward svc/jaeger-query 16686:16686 -n monitoring` | Traces |
| Alertmanager | `kubectl port-forward svc/alertmanager 9093:9093 -n monitoring` | Active alerts |

### Key metrics to watch in production

| Metric | Healthy | Warning | Critical |
|---|---|---|---|
| Error rate | < 1% | 5% | > 10% |
| P95 latency | < 5 s | 10 s | > 30 s |
| Daily cost (USD) | < $10 | $25 | > $50 |
| Cache hit rate | — | < 5% | — |
| Pod restarts/h | 0 | 1 | > 3 |

---

## Contact

- **On-call:** @devops-oncall  
- **Escalation:** @platform-team-lead  
- **Post-mortem template:** `docs/POSTMORTEM_TEMPLATE.md`

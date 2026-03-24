# PHASE 10: Deployment & Operations

> **Week:** 8  
> **Priority:** P1 (Production Readiness)  
> **Duration:** 4-5 days  
> **Dependencies:** PHASE 1-9 completed

---

## Objective

Production-ready deployment: optimized Docker images, Kubernetes manifests, CI/CD pipeline, and operational runbooks.

**Why this matters:**  
Everything before this was about building the system. This is about running it reliably in production.

---

## Pre-Flight Checklist

- [ ] All PHASE 1-9 completed
- [ ] All tests passing
- [ ] Security features enabled
- [ ] Monitoring configured

---

## Task 1: Optimize Production Dockerfile

**Create:** `Dockerfile.prod`

```dockerfile
# ── Stage 1: Builder ────────────────────────────────────────────────────
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# ── Stage 2: Runtime ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/
COPY bm25_index/ ./bm25_index/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set Python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/healthz/live || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and test:**

```bash
docker build -f Dockerfile.prod -t k8s-rag-chatbot:latest .

# Test locally
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  k8s-rag-chatbot:latest
```

---

## Task 2: Create Kubernetes Manifests

**Create:** `k8s/namespace.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: k8s-rag-chatbot
  labels:
    name: k8s-rag-chatbot
```

**Create:** `k8s/configmap.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: chatbot-config
  namespace: k8s-rag-chatbot
data:
  # Feature flags
  FF_USE_CHROMA: "true"
  FF_USE_OPENAI: "true"
  FF_USE_SESSION_MEMORY: "true"
  FF_USE_STREAMING: "false"
  
  # Models
  LLM_MODEL: "gpt-4o-mini"
  EMBEDDING_MODEL: "text-embedding-3-small"
  
  # Observability
  OTLP_ENDPOINT: "http://jaeger-collector:4317"
  ENVIRONMENT: "production"
  
  # Cost controls
  TOKEN_BUDGET_SESSION_DAILY: "100000"
  TOKEN_BUDGET_GLOBAL_DAILY: "1000000"
  COST_WARNING_THRESHOLD_USD: "10.0"
  COST_HARD_LIMIT_USD: "50.0"
```

**Create:** `k8s/secret.yaml`

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: chatbot-secrets
  namespace: k8s-rag-chatbot
type: Opaque
stringData:
  # IMPORTANT: Do NOT commit this file with real secrets!
  # Use external secret management (Sealed Secrets, External Secrets Operator, etc.)
  OPENAI_API_KEY: "your-api-key-here"
  # ANTHROPIC_API_KEY: "optional-for-fallback"
```

**Create:** `k8s/deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot-api
  namespace: k8s-rag-chatbot
  labels:
    app: chatbot-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: chatbot-api
  template:
    metadata:
      labels:
        app: chatbot-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: api
        image: k8s-rag-chatbot:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        
        # Environment from ConfigMap
        envFrom:
        - configMapRef:
            name: chatbot-config
        
        # Secrets
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: chatbot-secrets
              key: OPENAI_API_KEY
        
        # Resource limits
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /healthz/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /healthz/ready
            port: 8000
          initialDelaySeconds: 20
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        # Volume mounts
        volumeMounts:
        - name: chroma-data
          mountPath: /app/chroma_db
        - name: bm25-data
          mountPath: /app/bm25_index
      
      volumes:
      - name: chroma-data
        persistentVolumeClaim:
          claimName: chroma-pvc
      - name: bm25-data
        persistentVolumeClaim:
          claimName: bm25-pvc
```

**Create:** `k8s/pvc.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chroma-pvc
  namespace: k8s-rag-chatbot
spec:
  accessModes:
    - ReadWriteMany  # Multiple pods can read
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard  # Adjust for your cluster
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: bm25-pvc
  namespace: k8s-rag-chatbot
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard
```

**Create:** `k8s/service.yaml`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: chatbot-api
  namespace: k8s-rag-chatbot
  labels:
    app: chatbot-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: chatbot-api
```

**Create:** `k8s/ingress.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: chatbot-ingress
  namespace: k8s-rag-chatbot
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod  # If using cert-manager
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - chatbot.yourdomain.com
    secretName: chatbot-tls
  rules:
  - host: chatbot.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: chatbot-api
            port:
              number: 80
```

**Create:** `k8s/hpa.yaml` (Horizontal Pod Autoscaler)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: chatbot-api-hpa
  namespace: k8s-rag-chatbot
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: chatbot-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## Task 3: Create GitHub Actions CI/CD Pipeline

**Create:** `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
  
  lint:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install linters
      run: |
        pip install black flake8 mypy
    
    - name: Run black
      run: black --check src/
    
    - name: Run flake8
      run: flake8 src/ --max-line-length=100
  
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
```

**Create:** `.github/workflows/deploy.yml`

```yaml
name: Deploy

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ghcr.io/${{ github.repository }}
        tags: |
          type=ref,event=branch
          type=semver,pattern={{version}}
          type=sha,prefix={{branch}}-
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        file: Dockerfile.prod
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
  
  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
    
    - name: Deploy to staging
      run: |
        kubectl apply -f k8s/ --namespace=k8s-rag-chatbot-staging
        kubectl rollout status deployment/chatbot-api -n k8s-rag-chatbot-staging
  
  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up kubectl
      uses: azure/setup-kubectl@v3
    
    - name: Deploy to production
      run: |
        kubectl apply -f k8s/ --namespace=k8s-rag-chatbot
        kubectl rollout status deployment/chatbot-api -n k8s-rag-chatbot
```

---

## Task 4: Create Operational Runbooks

**Create:** `docs/RUNBOOK.md`

```markdown
# Production Runbook

## Common Scenarios

### Scenario 1: High Error Rate

**Symptoms:**
- Error rate > 5% in Prometheus
- Alert: `HighErrorRate` firing

**Diagnosis:**
```bash
# Check pod logs
kubectl logs -n k8s-rag-chatbot -l app=chatbot-api --tail=100

# Check error breakdown
curl http://chatbot-api/metrics | grep chat_requests_total

# Check recent errors in structured logs
kubectl logs -n k8s-rag-chatbot -l app=chatbot-api | jq 'select(.level=="error")'
```

**Resolution:**
1. If OpenAI errors (429): Circuit breaker should handle. Check fallback metrics.
2. If ChromaDB errors: Check PVC health, restart pods
3. If validation errors: Check for prompt injection spike

---

### Scenario 2: High Latency

**Symptoms:**
- P95 latency > 10s
- Alert: `HighLatency` firing

**Diagnosis:**
```bash
# Check latency breakdown
curl http://chatbot-api/metrics | grep latency_seconds

# Check trace in Jaeger
# Open http://jaeger:16686, search for slow traces

# Check resource usage
kubectl top pods -n k8s-rag-chatbot
```

**Resolution:**
1. If retrieval slow: Check ChromaDB index size, consider reindexing
2. If generation slow: Check OpenAI status, verify fallback working
3. If memory pressure: Scale up replicas with HPA

---

### Scenario 3: Cost Spike

**Symptoms:**
- Daily cost > $50
- Alert: `CostHardLimit` firing

**Diagnosis:**
```bash
# Check cost by session
kubectl logs -n k8s-rag-chatbot -l app=chatbot-api | \
  jq 'select(.event=="cost_tracked") | {session_id, cost_usd}' | \
  jq -s 'group_by(.session_id) | map({session: .[0].session_id, total: map(.cost_usd) | add})'

# Check circuit breaker status
curl http://chatbot-api/budget | jq '.circuit_breaker'
```

**Resolution:**
1. Identify high-cost sessions
2. Check if legitimate usage or attack
3. If attack: Block session/user, tighten rate limits
4. If legitimate: Increase budget or optimize prompts

---

### Scenario 4: Pod Crashes

**Symptoms:**
- Pods in CrashLoopBackOff
- Restarts > 10 in pod status

**Diagnosis:**
```bash
# Check pod status
kubectl get pods -n k8s-rag-chatbot

# Check previous logs
kubectl logs -n k8s-rag-chatbot <pod-name> --previous

# Describe pod for events
kubectl describe pod -n k8s-rag-chatbot <pod-name>
```

**Resolution:**
1. OOMKilled: Increase memory limits in deployment
2. Missing secrets: Check secret exists and is mounted
3. Index corruption: Delete PVC, rebuild index

---

## Deployment Procedures

### Standard Deployment

```bash
# 1. Build and tag
docker build -f Dockerfile.prod -t k8s-rag-chatbot:v1.2.3 .

# 2. Push to registry
docker push ghcr.io/yourorg/k8s-rag-chatbot:v1.2.3

# 3. Update deployment
kubectl set image deployment/chatbot-api \
  api=ghcr.io/yourorg/k8s-rag-chatbot:v1.2.3 \
  -n k8s-rag-chatbot

# 4. Watch rollout
kubectl rollout status deployment/chatbot-api -n k8s-rag-chatbot

# 5. Verify health
kubectl get pods -n k8s-rag-chatbot
curl http://chatbot-api/health
```

### Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/chatbot-api -n k8s-rag-chatbot

# Rollback to specific revision
kubectl rollout history deployment/chatbot-api -n k8s-rag-chatbot
kubectl rollout undo deployment/chatbot-api --to-revision=5 -n k8s-rag-chatbot
```

---

## Monitoring

### Key Metrics to Watch

- **Error rate**: < 1% (warning at 5%)
- **P95 latency**: < 5s (warning at 10s)
- **Daily cost**: < $10 (hard limit at $50)
- **Cache hit rate**: > 20%
- **Pod restarts**: < 1 per hour

### Dashboards

- Health: http://grafana/d/health
- Cost: http://grafana/d/cost
- Quality: http://grafana/d/quality

---

## Contact

- On-call: @devops-oncall
- Escalation: @platform-team-lead
- Documentation: https://docs.yourorg.com/k8s-rag-chatbot
```

---

## Task 5: Create Pre-Deployment Checklist

**Create:** `docs/DEPLOYMENT_CHECKLIST.md`

```markdown
# Production Deployment Checklist

## Pre-Deployment

### Code Quality
- [ ] All tests passing
- [ ] Code coverage > 80%
- [ ] No linting errors
- [ ] Security scan passed (Trivy)

### Configuration
- [ ] Secrets rotated (API keys)
- [ ] ConfigMap updated with production values
- [ ] Feature flags set correctly
- [ ] Cost limits configured

### Infrastructure
- [ ] Kubernetes cluster ready
- [ ] PVCs created and mounted
- [ ] Ingress configured with TLS
- [ ] Monitoring stack deployed (Prometheus, Grafana, Jaeger)

### Security
- [ ] Prompt injection detection enabled
- [ ] PII redaction enabled
- [ ] Rate limiting configured
- [ ] API key authentication enabled
- [ ] Network policies applied (optional)

### Observability
- [ ] Structured logging enabled
- [ ] Metrics endpoint working
- [ ] Traces sending to Jaeger
- [ ] Dashboards imported to Grafana
- [ ] Alerts configured in Prometheus

### Cost Controls
- [ ] Token budgets set
- [ ] Cost alerts configured
- [ ] Circuit breaker enabled
- [ ] Degraded mode tested

---

## Deployment

- [ ] Docker image built and tagged
- [ ] Image pushed to registry
- [ ] Kubernetes manifests applied
- [ ] Deployment rolled out successfully
- [ ] All pods healthy (3/3 ready)
- [ ] Health checks passing

---

## Post-Deployment

### Smoke Tests
- [ ] Basic query works
- [ ] Authentication required
- [ ] Rate limiting works
- [ ] Prompt injection blocked
- [ ] PII redacted
- [ ] Metrics endpoint accessible

### Monitoring
- [ ] Logs flowing to aggregator
- [ ] Metrics scraped by Prometheus
- [ ] Traces visible in Jaeger
- [ ] Dashboards showing data
- [ ] No alerts firing

### Performance
- [ ] Latency within SLA (< 5s P95)
- [ ] Error rate < 1%
- [ ] Cache hit rate tracked
- [ ] Cost tracking working

---

## Rollback Plan

If issues detected:

1. Check logs: `kubectl logs -n k8s-rag-chatbot -l app=chatbot-api`
2. Check metrics: `curl http://chatbot-api/metrics`
3. If critical: Rollback immediately
   ```bash
   kubectl rollout undo deployment/chatbot-api -n k8s-rag-chatbot
   ```
4. Document incident
5. Post-mortem within 24h

---

## Sign-Off

- [ ] Tech Lead approval
- [ ] DevOps approval
- [ ] Security approval (if changes affect auth/PII)
- [ ] Product Manager notified

**Deployed by:** ________________  
**Date:** ________________  
**Version:** ________________
```

---

## Verification Steps

**1. Deploy to local Kubernetes (minikube/kind):**

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Apply all manifests
kubectl apply -f k8s/

# Check status
kubectl get all -n k8s-rag-chatbot

# Port-forward to test
kubectl port-forward -n k8s-rag-chatbot svc/chatbot-api 8000:80
```

**2. Run smoke tests:**

```bash
# Health check
curl http://localhost:8000/health

# Authenticated request
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a Pod?", "session_id": "smoke_test"}'
```

**3. Verify monitoring:**

```bash
# Check Prometheus scraping
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Open http://localhost:9090/targets

# Check Grafana dashboards
kubectl port-forward -n monitoring svc/grafana 3000:3000
# Open http://localhost:3000

# Check Jaeger traces
kubectl port-forward -n monitoring svc/jaeger 16686:16686
# Open http://localhost:16686
```

**4. Test HPA scaling:**

```bash
# Generate load
for i in {1..100}; do
  curl -X POST http://localhost:8000/chat \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"question": "Test", "session_id": "load_test_'$i'"}' &
done

# Watch HPA
kubectl get hpa -n k8s-rag-chatbot -w
```

---

## Success Criteria

- [ ] Docker image builds in < 5 minutes
- [ ] Kubernetes deployment succeeds
- [ ] All pods healthy (3/3 ready)
- [ ] Health checks passing
- [ ] Metrics being scraped
- [ ] Traces visible in Jaeger
- [ ] HPA scales pods on load
- [ ] Rollout and rollback work
- [ ] CI/CD pipeline runs successfully

---

## Production Go-Live

**Final checklist before production launch:**

1. ✅ All 10 phases completed
2. ✅ Security hardened
3. ✅ Monitoring operational
4. ✅ Cost controls active
5. ✅ Runbooks documented
6. ✅ On-call rotation set up
7. ✅ Incident response plan ready
8. ✅ Backup/restore tested
9. ✅ Disaster recovery plan
10. ✅ Stakeholder sign-off

---

## Congratulations!

You've successfully transformed k8s-rag-chatbot from a **55% production maturity** PoC to a **90%+ production-ready** system.

**What changed:**
- ❌ No observability → ✅ Full telemetry (logs, metrics, traces)
- ❌ No security → ✅ Auth, PII detection, prompt injection blocking
- ❌ No cost controls → ✅ Budgets, circuit breakers, spending limits
- ❌ No quality checks → ✅ Eval framework, reranking, citation validation
- ❌ No deployment → ✅ K8s, CI/CD, monitoring stack

**Next steps:**
- Monitor production metrics
- Iterate based on user feedback
- Optimize costs based on usage patterns
- Consider adding tools (kubectl commands) with HITL approval

---

**The system is production-ready. Deploy with confidence.**

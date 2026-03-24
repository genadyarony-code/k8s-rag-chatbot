# PHASE 3: Cost Controls & Circuit Breakers

> **Week:** 2  
> **Priority:** P0 (Deploy Blocking)  
> **Duration:** 3-4 days  
> **Dependencies:** PHASE 1 (metrics), PHASE 2 (rate limiting)

---

## Objective

Implement token budgets, circuit breakers for OpenAI, cost tracking, and spending limits to prevent runaway costs.

**Why this matters:**  
From the reference article: *"If daily_cost > HARD_LIMIT: pause_non_critical_agents()"*  
Right now, a bug or attack could cost thousands of dollars in a single day.

---

## Pre-Flight Checklist

- [ ] PHASE 1 & 2 completed
- [ ] Prometheus metrics working
- [ ] Rate limiting functional
- [ ] All tests passing

---

## Task 1: Install Dependencies

**Add to `requirements.txt`:**

```
# Circuit breaker
pybreaker==1.0.1

# Redis for distributed token budgets (optional, will use in-memory fallback)
redis==5.0.1
```

**Install:**
```bash
pip install pybreaker==1.0.1 redis==5.0.1
```

---

## Task 2: Create Token Budget Manager

**Create:** `src/cost_control/__init__.py`

```python
# Empty init
```

**Create:** `src/cost_control/token_budget.py`

```python
"""
Token budget enforcement per session and globally.

Budgets:
- Per-session daily limit (default: 100k tokens)
- Global daily limit (default: 1M tokens)
- Per-request limit (default: 10k tokens)

Storage:
- In-memory (default, single instance)
- Redis (production, multi-instance)

Why token budgets?
- Prevent single session from draining credits
- Cap total daily spend
- Detect abnormal usage patterns
"""

import time
from typing import Optional
from datetime import datetime, timedelta

from src.observability.logging_config import get_logger
from src.config.settings import settings

log = get_logger(__name__)


class TokenBudget:
    """
    In-memory token budget tracker.
    
    Tracks:
    - Per-session token usage (resets daily)
    - Global token usage (resets daily)
    - Last reset timestamp
    
    For production with multiple instances, use RedisTokenBudget instead.
    """
    
    def __init__(
        self,
        session_daily_limit: int = 100_000,
        global_daily_limit: int = 1_000_000,
        per_request_limit: int = 10_000
    ):
        self.session_daily_limit = session_daily_limit
        self.global_daily_limit = global_daily_limit
        self.per_request_limit = per_request_limit
        
        # In-memory storage
        self._session_usage: dict[str, int] = {}  # session_id -> tokens used today
        self._global_usage: int = 0
        self._last_reset: datetime = datetime.now()
    
    def _maybe_reset_daily(self):
        """Reset counters if it's a new day."""
        now = datetime.now()
        if now.date() > self._last_reset.date():
            log.info(
                "token_budget_daily_reset",
                previous_global_usage=self._global_usage,
                session_count=len(self._session_usage)
            )
            self._session_usage.clear()
            self._global_usage = 0
            self._last_reset = now
    
    def check_and_reserve(
        self,
        session_id: str,
        estimated_tokens: int
    ) -> tuple[bool, Optional[str]]:
        """
        Check if request is within budget and reserve tokens.
        
        Args:
            session_id: Session identifier
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            (allowed, reason)
            
        Example:
            >>> budget = TokenBudget()
            >>> allowed, reason = budget.check_and_reserve("session1", 1000)
            >>> if not allowed:
            >>>     raise HTTPException(429, detail=reason)
        """
        
        self._maybe_reset_daily()
        
        # Check per-request limit
        if estimated_tokens > self.per_request_limit:
            reason = f"Request exceeds per-request limit ({self.per_request_limit:,} tokens)"
            log.warning(
                "token_budget_exceeded",
                session_id=session_id,
                limit_type="per_request",
                requested=estimated_tokens,
                limit=self.per_request_limit
            )
            return False, reason
        
        # Check session daily limit
        session_used = self._session_usage.get(session_id, 0)
        if session_used + estimated_tokens > self.session_daily_limit:
            reason = f"Session daily limit exceeded ({self.session_daily_limit:,} tokens)"
            log.warning(
                "token_budget_exceeded",
                session_id=session_id,
                limit_type="session_daily",
                used=session_used,
                requested=estimated_tokens,
                limit=self.session_daily_limit
            )
            return False, reason
        
        # Check global daily limit
        if self._global_usage + estimated_tokens > self.global_daily_limit:
            reason = f"Global daily limit exceeded ({self.global_daily_limit:,} tokens)"
            log.error(
                "token_budget_exceeded",
                session_id=session_id,
                limit_type="global_daily",
                used=self._global_usage,
                requested=estimated_tokens,
                limit=self.global_daily_limit
            )
            return False, reason
        
        # Reserve tokens
        self._session_usage[session_id] = session_used + estimated_tokens
        self._global_usage += estimated_tokens
        
        log.info(
            "token_budget_reserved",
            session_id=session_id,
            tokens=estimated_tokens,
            session_total=self._session_usage[session_id],
            global_total=self._global_usage
        )
        
        return True, None
    
    def get_usage(self, session_id: str) -> dict:
        """Get current usage stats for a session."""
        self._maybe_reset_daily()
        
        return {
            "session_used": self._session_usage.get(session_id, 0),
            "session_limit": self.session_daily_limit,
            "session_remaining": self.session_daily_limit - self._session_usage.get(session_id, 0),
            "global_used": self._global_usage,
            "global_limit": self.global_daily_limit,
            "global_remaining": self.global_daily_limit - self._global_usage,
        }


# Global budget instance
_budget: Optional[TokenBudget] = None


def get_token_budget() -> TokenBudget:
    """Get or create the global token budget tracker."""
    global _budget
    if _budget is None:
        _budget = TokenBudget(
            session_daily_limit=getattr(settings, 'token_budget_session_daily', 100_000),
            global_daily_limit=getattr(settings, 'token_budget_global_daily', 1_000_000),
            per_request_limit=getattr(settings, 'token_budget_per_request', 10_000)
        )
    return _budget
```

---

## Task 3: Create Circuit Breaker for OpenAI

**Create:** `src/cost_control/circuit_breaker.py`

```python
"""
Circuit breaker for OpenAI API calls.

Pattern:
- CLOSED: Normal operation, requests go through
- OPEN: Too many failures, requests blocked (fail fast)
- HALF_OPEN: Testing if service recovered, limited requests

Configuration:
- Failure threshold: 5 failures in 60 seconds
- Recovery timeout: 60 seconds
- Half-open max calls: 3

Why circuit breaker?
- Prevents hammering a failing service
- Fails fast instead of waiting for timeouts
- Automatic recovery testing
"""

from typing import Callable, Any
from functools import wraps

from pybreaker import CircuitBreaker, CircuitBreakerError

from src.observability.logging_config import get_logger

log = get_logger(__name__)


# Create circuit breaker instance
openai_breaker = CircuitBreaker(
    fail_max=5,              # Open after 5 failures
    timeout_duration=60,      # Stay open for 60 seconds
    exclude=[ValueError],     # Don't count input validation errors as failures
    name="openai_api"
)


# Listeners for circuit breaker state changes
def on_circuit_open(breaker, *args, **kwargs):
    """Called when circuit opens (too many failures)."""
    log.error(
        "circuit_breaker_opened",
        breaker_name=breaker.name,
        fail_count=breaker.fail_counter,
        message="OpenAI API circuit breaker OPENED - requests will be blocked"
    )


def on_circuit_half_open(breaker, *args, **kwargs):
    """Called when circuit enters half-open state (testing recovery)."""
    log.warning(
        "circuit_breaker_half_open",
        breaker_name=breaker.name,
        message="OpenAI API circuit breaker HALF-OPEN - testing recovery"
    )


def on_circuit_close(breaker, *args, **kwargs):
    """Called when circuit closes (service recovered)."""
    log.info(
        "circuit_breaker_closed",
        breaker_name=breaker.name,
        message="OpenAI API circuit breaker CLOSED - service recovered"
    )


# Register listeners
openai_breaker.add_listener(on_circuit_open, CircuitBreaker.EVENT_OPEN)
openai_breaker.add_listener(on_circuit_half_open, CircuitBreaker.EVENT_HALF_OPEN)
openai_breaker.add_listener(on_circuit_close, CircuitBreaker.EVENT_CLOSE)


def with_circuit_breaker(func: Callable) -> Callable:
    """
    Decorator to wrap OpenAI calls with circuit breaker.
    
    Usage:
        @with_circuit_breaker
        def call_openai(prompt):
            return client.chat.completions.create(...)
    """
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return openai_breaker.call(func, *args, **kwargs)
        except CircuitBreakerError:
            log.error(
                "circuit_breaker_blocked_request",
                function=func.__name__,
                breaker_state=openai_breaker.current_state
            )
            raise
    
    return wrapper
```

---

## Task 4: Create Cost Tracker

**Create:** `src/cost_control/cost_tracker.py`

```python
"""
Cost tracking and alerting.

Tracks:
- Cumulative daily cost
- Cost per session
- Cost per model

Alerts:
- Warning threshold (e.g., $10/day)
- Hard limit (e.g., $50/day)

Action on hard limit:
- Log critical alert
- Enable degraded mode (BM25 only, no OpenAI)
- Send notification (future: email, Slack)
"""

from datetime import datetime
from typing import Optional

from src.observability.logging_config import get_logger
from src.observability.metrics import chat_cost_usd_total

log = get_logger(__name__)


class CostTracker:
    """
    Daily cost tracker with alert thresholds.
    
    Pricing (as of March 2026):
    - gpt-4o-mini: $0.15 / 1M input tokens, $0.60 / 1M output tokens
    - text-embedding-3-small: $0.02 / 1M tokens
    """
    
    # Model pricing (USD per 1M tokens)
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"embedding": 0.02},
    }
    
    def __init__(
        self,
        warning_threshold_usd: float = 10.0,
        hard_limit_usd: float = 50.0
    ):
        self.warning_threshold = warning_threshold_usd
        self.hard_limit = hard_limit_usd
        
        # Daily tracking
        self._daily_cost: float = 0.0
        self._session_costs: dict[str, float] = {}
        self._last_reset: datetime = datetime.now()
        
        # Alert state
        self._warning_sent = False
        self._limit_exceeded = False
    
    def _maybe_reset_daily(self):
        """Reset counters if it's a new day."""
        now = datetime.now()
        if now.date() > self._last_reset.date():
            log.info(
                "cost_tracker_daily_reset",
                previous_cost=self._daily_cost,
                session_count=len(self._session_costs)
            )
            self._daily_cost = 0.0
            self._session_costs.clear()
            self._warning_sent = False
            self._limit_exceeded = False
            self._last_reset = now
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        embedding_tokens: int = 0
    ) -> float:
        """
        Calculate cost for a request.
        
        Args:
            model: Model name (e.g., "gpt-4o-mini")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            embedding_tokens: Number of embedding tokens (for embeddings)
            
        Returns:
            Cost in USD
        """
        
        if model not in self.PRICING:
            log.warning("unknown_model_pricing", model=model)
            return 0.0
        
        pricing = self.PRICING[model]
        cost = 0.0
        
        if "input" in pricing:
            cost += (input_tokens / 1_000_000) * pricing["input"]
        if "output" in pricing:
            cost += (output_tokens / 1_000_000) * pricing["output"]
        if "embedding" in pricing:
            cost += (embedding_tokens / 1_000_000) * pricing["embedding"]
        
        return cost
    
    def track_request(
        self,
        session_id: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        embedding_tokens: int = 0
    ) -> tuple[float, bool]:
        """
        Track cost for a request and check thresholds.
        
        Args:
            session_id: Session identifier
            model: Model used
            input_tokens: Input token count
            output_tokens: Output token count
            embedding_tokens: Embedding token count
            
        Returns:
            (cost, allowed)
            allowed=False if hard limit exceeded
        """
        
        self._maybe_reset_daily()
        
        # Calculate cost
        cost = self.calculate_cost(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            embedding_tokens=embedding_tokens
        )
        
        # Update tracking
        self._daily_cost += cost
        self._session_costs[session_id] = self._session_costs.get(session_id, 0) + cost
        
        # Update Prometheus metric
        chat_cost_usd_total.inc(cost)
        
        log.info(
            "cost_tracked",
            session_id=session_id,
            model=model,
            cost_usd=cost,
            daily_total=self._daily_cost
        )
        
        # Check thresholds
        if self._daily_cost >= self.hard_limit and not self._limit_exceeded:
            self._limit_exceeded = True
            log.critical(
                "cost_hard_limit_exceeded",
                daily_cost=self._daily_cost,
                limit=self.hard_limit,
                action="DEGRADED MODE ENABLED - OpenAI calls blocked"
            )
            return cost, False
        
        if self._daily_cost >= self.warning_threshold and not self._warning_sent:
            self._warning_sent = True
            log.warning(
                "cost_warning_threshold_exceeded",
                daily_cost=self._daily_cost,
                threshold=self.warning_threshold,
                limit=self.hard_limit
            )
        
        return cost, True
    
    def is_budget_available(self) -> bool:
        """Check if budget is still available (not exceeded hard limit)."""
        self._maybe_reset_daily()
        return not self._limit_exceeded
    
    def get_stats(self) -> dict:
        """Get current cost statistics."""
        self._maybe_reset_daily()
        
        return {
            "daily_cost_usd": self._daily_cost,
            "warning_threshold_usd": self.warning_threshold,
            "hard_limit_usd": self.hard_limit,
            "budget_remaining_usd": max(0, self.hard_limit - self._daily_cost),
            "budget_exceeded": self._limit_exceeded,
            "session_count": len(self._session_costs),
        }


# Global cost tracker
_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker(
            warning_threshold_usd=10.0,
            hard_limit_usd=50.0
        )
    return _tracker
```

---

## Task 5: Add Cost Control Configuration

**Modify:** `src/config/settings.py`

**Add cost settings:**
```python
class Settings(BaseSettings):
    # ... existing settings ...
    
    # ── Cost Controls ─────────────────────────────────────────────────────
    # Token budgets
    token_budget_session_daily: int = 100_000  # Tokens per session per day
    token_budget_global_daily: int = 1_000_000  # Total tokens per day
    token_budget_per_request: int = 10_000      # Max tokens per request
    
    # Cost limits (USD)
    cost_warning_threshold_usd: float = 10.0   # Warning alert
    cost_hard_limit_usd: float = 50.0          # Block OpenAI calls
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    
    # ... rest of config
```

---

## Task 6: Integrate Cost Controls into Agent

**Modify:** `src/agent/nodes.py`

**Add imports:**
```python
from src.cost_control.token_budget import get_token_budget
from src.cost_control.cost_tracker import get_cost_tracker
from src.cost_control.circuit_breaker import with_circuit_breaker, openai_breaker, CircuitBreakerError
from fastapi import HTTPException
```

**Wrap OpenAI calls with circuit breaker:**

```python
@with_circuit_breaker
def _call_openai_completion(client, **kwargs):
    """OpenAI call wrapped with circuit breaker."""
    return client.chat.completions.create(**kwargs)


def generate_node(state: dict) -> dict:
    start_time = time.time()
    
    if not settings.ff_use_openai:
        raw = "\n\n".join([f"[{c['section_title']}]\n{c['content']}" for c in state["context"]])
        return {**state, "answer": raw, "sources": [c["source"] for c in state["context"]]}

    # ── COST CONTROL: Check budget before calling OpenAI ──────────────────
    
    messages = build_prompt(state["question"], state["context"], state["history"])
    estimated_tokens = sum(len(_tokenizer.encode(m.get("content", ""))) for m in messages)
    
    # Check token budget
    budget = get_token_budget()
    allowed, reason = budget.check_and_reserve(state["session_id"], estimated_tokens)
    if not allowed:
        log.error(
            "request_blocked_by_budget",
            session_id=state["session_id"],
            estimated_tokens=estimated_tokens,
            reason=reason
        )
        # Return degraded response
        return {
            **state,
            "answer": f"⚠️ Daily token budget exceeded. {reason}",
            "sources": []
        }
    
    # Check cost tracker (hard limit)
    tracker = get_cost_tracker()
    if not tracker.is_budget_available():
        log.error(
            "request_blocked_by_cost_limit",
            session_id=state["session_id"],
            daily_cost=tracker.get_stats()["daily_cost_usd"]
        )
        # Enable degraded mode
        return {
            **state,
            "answer": "⚠️ Daily cost limit exceeded. System is in degraded mode.",
            "sources": []
        }
    
    # ──────────────────────────────────────────────────────────────────────
    
    log.info(
        "generation_started",
        session_id=state["session_id"],
        input_tokens=estimated_tokens
    )

    client = _get_openai_client()
    
    try:
        # Use circuit breaker wrapper
        response = _call_openai_completion(
            client,
            model=settings.llm_model,
            messages=messages,
            temperature=0.1,
            stream=False
        )
    except CircuitBreakerError:
        log.error(
            "openai_circuit_breaker_open",
            session_id=state["session_id"],
            breaker_state=openai_breaker.current_state
        )
        # Return degraded response
        return {
            **state,
            "answer": "⚠️ OpenAI service temporarily unavailable. Circuit breaker is open.",
            "sources": []
        }

    answer = response.choices[0].message.content
    total_tokens = response.usage.total_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    # ── COST TRACKING ──────────────────────────────────────────────────────
    
    cost_usd, _ = tracker.track_request(
        session_id=state["session_id"],
        model=settings.llm_model,
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )
    
    # Record metrics
    chat_tokens_total.labels(model=settings.llm_model).inc(total_tokens)
    
    elapsed = time.time() - start_time
    generation_latency_seconds.observe(elapsed)
    
    log.info(
        "generation_completed",
        session_id=state["session_id"],
        total_tokens=total_tokens,
        cost_usd=cost_usd,
        latency_seconds=elapsed
    )

    sources = list(set(c["source"] for c in state["context"]))

    if settings.ff_use_session_memory:
        session_memory.add(state["session_id"], state["question"], answer)

    return {**state, "answer": answer, "sources": sources}
```

---

## Task 7: Add Cost Monitoring Endpoints

**Modify:** `src/api/main.py`

**Add endpoints:**

```python
@app.get("/budget")
async def get_budget_status():
    """
    Get current token budget and cost status.
    
    Useful for monitoring dashboards.
    """
    from src.cost_control.token_budget import get_token_budget
    from src.cost_control.cost_tracker import get_cost_tracker
    
    budget = get_token_budget()
    tracker = get_cost_tracker()
    
    return {
        "budget": {
            "global_limit": budget.global_daily_limit,
            "global_used": budget._global_usage,
            "global_remaining": budget.global_daily_limit - budget._global_usage,
        },
        "cost": tracker.get_stats(),
        "circuit_breaker": {
            "state": str(openai_breaker.current_state),
            "fail_count": openai_breaker.fail_counter,
        }
    }
```

---

## Task 8: Write Cost Control Tests

**Create:** `tests/test_cost_control.py`

```python
"""
Tests for cost controls: token budgets, circuit breaker, cost tracking.
"""

import pytest
from pybreaker import CircuitBreakerError

from src.cost_control.token_budget import TokenBudget
from src.cost_control.cost_tracker import CostTracker
from src.cost_control.circuit_breaker import with_circuit_breaker


# ── Token Budget Tests ──────────────────────────────────────────────────

def test_token_budget_allows_within_limit():
    """Test that requests within budget are allowed."""
    budget = TokenBudget(session_daily_limit=10_000)
    
    allowed, reason = budget.check_and_reserve("session1", 1000)
    
    assert allowed
    assert reason is None


def test_token_budget_blocks_over_limit():
    """Test that requests exceeding budget are blocked."""
    budget = TokenBudget(session_daily_limit=10_000)
    
    # Use up budget
    budget.check_and_reserve("session1", 9000)
    
    # This should be blocked
    allowed, reason = budget.check_and_reserve("session1", 2000)
    
    assert not allowed
    assert "exceeded" in reason.lower()


def test_token_budget_per_request_limit():
    """Test that oversized requests are blocked."""
    budget = TokenBudget(per_request_limit=5000)
    
    allowed, reason = budget.check_and_reserve("session1", 10000)
    
    assert not allowed
    assert "per-request" in reason.lower()


def test_token_budget_global_limit():
    """Test global daily limit across all sessions."""
    budget = TokenBudget(global_daily_limit=20_000)
    
    # Multiple sessions consume budget
    budget.check_and_reserve("session1", 8000)
    budget.check_and_reserve("session2", 8000)
    
    # This should exceed global limit
    allowed, reason = budget.check_and_reserve("session3", 5000)
    
    assert not allowed
    assert "global" in reason.lower()


# ── Cost Tracker Tests ──────────────────────────────────────────────────

def test_cost_calculation():
    """Test cost calculation for different models."""
    tracker = CostTracker()
    
    # 1000 input, 500 output tokens for gpt-4o-mini
    cost = tracker.calculate_cost(
        model="gpt-4o-mini",
        input_tokens=1000,
        output_tokens=500
    )
    
    # Should be: (1000/1M * 0.15) + (500/1M * 0.60) = 0.00045
    expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
    assert abs(cost - expected) < 0.00001


def test_cost_warning_threshold():
    """Test that warning is logged when threshold exceeded."""
    tracker = CostTracker(warning_threshold_usd=0.001, hard_limit_usd=0.01)
    
    # Simulate expensive request
    cost, allowed = tracker.track_request(
        session_id="test",
        model="gpt-4o-mini",
        input_tokens=10_000,
        output_tokens=5_000
    )
    
    # Should still be allowed but warning sent
    assert allowed
    assert tracker._warning_sent


def test_cost_hard_limit():
    """Test that hard limit blocks requests."""
    tracker = CostTracker(warning_threshold_usd=0.001, hard_limit_usd=0.002)
    
    # Exceed limit
    tracker.track_request("test", "gpt-4o-mini", input_tokens=100_000, output_tokens=50_000)
    
    # Next request should be blocked
    assert not tracker.is_budget_available()


# ── Circuit Breaker Tests ───────────────────────────────────────────────

def test_circuit_breaker_opens_on_failures():
    """Test that circuit breaker opens after repeated failures."""
    
    call_count = 0
    
    @with_circuit_breaker
    def failing_function():
        nonlocal call_count
        call_count += 1
        raise Exception("Simulated OpenAI failure")
    
    # Reset circuit breaker
    from src.cost_control.circuit_breaker import openai_breaker
    openai_breaker._fail_counter = 0
    openai_breaker._state = openai_breaker.STATE_CLOSED
    
    # Trigger failures
    for _ in range(6):
        try:
            failing_function()
        except Exception:
            pass
    
    # Circuit should now be open
    with pytest.raises(CircuitBreakerError):
        failing_function()
```

**Run tests:**
```bash
pytest tests/test_cost_control.py -v
```

---

## Verification Steps

**1. Test token budget enforcement:**

```bash
# Start API
uvicorn src.api.main:app --reload --port 8000

# Check budget status
curl http://localhost:8000/budget

# Should show available budget
```

**2. Simulate budget exhaustion:**

```python
# test_budget.py
import requests

for i in range(15):
    response = requests.post(
        "http://localhost:8000/chat",
        json={"question": "x" * 5000, "session_id": "budget_test"}  # Long question
    )
    print(f"Request {i}: {response.status_code}")

# Should eventually get budget exceeded message
```

**3. Test circuit breaker:**

```bash
# Temporarily break OpenAI (invalid API key)
export OPENAI_API_KEY="invalid_key_test"

# Send requests - circuit should open after 5 failures
for i in {1..10}; do
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"question": "Test", "session_id": "circuit_test"}'
  echo ""
done

# Later requests should fail fast (circuit breaker open)
```

**4. Check cost tracking:**

```bash
# Check metrics
curl http://localhost:8000/metrics | grep chat_cost_usd_total

# Should show cumulative cost

# Check budget endpoint
curl http://localhost:8000/budget | jq

# Should show cost stats
```

**5. Check logs for cost events:**

```bash
tail -f logs/app.log | grep -E "(cost_tracked|cost_warning|cost_hard_limit)"
```

---

## Success Criteria

- [ ] Token budget blocks requests over limit
- [ ] Cost tracker logs all requests
- [ ] Circuit breaker opens after 5 failures
- [ ] Circuit breaker blocks requests when open
- [ ] Hard cost limit triggers degraded mode
- [ ] `/budget` endpoint shows accurate stats
- [ ] All tests pass: `pytest tests/test_cost_control.py -v`

---

## Common Issues

**Issue:** Budget resets unexpectedly  
**Fix:** Daily reset is date-based. In tests, mock datetime.now()

**Issue:** Circuit breaker doesn't open  
**Fix:** Check that exceptions are not in exclude list

**Issue:** Cost calculations wrong  
**Fix:** Verify pricing constants match current OpenAI pricing

---

## Next Phase

Once verified:

**→ Proceed to `PHASE_04_AUTH.md`**

Authentication builds on cost controls (need user IDs for budgets).

# PHASE 7: Continuous Evaluation

> **Week:** 5-6  
> **Priority:** P2 (Quality Improvement)  
> **Duration:** 5-6 days  
> **Dependencies:** PHASE 5 (metrics), PHASE 6 (quality improvements to measure)

---

## Objective

Implement LLM-as-judge for live response evaluation, automated regression detection, A/B testing framework, and user feedback collection.

**Why this matters:**  
From reference: *"You can't improve what you don't measure. Production is your true eval set."*  
Right now you have no idea if responses are getting better or worse over time.

---

## Pre-Flight Checklist

- [ ] PHASE 1-6 completed
- [ ] RAG quality improvements deployed
- [ ] Metrics tracking working
- [ ] All tests passing

---

## Task 1: Create Evaluation Dataset

**Create:** `tests/eval/eval_dataset.json`

```json
{
  "version": "1.0",
  "created_at": "2026-03-24",
  "description": "Golden evaluation set for K8s RAG chatbot",
  "test_cases": [
    {
      "id": "pod_basics_001",
      "question": "What is a Kubernetes Pod?",
      "expected_keywords": ["smallest deployable unit", "container", "shared"],
      "expected_sources": ["pods.md"],
      "difficulty": "easy"
    },
    {
      "id": "crashloop_001",
      "question": "How do I debug a CrashLoopBackOff?",
      "expected_keywords": ["logs", "kubectl logs", "exit code", "crash"],
      "expected_sources": ["troubleshooting.md"],
      "difficulty": "medium"
    },
    {
      "id": "deployment_vs_stateful_001",
      "question": "What's the difference between Deployments and StatefulSets?",
      "expected_keywords": ["stateless", "stateful", "persistent", "identity"],
      "expected_sources": ["deployments.md", "statefulsets.md"],
      "difficulty": "hard"
    },
    {
      "id": "networking_001",
      "question": "How does Kubernetes networking work?",
      "expected_keywords": ["CNI", "pod network", "service"],
      "expected_sources": ["networking.md"],
      "difficulty": "medium"
    },
    {
      "id": "security_001",
      "question": "What are Kubernetes RBAC best practices?",
      "expected_keywords": ["role", "rolebinding", "least privilege"],
      "expected_sources": ["security.md"],
      "difficulty": "hard"
    }
  ]
}
```

---

## Task 2: Create LLM-as-Judge Module

**Create:** `src/evaluation/__init__.py`

```python
# Empty init
```

**Create:** `src/evaluation/llm_judge.py`

```python
"""
LLM-as-judge for response quality evaluation.

Criteria:
- Accuracy: Is the answer factually correct?
- Completeness: Does it answer all parts of the question?
- Relevance: Is it relevant to the question?
- Grounding: Is it supported by the sources?

Output: Score 1-5 + explanation
"""

from typing import Dict, List
from openai import OpenAI
import json

from src.config.settings import settings
from src.observability.logging_config import get_logger

log = get_logger(__name__)


JUDGE_PROMPT = """You are an expert evaluator for a Kubernetes documentation chatbot.

Evaluate this response on the following criteria (1-5 scale):

1. **Accuracy**: Is the information factually correct?
2. **Completeness**: Does it fully answer the question?
3. **Relevance**: Is it relevant and on-topic?
4. **Grounding**: Is it supported by the provided sources?

Question: {question}

Answer: {answer}

Sources used:
{sources}

Return your evaluation as JSON:
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "relevance": <1-5>,
  "grounding": <1-5>,
  "overall": <1-5>,
  "explanation": "<brief explanation>",
  "issues": ["<issue 1>", "<issue 2>", ...]
}}

Be strict but fair. A score of 5 should be rare (perfect answer).
"""


class LLMJudge:
    """
    LLM-as-judge for response evaluation.
    
    Uses GPT-4o-mini to evaluate response quality.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        log.info("llm_judge_initialized", model=model)
    
    def evaluate(
        self,
        question: str,
        answer: str,
        sources: List[str]
    ) -> Dict:
        """
        Evaluate a response.
        
        Args:
            question: User question
            answer: Generated answer
            sources: Sources used
            
        Returns:
            Evaluation dict with scores and explanation
        """
        
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            sources="\n".join(f"- {s}" for s in sources)
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            output = response.choices[0].message.content.strip()
            
            # Parse JSON (strip markdown fences if present)
            if output.startswith("```json"):
                output = output.split("```json")[1].split("```")[0].strip()
            elif output.startswith("```"):
                output = output.split("```")[1].split("```")[0].strip()
            
            evaluation = json.loads(output)
            
            log.info(
                "llm_judge_evaluation_completed",
                overall_score=evaluation.get("overall"),
                accuracy=evaluation.get("accuracy"),
                completeness=evaluation.get("completeness")
            )
            
            return evaluation
        
        except Exception as e:
            log.error("llm_judge_evaluation_failed", error=str(e))
            return {
                "accuracy": 0,
                "completeness": 0,
                "relevance": 0,
                "grounding": 0,
                "overall": 0,
                "explanation": f"Evaluation failed: {str(e)}",
                "issues": ["evaluation_error"]
            }


# Global judge instance
_judge = None


def get_judge() -> LLMJudge:
    """Get or create the global LLM judge."""
    global _judge
    if _judge is None:
        _judge = LLMJudge()
    return _judge
```

---

## Task 3: Create Evaluation Runner

**Create:** `src/evaluation/eval_runner.py`

```python
"""
Evaluation runner for golden test set.

Runs all test cases from eval_dataset.json and generates report.
"""

import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import asyncio

from src.agent.nodes import retrieve_node, generate_node
from src.evaluation.llm_judge import get_judge
from src.observability.logging_config import get_logger

log = get_logger(__name__)


def load_eval_dataset(path: str = "tests/eval/eval_dataset.json") -> Dict:
    """Load evaluation dataset."""
    with open(path) as f:
        return json.load(f)


async def run_eval_case(test_case: Dict) -> Dict:
    """
    Run a single evaluation case.
    
    Args:
        test_case: Test case from dataset
        
    Returns:
        Results dict with answer, sources, scores
    """
    
    question = test_case["question"]
    case_id = test_case["id"]
    
    log.info("eval_case_started", case_id=case_id, question=question)
    
    # Run through agent
    state = {"question": question, "session_id": f"eval_{case_id}"}
    
    # Retrieve
    state = await asyncio.to_thread(retrieve_node, state)
    
    # Generate
    state = await asyncio.to_thread(generate_node, state)
    
    answer = state["answer"]
    sources = state["sources"]
    
    # Evaluate with LLM judge
    judge = get_judge()
    evaluation = judge.evaluate(question, answer, sources)
    
    # Check expected keywords
    keyword_matches = [
        kw for kw in test_case.get("expected_keywords", [])
        if kw.lower() in answer.lower()
    ]
    keyword_coverage = len(keyword_matches) / max(len(test_case.get("expected_keywords", [])), 1)
    
    # Check expected sources
    source_matches = [
        src for src in test_case.get("expected_sources", [])
        if any(src in s for s in sources)
    ]
    source_coverage = len(source_matches) / max(len(test_case.get("expected_sources", [])), 1)
    
    result = {
        "case_id": case_id,
        "question": question,
        "answer": answer,
        "sources": sources,
        "evaluation": evaluation,
        "keyword_coverage": keyword_coverage,
        "source_coverage": source_coverage,
        "difficulty": test_case.get("difficulty", "unknown")
    }
    
    log.info(
        "eval_case_completed",
        case_id=case_id,
        overall_score=evaluation.get("overall"),
        keyword_coverage=keyword_coverage,
        source_coverage=source_coverage
    )
    
    return result


async def run_full_eval(dataset_path: str = "tests/eval/eval_dataset.json") -> Dict:
    """
    Run full evaluation on all test cases.
    
    Returns:
        Report dict with results and aggregates
    """
    
    dataset = load_eval_dataset(dataset_path)
    test_cases = dataset["test_cases"]
    
    log.info("eval_run_started", total_cases=len(test_cases))
    
    # Run all cases
    results = []
    for test_case in test_cases:
        result = await run_eval_case(test_case)
        results.append(result)
    
    # Aggregate scores
    overall_scores = [r["evaluation"]["overall"] for r in results]
    avg_overall = sum(overall_scores) / len(overall_scores)
    
    accuracy_scores = [r["evaluation"]["accuracy"] for r in results]
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    
    keyword_coverages = [r["keyword_coverage"] for r in results]
    avg_keyword_coverage = sum(keyword_coverages) / len(keyword_coverages)
    
    source_coverages = [r["source_coverage"] for r in results]
    avg_source_coverage = sum(source_coverages) / len(source_coverages)
    
    # Count failures (overall < 3)
    failures = [r for r in results if r["evaluation"]["overall"] < 3]
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_version": dataset.get("version"),
        "total_cases": len(test_cases),
        "results": results,
        "aggregates": {
            "avg_overall_score": avg_overall,
            "avg_accuracy_score": avg_accuracy,
            "avg_keyword_coverage": avg_keyword_coverage,
            "avg_source_coverage": avg_source_coverage,
            "failure_count": len(failures),
            "failure_rate": len(failures) / len(test_cases)
        },
        "failures": failures
    }
    
    log.info(
        "eval_run_completed",
        total_cases=len(test_cases),
        avg_overall_score=avg_overall,
        failures=len(failures)
    )
    
    return report


def save_eval_report(report: Dict, output_path: str = "tests/eval/reports"):
    """Save evaluation report to file."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_path}/eval_report_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    
    log.info("eval_report_saved", path=filename)
    return filename
```

**Create:** `tests/eval/run_eval.py`

```python
"""
CLI script to run evaluations.

Usage:
    python tests/eval/run_eval.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.eval_runner import run_full_eval, save_eval_report


async def main():
    print("Running full evaluation...")
    
    report = await run_full_eval()
    
    # Save report
    filename = save_eval_report(report)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total cases: {report['total_cases']}")
    print(f"Average overall score: {report['aggregates']['avg_overall_score']:.2f}/5")
    print(f"Average accuracy: {report['aggregates']['avg_accuracy_score']:.2f}/5")
    print(f"Keyword coverage: {report['aggregates']['avg_keyword_coverage']:.1%}")
    print(f"Source coverage: {report['aggregates']['avg_source_coverage']:.1%}")
    print(f"Failures (<3 score): {report['aggregates']['failure_count']}")
    print(f"\nReport saved: {filename}")
    
    if report['failures']:
        print("\n" + "="*60)
        print("FAILURES")
        print("="*60)
        for failure in report['failures']:
            print(f"\n{failure['case_id']}: {failure['question']}")
            print(f"Score: {failure['evaluation']['overall']}/5")
            print(f"Issues: {', '.join(failure['evaluation'].get('issues', []))}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Task 4: Create Live Sampling Evaluator

**Create:** `src/evaluation/live_sampler.py`

```python
"""
Live sampling evaluator.

Evaluates a sample of production requests with LLM-as-judge.

Strategy:
- Sample 10% of requests (configurable)
- Evaluate asynchronously (don't block response)
- Store results for analysis
"""

import random
from typing import Dict
import asyncio

from src.evaluation.llm_judge import get_judge
from src.observability.logging_config import get_logger

log = get_logger(__name__)


class LiveSampler:
    """
    Sample and evaluate live production requests.
    """
    
    def __init__(self, sample_rate: float = 0.1):
        """
        Initialize sampler.
        
        Args:
            sample_rate: Fraction of requests to evaluate (0.0-1.0)
        """
        self.sample_rate = sample_rate
        self.judge = get_judge()
        log.info("live_sampler_initialized", sample_rate=sample_rate)
    
    def should_sample(self) -> bool:
        """Decide if this request should be sampled."""
        return random.random() < self.sample_rate
    
    async def evaluate_async(
        self,
        question: str,
        answer: str,
        sources: list,
        session_id: str
    ):
        """
        Evaluate a request asynchronously (non-blocking).
        
        Called after response is sent to user.
        """
        
        try:
            evaluation = await asyncio.to_thread(
                self.judge.evaluate,
                question,
                answer,
                sources
            )
            
            log.info(
                "live_evaluation_completed",
                session_id=session_id,
                overall_score=evaluation.get("overall"),
                accuracy=evaluation.get("accuracy")
            )
            
            # Store result (for now, just log; could save to DB)
            # In production: write to PostgreSQL or ClickHouse
            
        except Exception as e:
            log.error("live_evaluation_failed", session_id=session_id, error=str(e))


# Global sampler
_sampler = None


def get_sampler() -> LiveSampler:
    """Get or create the global live sampler."""
    global _sampler
    if _sampler is None:
        _sampler = LiveSampler(sample_rate=0.1)  # 10%
    return _sampler
```

**Integrate into chat endpoint:**

In `src/api/main.py`:

```python
from src.evaluation.live_sampler import get_sampler

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(...):
    # ... existing code ...
    
    # After generating response
    result = await asyncio.to_thread(...)
    
    # ── LIVE SAMPLING EVALUATION ──────────────────────────────────────
    sampler = get_sampler()
    if sampler.should_sample():
        # Evaluate asynchronously (don't wait)
        asyncio.create_task(
            sampler.evaluate_async(
                question=validated_question,
                answer=result["answer"],
                sources=result["sources"],
                session_id=request.session_id
            )
        )
    # ──────────────────────────────────────────────────────────────────
    
    return ChatResponse(...)
```

---

## Task 5: Create A/B Testing Framework

**Create:** `src/evaluation/ab_testing.py`

```python
"""
A/B testing framework for prompt variants.

Example:
- Variant A: Current prompt
- Variant B: Prompt with "Be concise"
- Variant C: Prompt with "Use examples"

Assign variants deterministically based on session_id hash.
"""

import hashlib
from typing import Dict, List

from src.observability.logging_config import get_logger

log = get_logger(__name__)


class ABTest:
    """
    A/B test configuration.
    """
    
    def __init__(
        self,
        test_id: str,
        variants: Dict[str, Dict],
        traffic_split: Dict[str, float]
    ):
        """
        Initialize A/B test.
        
        Args:
            test_id: Unique test identifier
            variants: Dict of variant_name -> config
            traffic_split: Dict of variant_name -> traffic fraction
                Must sum to 1.0
                
        Example:
            >>> test = ABTest(
            ...     test_id="prompt_style_v1",
            ...     variants={
            ...         "control": {"instruction": ""},
            ...         "concise": {"instruction": "Be concise"},
            ...         "detailed": {"instruction": "Provide detailed explanations"}
            ...     },
            ...     traffic_split={
            ...         "control": 0.5,
            ...         "concise": 0.25,
            ...         "detailed": 0.25
            ...     }
            ... )
        """
        
        self.test_id = test_id
        self.variants = variants
        self.traffic_split = traffic_split
        
        # Validate
        assert abs(sum(traffic_split.values()) - 1.0) < 0.001, "Traffic split must sum to 1.0"
        
        log.info(
            "ab_test_initialized",
            test_id=test_id,
            variants=list(variants.keys()),
            traffic_split=traffic_split
        )
    
    def assign_variant(self, session_id: str) -> str:
        """
        Assign a variant based on session_id hash.
        
        Uses deterministic hashing so same session always gets same variant.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Variant name
        """
        
        # Hash session_id to [0, 1)
        hash_value = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        fraction = (hash_value % 1000000) / 1000000.0
        
        # Assign based on traffic split
        cumulative = 0.0
        for variant_name, split in self.traffic_split.items():
            cumulative += split
            if fraction < cumulative:
                log.info(
                    "ab_variant_assigned",
                    test_id=self.test_id,
                    session_id=session_id,
                    variant=variant_name
                )
                return variant_name
        
        # Fallback (should never reach)
        return list(self.variants.keys())[0]
    
    def get_variant_config(self, session_id: str) -> Dict:
        """Get the config for the assigned variant."""
        variant_name = self.assign_variant(session_id)
        return self.variants[variant_name]


# Example A/B test: prompt style
PROMPT_STYLE_TEST = ABTest(
    test_id="prompt_style_v1",
    variants={
        "control": {
            "instruction": ""
        },
        "concise": {
            "instruction": "Be concise and direct. Answer in 2-3 sentences maximum unless more detail is explicitly requested."
        },
        "detailed": {
            "instruction": "Provide detailed explanations with examples. Break down complex topics step-by-step."
        }
    },
    traffic_split={
        "control": 0.5,
        "concise": 0.25,
        "detailed": 0.25
    }
)


def get_active_tests() -> List[ABTest]:
    """Get list of active A/B tests."""
    return [
        PROMPT_STYLE_TEST,
        # Add more tests here
    ]
```

**Use in prompts:**

In `src/agent/prompts.py`:

```python
from src.evaluation.ab_testing import PROMPT_STYLE_TEST

def build_prompt(question: str, context: list, history: list, session_id: str) -> list:
    # ... existing prompt building ...
    
    # A/B test: add variant instruction
    variant_config = PROMPT_STYLE_TEST.get_variant_config(session_id)
    if variant_config.get("instruction"):
        system_message += f"\n\n{variant_config['instruction']}"
    
    # ... rest of code
```

---

## Task 6: Create Regression Detection

**Create:** `src/evaluation/regression_detector.py`

```python
"""
Regression detection by comparing eval runs.

Strategy:
- Store baseline eval report
- Compare new runs against baseline
- Alert if scores drop significantly
"""

import json
from pathlib import Path
from typing import Dict, List

from src.observability.logging_config import get_logger

log = get_logger(__name__)


def load_baseline(path: str = "tests/eval/baseline.json") -> Dict:
    """Load baseline evaluation report."""
    with open(path) as f:
        return json.load(f)


def detect_regressions(
    current_report: Dict,
    baseline_report: Dict,
    threshold: float = 0.3
) -> List[Dict]:
    """
    Detect regressions by comparing current to baseline.
    
    Args:
        current_report: New eval report
        baseline_report: Baseline eval report
        threshold: Score drop threshold to flag as regression (0.3 = 30%)
        
    Returns:
        List of regression findings
    """
    
    regressions = []
    
    # Overall score regression
    current_overall = current_report["aggregates"]["avg_overall_score"]
    baseline_overall = baseline_report["aggregates"]["avg_overall_score"]
    
    if current_overall < baseline_overall * (1 - threshold):
        regressions.append({
            "type": "overall_score",
            "current": current_overall,
            "baseline": baseline_overall,
            "drop": baseline_overall - current_overall,
            "severity": "critical"
        })
    
    # Per-case regressions
    baseline_cases = {r["case_id"]: r for r in baseline_report["results"]}
    
    for current_case in current_report["results"]:
        case_id = current_case["case_id"]
        
        if case_id not in baseline_cases:
            continue
        
        baseline_case = baseline_cases[case_id]
        
        current_score = current_case["evaluation"]["overall"]
        baseline_score = baseline_case["evaluation"]["overall"]
        
        if current_score < baseline_score * (1 - threshold):
            regressions.append({
                "type": "case_regression",
                "case_id": case_id,
                "question": current_case["question"],
                "current_score": current_score,
                "baseline_score": baseline_score,
                "drop": baseline_score - current_score,
                "severity": "warning"
            })
    
    log.info("regression_detection_completed", regressions_found=len(regressions))
    
    return regressions
```

---

## Task 7: Add Eval Metrics

**Modify:** `src/observability/metrics.py`

**Add:**
```python
# ── Evaluation Metrics ──────────────────────────────────────────────────

eval_score = Histogram(
    "eval_score",
    "LLM judge evaluation scores",
    ["metric"],  # accuracy, completeness, relevance, grounding, overall
    buckets=[1, 2, 3, 4, 5]
)

ab_test_assignments = Counter(
    "ab_test_assignments_total",
    "A/B test variant assignments",
    ["test_id", "variant"]
)
```

**Use in evaluation modules.**

---

## Verification Steps

**1. Run baseline evaluation:**

```bash
python tests/eval/run_eval.py

# Save as baseline
cp tests/eval/reports/eval_report_*.json tests/eval/baseline.json
```

**2. Test live sampling:**

```bash
# Send requests, check logs for live_evaluation_completed
for i in {1..20}; do
  curl -X POST http://localhost:8000/chat \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"What is a Pod?\", \"session_id\": \"live_test_$i\"}"
done

# Check logs - should see ~10% evaluated
grep live_evaluation_completed logs/app.log | wc -l
```

**3. Test A/B assignments:**

```bash
# Same session should always get same variant
for i in {1..5}; do
  curl -X POST http://localhost:8000/chat \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"question": "Test", "session_id": "ab_test_session"}'
done

# Check logs - all should get same variant
grep ab_variant_assigned logs/app.log | grep ab_test_session
```

**4. Test regression detection:**

```python
# test_regression.py
from src.evaluation.regression_detector import detect_regressions, load_baseline

baseline = load_baseline()

# Simulate degraded performance
current = baseline.copy()
current["aggregates"]["avg_overall_score"] = 2.5  # Down from ~4.0

regressions = detect_regressions(current, baseline)

for reg in regressions:
    print(f"{reg['type']}: {reg['severity']}")
    print(f"  Drop: {reg['drop']:.2f}")
```

---

## Success Criteria

- [ ] Eval runner completes without errors
- [ ] Baseline report shows reasonable scores (>3.0 avg)
- [ ] Live sampling evaluates ~10% of requests
- [ ] A/B test assigns variants deterministically
- [ ] Regression detector flags score drops
- [ ] All metrics tracking eval scores

---

## Next Phase

Once verified:

**→ Proceed to `PHASE_08_HITL.md`**

Human-in-the-loop builds on evaluation (low-confidence responses need approval).

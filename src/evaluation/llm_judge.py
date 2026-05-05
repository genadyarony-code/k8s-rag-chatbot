"""
LLM-as-judge for response quality evaluation.

Each response is scored on four criteria:
- Accuracy:      Is the information factually correct?
- Completeness:  Does it fully answer the question?
- Relevance:     Is it on-topic and directly useful?
- Grounding:     Is every claim supported by the cited sources?

All criteria use a 1–5 integer scale. The judge is GPT-4o-mini with
temperature=0 for consistent, reproducible scores.

Why LLM-as-judge?
Human evaluation does not scale to production traffic. An LLM judge
correlates well with human judgements on factual Q&A tasks and is
consistent across runs (temperature=0, fixed prompt).

Bias note: GPT-4o-mini evaluating GPT-4o-mini responses introduces
self-serving bias. A production system should use a different model
family for the judge. For this project the cost and latency trade-off
favours keeping a single provider.
"""

import json
from typing import Optional

from openai import OpenAI

from src.config.settings import settings
from src.observability.logging_config import get_logger
from src.observability.metrics import eval_score

log = get_logger(__name__)

_JUDGE_PROMPT = """\
You are an expert evaluator for a Kubernetes documentation chatbot.

Evaluate this response on the following criteria using a 1–5 integer scale:

1. accuracy:     Is the information factually correct?
2. completeness: Does it fully answer the question?
3. relevance:    Is it on-topic and directly useful?
4. grounding:    Is every claim supported by the provided sources?

A score of 5 = perfect; 3 = acceptable; 1 = wrong or unhelpful.
Be strict — a 5 should be rare.

Question: {question}

Answer:
{answer}

Sources used:
{sources}

Return ONLY valid JSON — no markdown fences, no extra text:
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "relevance": <1-5>,
  "grounding": <1-5>,
  "overall": <1-5>,
  "explanation": "<one sentence summary>",
  "issues": ["<issue 1>", "<issue 2>"]
}}\
"""


class LLMJudge:
    """
    LLM-as-judge for response evaluation.

    Thread-safe for concurrent use — the OpenAI client is stateless between
    calls. The singleton instance is cached after first initialisation.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model
        log.info("llm_judge_initialized", model=model)

    def evaluate(
        self,
        question: str,
        answer: str,
        sources: list[str],
    ) -> dict:
        """
        Evaluate a (question, answer, sources) triple.

        Returns a dict with keys: accuracy, completeness, relevance, grounding,
        overall (all 1–5 ints), explanation (str), issues (list[str]).
        On any error returns all zeros so callers can safely read the keys.
        """
        prompt = _JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            sources="\n".join(f"- {s}" for s in sources) or "(none)",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )

            raw = response.choices[0].message.content.strip()

            # Strip optional markdown fences GPT sometimes adds
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.rsplit("```", 1)[0].strip()

            evaluation: dict = json.loads(raw)

            # Emit Prometheus metrics for each criterion
            for criterion in ("accuracy", "completeness", "relevance", "grounding", "overall"):
                if criterion in evaluation:
                    eval_score.labels(metric=criterion).observe(evaluation[criterion])

            log.info(
                "llm_judge_completed",
                overall=evaluation.get("overall"),
                accuracy=evaluation.get("accuracy"),
                completeness=evaluation.get("completeness"),
            )

            return evaluation

        except Exception as exc:
            log.error("llm_judge_failed", error=str(exc))
            return {
                "accuracy": 0,
                "completeness": 0,
                "relevance": 0,
                "grounding": 0,
                "overall": 0,
                "explanation": f"Evaluation failed: {exc}",
                "issues": ["evaluation_error"],
            }


# ── Lazy singleton ────────────────────────────────────────────────────────────

_judge: Optional[LLMJudge] = None


def get_judge() -> LLMJudge:
    """Get or create the global LLM judge instance."""
    global _judge
    if _judge is None:
        _judge = LLMJudge()
    return _judge

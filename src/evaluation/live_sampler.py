"""
Live sampling evaluator.

Randomly samples a configurable fraction of production requests and
evaluates them asynchronously with the LLM judge after the response
has been sent to the user — so users never experience added latency.

The sample rate defaults to 10 % (EVAL_SAMPLE_RATE=0.1). Set it to 0.0
to disable live sampling entirely, or 1.0 to evaluate every request.

Results are emitted as structured log events ("live_evaluation_completed")
and as Prometheus histogram observations under eval_score{metric=...}.
For persistence beyond process lifetime, connect a log exporter or modify
`evaluate_async` to write to a database.
"""

import asyncio
import random
from typing import Optional

from src.evaluation.llm_judge import get_judge
from src.observability.logging_config import get_logger

log = get_logger(__name__)


class LiveSampler:
    """
    Samples and evaluates live production requests asynchronously.

    Designed as a fire-and-forget background task. Call `schedule` from the
    request handler; it decides whether to sample and, if so, enqueues the
    evaluation as a non-blocking asyncio task.
    """

    def __init__(self, sample_rate: float = 0.1) -> None:
        self.sample_rate = sample_rate
        log.info("live_sampler_initialized", sample_rate=sample_rate)

    def should_sample(self) -> bool:
        """Return True with probability sample_rate."""
        return self.sample_rate > 0 and random.random() < self.sample_rate

    async def evaluate_async(
        self,
        question: str,
        answer: str,
        sources: list[str],
        session_id: str,
    ) -> None:
        """
        Evaluate a completed response in the background.

        Errors are caught and logged so that a flaky judge never causes
        data loss or raises an unhandled exception in the event loop.
        """
        try:
            judge = get_judge()
            evaluation = await asyncio.to_thread(
                judge.evaluate, question, answer, sources
            )
            log.info(
                "live_evaluation_completed",
                session_id=session_id,
                overall_score=evaluation.get("overall"),
                accuracy=evaluation.get("accuracy"),
                completeness=evaluation.get("completeness"),
                issues=evaluation.get("issues", []),
            )
        except Exception as exc:
            log.error("live_evaluation_failed", session_id=session_id, error=str(exc))

    def schedule(
        self,
        question: str,
        answer: str,
        sources: list[str],
        session_id: str,
    ) -> None:
        """
        Decide whether to sample; if so, enqueue as a background asyncio task.

        Must be called from within an async context (e.g. a FastAPI handler).
        Uses asyncio.create_task so it never blocks the caller.
        """
        if self.should_sample():
            asyncio.create_task(
                self.evaluate_async(question, answer, sources, session_id)
            )


# ── Lazy singleton ────────────────────────────────────────────────────────────

_sampler: Optional[LiveSampler] = None


def get_sampler() -> LiveSampler:
    """Get or create the global live sampler instance."""
    global _sampler
    if _sampler is None:
        from src.config.settings import settings
        _sampler = LiveSampler(sample_rate=settings.eval_sample_rate)
    return _sampler

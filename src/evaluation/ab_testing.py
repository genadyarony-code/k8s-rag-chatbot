"""
A/B testing framework for prompt variants.

Variants are assigned deterministically from session_id — the same session
always receives the same variant. This is important for a conversational
system: mid-session variant switching would produce inconsistent response
styles and invalidate the experiment.

Assignment algorithm:
    bucket = MD5(session_id) mod 1_000_000   →  in [0, 999_999]
    map bucket to a variant via cumulative traffic split

Traffic split must sum to exactly 1.0 (±0.001 tolerance).

Current active test: prompt_style_v1
    control  (50 %) — standard system prompt
    concise  (25 %) — "Be concise and direct"
    detailed (25 %) — "Provide detailed explanations with examples"

To add a new test, instantiate ABTest and add it to the list returned by
get_active_tests(). Tests are independent; a session participates in all
active tests simultaneously.
"""

import hashlib
from typing import Optional

from src.observability.logging_config import get_logger
from src.observability.metrics import ab_test_assignments

log = get_logger(__name__)


class ABTest:
    """
    Single A/B test with deterministic session-based variant assignment.

    Args:
        test_id:       Unique identifier, e.g. "prompt_style_v1".
        variants:      Mapping of variant_name → config dict.
        traffic_split: Mapping of variant_name → fraction (must sum to 1.0).
    """

    def __init__(
        self,
        test_id: str,
        variants: dict[str, dict],
        traffic_split: dict[str, float],
    ) -> None:
        assert abs(sum(traffic_split.values()) - 1.0) < 0.001, (
            f"[{test_id}] traffic_split must sum to 1.0, got {sum(traffic_split.values())}"
        )
        self.test_id = test_id
        self.variants = variants
        self.traffic_split = traffic_split
        log.info(
            "ab_test_initialized",
            test_id=test_id,
            variants=list(variants.keys()),
        )

    def assign_variant(self, session_id: str) -> str:
        """
        Return the variant name for a given session, deterministically.

        The same session_id always maps to the same variant so that
        conversation style is consistent throughout a session.
        """
        hash_int = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        bucket = (hash_int % 1_000_000) / 1_000_000.0

        cumulative = 0.0
        for variant_name, fraction in self.traffic_split.items():
            cumulative += fraction
            if bucket < cumulative:
                ab_test_assignments.labels(
                    test_id=self.test_id, variant=variant_name
                ).inc()
                log.info(
                    "ab_variant_assigned",
                    test_id=self.test_id,
                    session_id=session_id,
                    variant=variant_name,
                )
                return variant_name

        # Fallback — only reachable on floating-point edge cases
        return next(iter(self.variants))

    def get_variant_config(self, session_id: str) -> dict:
        """Return the config dict for the variant assigned to this session."""
        return self.variants[self.assign_variant(session_id)]


# ── Active tests ──────────────────────────────────────────────────────────────

PROMPT_STYLE_TEST = ABTest(
    test_id="prompt_style_v1",
    variants={
        "control": {
            "instruction": "",
        },
        "concise": {
            "instruction": (
                "Be concise and direct. Answer in 2–3 sentences unless "
                "more detail is explicitly requested."
            ),
        },
        "detailed": {
            "instruction": (
                "Provide detailed explanations with concrete examples. "
                "Break down complex topics step-by-step."
            ),
        },
    },
    traffic_split={
        "control": 0.5,
        "concise": 0.25,
        "detailed": 0.25,
    },
)


def get_active_tests() -> list[ABTest]:
    """Return the list of currently running A/B tests."""
    return [PROMPT_STYLE_TEST]

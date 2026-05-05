"""
Confidence scoring for generated responses.

A composite score that answers: "How much should we trust this answer?"

Formula (weighted sum):
    confidence = 0.4 × retrieval_score       (is the evidence good?)
               + 0.3 × citation_score        (is the answer grounded?)
               + 0.2 × eval_score_normalised (did the LLM judge like it?)
               + 0.1 × source_diversity      (is evidence spread across sources?)

Thresholds:
    high   (≥ 0.80)  — serve directly; no approval needed for read-only actions
    medium (≥ 0.50)  — serve with optional review flag
    low    (< 0.50)  — flag for human review; block write/delete actions

Why these weights?
- Retrieval quality is the single biggest predictor of answer quality in RAG.
  A retrieval score < 0.4 almost always produces an unsatisfying answer.
- Citation grounding catches confabulation: if the answer is not backed by
  any chunk, something went wrong.
- Eval score and source diversity are supporting signals, not primary ones.

Action approval policy (for future tool-calling agents):
    read    — approval only when confidence < 0.3 (clearly broken retrieval)
    write   — approval unless confidence ≥ 0.9    (very high bar)
    delete  — always requires human approval       (irreversible)
"""

from src.observability.logging_config import get_logger

log = get_logger(__name__)


def calculate_confidence(
    retrieval_score: float,
    citation_valid: bool,
    eval_score: float | None = None,
    source_count: int = 0,
) -> float:
    """
    Compute a composite confidence score in [0, 1].

    Args:
        retrieval_score: Top chunk similarity score from retrieval (0–1).
        citation_valid:  True if citation validation passed (or was skipped).
        eval_score:      Optional LLM judge score (1–5 scale). Pass None to
                         use a neutral mid-point assumption.
        source_count:    Number of distinct source documents cited.

    Returns:
        Composite confidence score clamped to [0.0, 1.0].
    """
    # Retrieval component (40 %)
    retrieval_component = retrieval_score * 0.4

    # Citation component (30 %)
    citation_component = (1.0 if citation_valid else 0.3) * 0.3

    # Eval component (20 %) — neutral 0.5 when no judge score is available
    if eval_score is not None:
        eval_normalised = (eval_score - 1) / 4.0   # map 1–5 → 0–1
        eval_component = eval_normalised * 0.2
    else:
        eval_component = 0.5 * 0.2

    # Source diversity component (10 %) — capped at 5 sources
    diversity = min(source_count / 5.0, 1.0)
    source_component = diversity * 0.1

    confidence = retrieval_component + citation_component + eval_component + source_component
    confidence = max(0.0, min(1.0, confidence))

    log.info(
        "confidence_calculated",
        confidence=round(confidence, 3),
        retrieval_score=retrieval_score,
        citation_valid=citation_valid,
        source_count=source_count,
    )

    return confidence


def get_confidence_level(confidence: float) -> str:
    """Map a confidence score to a human-readable level: 'high' | 'medium' | 'low'."""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"


def requires_approval(confidence: float, action_type: str = "read") -> bool:
    """
    Return True if the action requires human approval.

    Args:
        confidence:   Composite confidence score (0–1).
        action_type:  One of 'read' | 'write' | 'delete'.

    Policy:
        read   — only flag if confidence < 0.3 (clearly broken)
        write  — flag unless confidence ≥ 0.9 (very high bar)
        delete — always flag (irreversible operation)
    """
    if action_type == "read":
        return confidence < 0.3
    elif action_type == "write":
        return confidence < 0.9
    elif action_type == "delete":
        return True
    return True  # unknown action types are blocked by default

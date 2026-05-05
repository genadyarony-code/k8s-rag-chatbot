"""
Advanced prompt engineering utilities.

These are drop-in additions to the standard build_prompt() flow. Each
technique addresses a specific failure mode:

- Chain-of-thought (CoT): prevents shallow one-sentence answers on
  complex K8s topics by forcing the model to reason step-by-step first.
  Best for: multi-component questions, debugging scenarios.

- Few-shot examples: anchors the model to the expected answer format
  (concise, technical, with kubectl commands). Especially useful for
  new A/B test variants where the base system prompt changes.
  Cost: +~100 tokens per example pair.

- Self-consistency: generates N independent answers in one call and
  asks the model to consolidate them. Improves factual accuracy on
  ambiguous questions without running N separate API calls.
  Best for: "What is the difference between X and Y?" type queries.

All functions are pure (no I/O, no state) and can be freely mixed.
"""

from src.observability.logging_config import get_logger

log = get_logger(__name__)

# ── Chain-of-Thought ──────────────────────────────────────────────────────────

_COT_INSTRUCTION = """\
Before giving your final answer, think through the problem step by step:

1. What is the user specifically asking?
2. Which Kubernetes concepts or components are involved?
3. How do they relate or interact?
4. Are there common pitfalls or misconceptions to address?

Then provide a clear, structured answer based on your reasoning."""


def add_chain_of_thought(system_prompt: str) -> str:
    """
    Append a chain-of-thought instruction to `system_prompt`.

    Args:
        system_prompt: The base system prompt string.

    Returns:
        Augmented system prompt with CoT instruction appended.
    """
    return f"{system_prompt}\n\n{_COT_INSTRUCTION}"


# ── Few-Shot Examples ─────────────────────────────────────────────────────────

_K8S_FEW_SHOT_EXAMPLES: list[dict] = [
    {
        "question": "What is a Pod?",
        "answer": (
            "A Pod is the smallest deployable unit in Kubernetes. It groups one "
            "or more containers that share network namespace and storage volumes. "
            "Pods are ephemeral — when they terminate they are not restarted in "
            "place; a controller (Deployment, StatefulSet) creates a replacement."
        ),
    },
    {
        "question": "How do I debug a CrashLoopBackOff?",
        "answer": (
            "CrashLoopBackOff means the container starts, crashes, and Kubernetes "
            "keeps restarting it with exponential back-off. Typical steps:\n"
            "1. `kubectl logs <pod> --previous` — inspect the last crash output.\n"
            "2. Check the exit code in `kubectl describe pod <pod>`; exit 137 = OOMKilled.\n"
            "3. Verify the image exists and the registry is accessible.\n"
            "4. Confirm the container command/args are correct.\n"
            "5. Review resource limits — raise memory if the pod is OOMKilled."
        ),
    },
]


def build_few_shot_messages(
    question: str,
    examples: list[dict] | None = None,
) -> list[dict]:
    """
    Prepend few-shot Q&A examples to the message list.

    The examples are injected before the real question so the model sees
    the desired response style before having to answer.

    Args:
        question: The live user question.
        examples: Optional custom examples; defaults to _K8S_FEW_SHOT_EXAMPLES.

    Returns:
        Message list: [user, assistant, user, assistant, ..., user(question)]
    """
    if examples is None:
        examples = _K8S_FEW_SHOT_EXAMPLES

    messages: list[dict] = []
    for ex in examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})

    messages.append({"role": "user", "content": question})
    return messages


# ── Self-Consistency ──────────────────────────────────────────────────────────

def self_consistency_prompt(question: str, n_samples: int = 3) -> str:
    """
    Build a prompt that asks the model to generate N independent answers and
    then synthesise the best consolidated response.

    This is a single-call approximation of the full self-consistency technique
    (which normally requires N separate API calls). The model is asked to
    produce diversity within one generation, then self-select the best parts.

    Args:
        question:  The user's question.
        n_samples: Number of draft answers to generate before consolidating.

    Returns:
        A prompt string to use as the user message.
    """
    return (
        f"Generate {n_samples} independent, concise answers to the question below. "
        f"Then analyse them and write the single best consolidated answer.\n\n"
        f"Question: {question}\n\n"
        + "\n".join(f"Draft {i + 1}: ..." for i in range(n_samples))
        + "\n\nBest consolidated answer:"
    )


def detect_complex_query(question: str) -> bool:
    """
    Heuristic to decide whether a query benefits from chain-of-thought.

    Returns True for comparison, multi-component, or debugging questions.
    """
    import re
    patterns = [
        r"\b(difference|compare|vs\.?|versus|better)\b",
        r"\b(why|how does|explain|what causes)\b",
        r"\bdebug|troubleshoot|fix|diagnose\b",
    ]
    lowered = question.lower()
    return any(re.search(p, lowered) for p in patterns)

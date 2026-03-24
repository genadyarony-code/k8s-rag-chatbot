"""
The system prompt keeps the model grounded: answer only from the provided context,
cite sources, admit when something isn't covered. For a technical chatbot, a
confident wrong answer is worse than an honest "I don't have that information".

build_prompt() injects the retrieved chunks as a fake user→assistant exchange
before the real question. The model "confirms" it read the docs, which in practice
produces more grounded answers than dumping context into the system message.

A/B testing: if `session_id` is provided, the active A/B test may append an
extra instruction to the system prompt. The same session always receives the
same variant (deterministic MD5 bucketing) so conversation style is consistent.
"""

from src.evaluation.ab_testing import PROMPT_STYLE_TEST

# The system prompt is intentionally strict about grounding — the model must
# cite only what's in the provided context and admit uncertainty rather than
# hallucinate plausible-sounding but incorrect Kubernetes information.
_BASE_SYSTEM_PROMPT = """You are a Kubernetes expert assistant.

Your role:
- Answer questions about Kubernetes based ONLY on the provided context
- If the context doesn't contain enough information, say so explicitly
- Always cite your sources (document name and section)
- Be concise and technical — the user is a developer

Rules:
- Never make up information not in the context
- If asked something outside Kubernetes, redirect politely
- Use code examples when relevant
"""


def build_prompt(
    question: str,
    context: list[dict],
    history: list[dict],
    session_id: str = "",
) -> list[dict]:
    """
    Assemble the full message list: system → context injection → history → question.

    Args:
        question:   Validated user question.
        context:    Retrieved chunks (each has content, source, section_title).
        history:    Recent conversation turns (up to 6 messages).
        session_id: Used for A/B variant assignment. Empty string → control variant.
    """
    system_prompt = _BASE_SYSTEM_PROMPT

    # A/B test: append a style instruction from the assigned variant
    if session_id:
        variant_cfg = PROMPT_STYLE_TEST.get_variant_config(session_id)
        extra = variant_cfg.get("instruction", "")
        if extra:
            system_prompt = system_prompt.rstrip() + f"\n\nStyle instruction: {extra}\n"

    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if context:
        context_text = "\n\n---\n\n".join([
            f"Source: {c['source']} | Section: {c['section_title']}\n{c['content']}"
            for c in context
        ])
        messages.append({
            "role": "user",
            "content": f"Here is the relevant documentation:\n\n{context_text}",
        })
        messages.append({
            "role": "assistant",
            "content": "I have reviewed the provided documentation. I'm ready to answer your question.",
        })

    # history — last 6 messages (3 pairs)
    for msg in history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": question})
    return messages

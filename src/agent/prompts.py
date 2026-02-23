"""
The system prompt keeps the model grounded, answer only from the provided context,
cite sources, admit when something isn't covered. For a technical chatbot, a
confident wrong answer is worse than an honest "I don't have that information".

build_prompt() injects the retrieved chunks as a fake user→assistant exchange
before the real question. The model "confirms" it read the docs, which in practice
produces more grounded answers than dumping context into the system message.
"""

# The system prompt is intentionally strict about grounding — the model must
# cite only what's in the provided context and admit uncertainty rather than
# hallucinate plausible-sounding but incorrect Kubernetes information.
SYSTEM_PROMPT = """You are a Kubernetes expert assistant.

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


def build_prompt(question: str, context: list[dict], history: list[dict]) -> list[dict]:
    """Assembles the full message list: system → context injection → history → question."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        context_text = "\n\n---\n\n".join([
            f"Source: {c['source']} | Section: {c['section_title']}\n{c['content']}"
            for c in context
        ])
        messages.append({
            "role": "user",
            "content": f"Here is the relevant documentation:\n\n{context_text}"
        })
        messages.append({
            "role": "assistant",
            "content": "I have reviewed the provided documentation. I'm ready to answer your question."
        })

    # history — last 6 messages (3 pairs)
    for msg in history[-6:]:
        messages.append(msg)

    messages.append({"role": "user", "content": question})
    return messages

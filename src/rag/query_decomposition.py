"""
Query decomposition for complex multi-hop questions.

Why decompose?
A question like "What's the difference between Deployments and StatefulSets?"
needs documents about both resources. A single embedding for the whole question
will land in a blend of both topics and may miss relevant chunks from either.
Decomposing into two targeted sub-queries and merging the results is strictly
better.

Strategy:
1. Cheap regex heuristic (`should_decompose`) to gate LLM usage — only
   comparison / multi-part queries are candidates.
2. GPT-4o-mini extracts up to 3 independent sub-questions (< 200 tokens).
3. The calling code runs retrieval for each sub-query and deduplicates results.

Cost: ~$0.000030 per decomposed query (200 tokens × $0.15/M input).
"""

import re
from typing import Optional

from openai import OpenAI

from src.config.settings import settings
from src.observability.logging_config import get_logger
from src.observability.metrics import query_decomposition_count

log = get_logger(__name__)

_DECOMPOSITION_PROMPT = """\
You are a query decomposition assistant for a Kubernetes knowledge base.

Given a complex question, break it down into simpler sub-questions that can
be answered independently from the knowledge base.

Rules:
- Only decompose if the question has multiple distinct parts or requires a comparison.
- Each sub-question must be self-contained.
- Return at most 3 sub-questions.
- If the question is already simple, return it unchanged on a single line.

Examples:

Question: "What is the difference between Deployments and StatefulSets?"
1. What are Kubernetes Deployments and how do they work?
2. What are Kubernetes StatefulSets and how do they work?

Question: "What causes ImagePullBackOff and how do I fix it?"
1. What causes ImagePullBackOff in Kubernetes?
2. How do I fix ImagePullBackOff in Kubernetes?

Question: "What is a Pod?"
1. What is a Pod?

Now decompose:
{question}

Return ONLY the numbered sub-questions, one per line."""

_COMPARISON_RE = re.compile(
    r"\b(difference\s+between|compare|vs\.?|versus)\b",
    re.IGNORECASE,
)


def should_decompose(query: str) -> bool:
    """
    Cheap heuristic: return True if the query looks complex enough to decompose.

    Triggers:
    - Explicit comparison language ("difference between", "vs", "compare").
    - More than one question mark.
    - Long query (> 50 chars) that contains " and ".
    """
    if _COMPARISON_RE.search(query):
        return True
    if query.count("?") > 1:
        return True
    if " and " in query.lower() and len(query) > 50:
        return True
    return False


def decompose_query(query: str) -> list[str]:
    """
    Decompose a complex query into independent sub-queries via LLM.

    Returns the original query unchanged if decomposition is not needed or
    if the LLM call fails.

    Args:
        query: User question.

    Returns:
        List of sub-queries (length 1 for simple questions).
    """
    if not should_decompose(query):
        log.info("query_decomposition_skipped", reason="heuristic_not_triggered")
        return [query]

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = _DECOMPOSITION_PROMPT.format(question=query)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        sub_queries: list[str] = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading numbering: "1. ", "2) ", etc.
            if line and line[0].isdigit():
                line = re.sub(r"^\d+[.)]\s*", "", line)
            if line:
                sub_queries.append(line)

        if not sub_queries:
            return [query]

        log.info(
            "query_decomposed",
            original=query[:100],
            sub_queries=sub_queries,
            count=len(sub_queries),
        )
        query_decomposition_count.labels(sub_query_count=str(len(sub_queries))).inc()
        return sub_queries

    except Exception as exc:
        log.error("query_decomposition_failed", error=str(exc))
        return [query]

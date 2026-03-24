"""
Citation validation using sentence-level semantic similarity.

Why validate?
LLMs occasionally confabulate facts that sound plausible but are not
grounded in the retrieved context. Checking each sentence of the answer
against the top chunks catches obvious hallucinations before they reach
the user.

Strategy:
1. Split the generated answer into sentences.
2. Embed each sentence and all chunk texts with a lightweight local model
   (all-MiniLM-L6-v2, ~80 MB).
3. Flag any sentence whose maximum cosine similarity to any chunk is below
   the threshold (default 0.5).
4. Log flagged sentences as warnings; optionally append a disclaimer.

Trade-offs:
- Adds ~50 ms per response (model is cached after first load).
- False-positive rate is low for factual K8s answers but can occur for
  meta-sentences like "In summary, ...".
- Disabled by default (FF_USE_CITATION_VALIDATION=false) because it
  requires the sentence-transformers / torch stack (~2 GB).

Set FF_USE_CITATION_VALIDATION=true to enable.
"""

from typing import Optional

from src.observability.logging_config import get_logger
from src.observability.metrics import citation_validation_failures

log = get_logger(__name__)


class CitationValidator:
    """
    Sentence-level grounding checker.

    The SentenceTransformer model is loaded lazily on first use so that
    importing this module does not fail if sentence-transformers is not
    installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Delay import to keep the module importable without torch installed
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        self.model = SentenceTransformer(model_name)
        log.info("citation_validator_initialized", model=model_name)

    def validate(
        self,
        answer: str,
        chunks: list[dict],
        threshold: float = 0.5,
    ) -> tuple[bool, list[str]]:
        """
        Check that every sentence in `answer` is supported by at least one chunk.

        Args:
            answer:    Generated answer text.
            chunks:    Retrieved context chunks (must contain "content" key).
            threshold: Minimum cosine similarity to consider a sentence grounded.

        Returns:
            (is_valid, unsupported_sentences)
            is_valid=False means at least one sentence is below threshold.
        """
        from sentence_transformers import util  # type: ignore[import]

        if not chunks:
            return False, [answer]

        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not sentences:
            return True, []

        chunk_texts = [c["content"] for c in chunks]

        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        chunk_embeddings = self.model.encode(chunk_texts, convert_to_tensor=True)

        unsupported: list[str] = []
        for i, sentence in enumerate(sentences):
            sims = util.cos_sim(sentence_embeddings[i], chunk_embeddings)[0]
            max_sim = float(sims.max())

            if max_sim < threshold:
                unsupported.append(sentence)
                log.warning(
                    "unsupported_claim_detected",
                    sentence_preview=sentence[:100],
                    max_similarity=round(max_sim, 3),
                    threshold=threshold,
                )

        is_valid = len(unsupported) == 0

        if not is_valid:
            citation_validation_failures.inc()

        log.info(
            "citation_validation_completed",
            total_sentences=len(sentences),
            unsupported_count=len(unsupported),
            is_valid=is_valid,
        )

        return is_valid, unsupported


# ── Lazy singleton ────────────────────────────────────────────────────────────

_validator: Optional[CitationValidator] = None


def get_validator() -> CitationValidator:
    """Get or create the global citation validator."""
    global _validator
    if _validator is None:
        _validator = CitationValidator()
    return _validator

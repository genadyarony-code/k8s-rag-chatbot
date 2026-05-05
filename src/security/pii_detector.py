"""
PII (Personally Identifiable Information) detection and redaction.

Uses Microsoft Presidio to detect and redact:
- Credit card numbers
- US Social Security Numbers
- Phone numbers
- Email addresses
- IP addresses
- Person names
- Location / address strings

Why redact before sending to OpenAI?
- GDPR / CCPA compliance: PII should not leave the user's control
- OpenAI terms of service: do not send sensitive personal data
- Audit trail: log what was redacted (type + position), never the value itself

The detector is a lazy singleton — it takes ~2s to initialize (spaCy model load)
and should only be created once per process lifetime.
"""

from typing import Optional

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.observability.logging_config import get_logger

log = get_logger(__name__)

# PII entity types to detect
_ENTITIES = [
    "CREDIT_CARD",
    "US_SSN",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IP_ADDRESS",
    "PERSON",
    "LOCATION",
]


class PIIDetector:
    """
    PII detector backed by Microsoft Presidio.

    Usage:
        detector = PIIDetector()
        clean_text, findings = detector.detect_and_redact(user_input)
    """

    def __init__(self) -> None:
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        log.info("pii_detector_initialized")

    def detect_and_redact(
        self,
        text: str,
        language: str = "en",
    ) -> tuple[str, list]:
        """
        Detect and redact PII from text.

        Args:
            text: Input text to analyze
            language: Language code (default: "en")

        Returns:
            (redacted_text, findings)
            findings is a list of dicts: {type, start, end, score}

        Example:
            >>> detector = PIIDetector()
            >>> clean, findings = detector.detect_and_redact(
            ...     "My SSN is 123-45-6789 and email is user@example.com"
            ... )
            >>> print(clean)
            "My SSN is <US_SSN> and email is <EMAIL_ADDRESS>"
            >>> print(findings)
            [{"type": "US_SSN", ...}, {"type": "EMAIL_ADDRESS", ...}]
        """
        results = self.analyzer.analyze(
            text=text,
            language=language,
            entities=_ENTITIES,
        )

        if not results:
            return text, []

        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
        )

        findings = [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
            }
            for r in results
        ]

        log.warning(
            "pii_detected_and_redacted",
            findings_count=len(findings),
            entity_types=[f["type"] for f in findings],
        )

        return anonymized.text, findings


# ── Lazy singleton ────────────────────────────────────────────────────────────

_detector: Optional[PIIDetector] = None


def get_pii_detector() -> PIIDetector:
    """
    Get or create the global PII detector instance.
    Initialization is deferred to first use (spaCy model load takes ~2s).
    """
    global _detector
    if _detector is None:
        _detector = PIIDetector()
    return _detector

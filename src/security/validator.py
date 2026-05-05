"""
Consolidated input validation pipeline for the chat endpoint.

Validation order matters:
1. Sanitize first (cheap, reduces attack surface for later steps)
2. Prompt injection check (regex, sub-millisecond)
3. PII detection and redaction (spaCy, ~50-100ms)
4. Rate limiting is handled at the FastAPI middleware level (not here)

Each step is guarded by a settings flag so individual checks can be disabled
without redeploying (e.g. for debugging or high-throughput load testing).
"""

from fastapi import HTTPException

from src.config.settings import settings
from src.observability.logging_config import get_logger
from src.security.prompt_injection import detect_prompt_injection, sanitize_input

log = get_logger(__name__)


def validate_chat_input(
    user_input: str,
    session_id: str,
    max_length: int = 10000,
) -> str:
    """
    Validate and sanitize user input for the chat endpoint.

    Args:
        user_input: Raw user question
        session_id: Session identifier (used for logging)
        max_length: Maximum allowed input length in characters

    Returns:
        Validated and sanitized input (PII redacted when detection is enabled)

    Raises:
        HTTPException(413): Input exceeds max_length
        HTTPException(400): Input contains a prompt injection attempt
    """
    # ── Step 1: Sanitize ──────────────────────────────────────────────────────
    try:
        clean_input = sanitize_input(user_input, max_length=max_length)
    except ValueError as e:
        log.warning(
            "input_validation_failed",
            session_id=session_id,
            reason="oversized",
            length=len(user_input),
        )
        raise HTTPException(status_code=413, detail=str(e))

    # ── Step 2: Prompt injection detection ───────────────────────────────────
    if settings.security_enable_prompt_injection_check:
        is_malicious, reason = detect_prompt_injection(clean_input)
        if is_malicious:
            log.error(
                "prompt_injection_blocked",
                session_id=session_id,
                reason=reason,
                input_preview=clean_input[:100],
            )
            raise HTTPException(
                status_code=400,
                detail="Input rejected: potential prompt injection detected",
            )

    # ── Step 3: PII detection and redaction ───────────────────────────────────
    if settings.security_enable_pii_detection:
        from src.security.pii_detector import get_pii_detector
        detector = get_pii_detector()
        redacted_input, findings = detector.detect_and_redact(clean_input)

        if findings:
            log.warning(
                "pii_redacted_from_input",
                session_id=session_id,
                pii_types=[f["type"] for f in findings],
                redaction_count=len(findings),
            )
        clean_input = redacted_input

    log.info(
        "input_validation_passed",
        session_id=session_id,
        original_length=len(user_input),
        final_length=len(clean_input),
    )

    return clean_input

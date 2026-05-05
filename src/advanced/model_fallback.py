"""
Model fallback cascade for resilience.

When the primary model (gpt-4o-mini) is unavailable — due to rate limits,
service errors, or a tripped circuit breaker — the handler automatically
tries the next model in the cascade without any changes to the calling code.

Default cascade (OpenAI-only):
    1. gpt-4o-mini      — primary: fast, cheap, high quality
    2. gpt-3.5-turbo    — fallback 1: faster, slightly cheaper
    (3. claude-3-haiku  — fallback 2: requires ANTHROPIC_API_KEY in settings)

Error handling:
- RateLimitError (429) → try next model immediately
- APIStatusError  (503, 529) → try next model immediately
- Any other exception → log as error, try next model

Metrics:
- model_fallback_attempts_total{from_model, to_model, reason} incremented
  each time a fallback transition occurs.

Usage:
    handler = get_fallback_handler()
    result = handler.call_with_fallback(messages, temperature=0.1, session_id=sid)
    # result = {"content": str, "model": str, "provider": str,
    #            "input_tokens": int, "output_tokens": int, "total_tokens": int}
"""

from typing import Optional

from openai import OpenAI, RateLimitError, APIStatusError

from src.config.settings import settings
from src.observability.logging_config import get_logger
from src.observability.metrics import model_fallback_attempts

log = get_logger(__name__)

# OpenAI model cascade — always available when OPENAI_API_KEY is set
_OPENAI_CASCADE = ["gpt-4o-mini", "gpt-3.5-turbo"]


class ModelFallbackHandler:
    """
    Transparent LLM caller that falls back through a model cascade on failure.

    Thread-safe: the OpenAI client uses connection pooling internally.
    The Anthropic client is created lazily only if the package is installed
    and ANTHROPIC_API_KEY is provided.
    """

    def __init__(self) -> None:
        self._openai = OpenAI(api_key=settings.openai_api_key)

        # Build cascade: [(model_name, call_fn), ...]
        self._cascade: list[tuple[str, object]] = [
            (m, "openai") for m in _OPENAI_CASCADE
        ]

        # Optional Anthropic fallback (lazy-imported to avoid hard dependency)
        if settings.anthropic_api_key:
            try:
                from anthropic import Anthropic  # type: ignore[import]
                self._anthropic = Anthropic(api_key=settings.anthropic_api_key)
                self._cascade.append(("claude-3-haiku-20240307", "anthropic"))
                log.info("anthropic_fallback_enabled")
            except ImportError:
                log.warning(
                    "anthropic_package_missing",
                    hint="pip install anthropic to enable Claude fallback",
                )
                self._anthropic = None
        else:
            self._anthropic = None

        log.info("model_fallback_initialized", cascade=[m for m, _ in self._cascade])

    def _call_openai(
        self, model: str, messages: list, temperature: float, max_tokens: int
    ) -> dict:
        response = self._openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "model": model,
            "provider": "openai",
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    def _call_anthropic(
        self, model: str, messages: list, temperature: float, max_tokens: int
    ) -> dict:
        # Convert OpenAI message format to Anthropic's format
        system_text = ""
        anthropic_msgs = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"]
            else:
                anthropic_msgs.append({"role": msg["role"], "content": msg["content"]})

        response = self._anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_text or None,
            messages=anthropic_msgs,
        )
        total = response.usage.input_tokens + response.usage.output_tokens
        return {
            "content": response.content[0].text,
            "model": model,
            "provider": "anthropic",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": total,
        }

    def call_with_fallback(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Call the first available model in the cascade.

        Args:
            messages:    OpenAI-format message list.
            temperature: Sampling temperature.
            max_tokens:  Maximum completion tokens.
            session_id:  For structured logging.

        Returns:
            Dict with keys: content, model, provider, input_tokens,
            output_tokens, total_tokens.

        Raises:
            RuntimeError if every model in the cascade fails.
        """
        prev_model: Optional[str] = None
        last_exc: Optional[Exception] = None

        for model, provider in self._cascade:
            try:
                log.info(
                    "model_fallback_attempting",
                    model=model,
                    provider=provider,
                    session_id=session_id,
                )

                if provider == "openai":
                    result = self._call_openai(model, messages, temperature, max_tokens)
                else:
                    result = self._call_anthropic(model, messages, temperature, max_tokens)

                if prev_model:
                    # We successfully fell back — record the transition
                    model_fallback_attempts.labels(
                        from_model=prev_model,
                        to_model=model,
                        reason=type(last_exc).__name__ if last_exc else "unknown",
                    ).inc()

                log.info(
                    "model_fallback_success",
                    model=model,
                    provider=provider,
                    total_tokens=result["total_tokens"],
                    session_id=session_id,
                )
                return result

            except (RateLimitError, APIStatusError) as exc:
                log.warning(
                    "model_rate_limited_or_unavailable",
                    model=model,
                    error=str(exc),
                    session_id=session_id,
                )
                prev_model = model
                last_exc = exc

            except Exception as exc:
                log.error(
                    "model_call_failed",
                    model=model,
                    error=str(exc),
                    session_id=session_id,
                )
                prev_model = model
                last_exc = exc

        raise RuntimeError(
            f"All models in cascade failed. Last error: {last_exc}"
        )


# ── Lazy singleton ────────────────────────────────────────────────────────────

_handler: Optional[ModelFallbackHandler] = None


def get_fallback_handler() -> ModelFallbackHandler:
    """Get or create the global model fallback handler."""
    global _handler
    if _handler is None:
        _handler = ModelFallbackHandler()
    return _handler

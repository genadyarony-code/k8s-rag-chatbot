"""
Structured logging configuration using structlog.

All logs are emitted as JSON with consistent fields:
- timestamp (ISO 8601)
- level (debug/info/warning/error)
- logger (module name)
- event (human-readable message)
- context (structured data)

Example output:
{
  "timestamp": "2026-03-24T10:30:45.123Z",
  "level": "info",
  "logger": "src.agent.nodes",
  "event": "chat_request",
  "session_id": "abc123",
  "tokens": 1234,
  "cost_usd": 0.000185
}
"""

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Set up structlog for the entire application.

    Call this once at application startup (in FastAPI lifespan).
    After calling this, all loggers will emit structured JSON.

    Args:
        log_level: Minimum log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )


def get_logger(name: str) -> Any:
    """
    Get a structured logger for a module.

    Usage:
        log = get_logger(__name__)
        log.info("user_login", user_id="abc123", method="api_key")

    Args:
        name: Usually __name__ to get the module path

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)

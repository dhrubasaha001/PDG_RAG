"""
backend/core/logging.py
Structured JSON logging via structlog.
Every component gets a bound logger with a 'component' field.

Usage:
    from backend.core.logging import get_logger
    logger = get_logger("ingestion.pdf_parser")
    logger.info("pdf_parsed", page_count=12, doc_id="abc123")
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def _configure_structlog(log_level: str = "INFO", log_format: str = "json") -> None:
    """Configure structlog once at application startup."""

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Avoid adding duplicate handlers on repeated calls (e.g. during tests)
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)

    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog BoundLogger pre-bound with a 'component' field.

    Args:
        component: Dot-separated component name, e.g. 'ingestion.pdf_parser'

    Returns:
        BoundLogger instance ready to use.
    """
    return structlog.get_logger(component).bind(component=component)


# ---------------------------------------------------------------------------
# Bootstrap logging as soon as this module is imported.
# Settings are read lazily to avoid circular imports.
# ---------------------------------------------------------------------------
def _bootstrap() -> None:
    try:
        from backend.core.config import settings  # noqa: PLC0415
        _configure_structlog(settings.LOG_LEVEL, settings.LOG_FORMAT)
    except Exception:
        # Fallback: plain INFO JSON logging if config is not yet available
        _configure_structlog("INFO", "json")


_bootstrap()

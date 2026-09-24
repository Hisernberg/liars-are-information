"""Structured logging.

Every phase logs through here; nothing in this codebase uses ``print``.  Logs go
to stderr (human-readable) and, when a run directory is configured, to a
JSON-lines file under ``results/logs/`` so a run's log can be joined back to its
manifest by ``run_id``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    run_id: str | None = None,
    force: bool = False,
) -> structlog.stdlib.BoundLogger:
    """Configure structlog once per process and return the root logger."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return structlog.get_logger()

    handlers: list[logging.Handler] = []

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        )
    )
    handlers.append(stderr_handler)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        name = f"{run_id or 'run'}.jsonl"
        file_handler = logging.FileHandler(log_dir / name, encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(processor=structlog.processors.JSONRenderer())
        )
        handlers.append(file_handler)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper()))

    # Third-party libraries log every HTTP request at INFO; that noise would
    # swamp the JSONL run log that manifests are joined against.
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "filelock",
        "fsspec",
        "datasets",
        "matplotlib",
        "fontTools",
        "fontTools.subset",
        "PIL",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True
    return structlog.get_logger()


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring defaults on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name) if name else structlog.get_logger()

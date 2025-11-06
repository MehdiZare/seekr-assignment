"""Centralized JSON logging configuration for CloudWatch compatibility."""

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Context variable to store session ID across async calls
session_context: ContextVar[Optional[str]] = ContextVar("session_context", default=None)


class CloudWatchJsonFormatter(logging.Formatter):
    """Custom JSON formatter for CloudWatch with additional context fields."""

    STANDARD_ATTRS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log records as JSON for CloudWatch ingestion."""
        timestamp = (
            datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

        log_record: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": os.getenv("ENVIRONMENT", "development"),
        }

        # Prefer explicit session_id on the record, fall back to context.
        session_id = getattr(record, "session_id", None) or session_context.get()
        if session_id:
            log_record["session_id"] = session_id

        # Include any custom extra fields that were attached to the record.
        for key, value in record.__dict__.items():
            if key in self.STANDARD_ATTRS or key == "session_id":
                continue
            if value is None:
                continue
            log_record[key] = value

        # Add exception information if available.
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(log_record, default=self._json_default)

    @staticmethod
    def _json_default(value: Any) -> Any:
        """Provide JSON-safe fallbacks for non-serializable objects."""
        if isinstance(value, (set, frozenset)):
            return list(value)
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        try:
            return str(value)
        except Exception:
            return repr(value)


def setup_json_logging(level: str = "INFO") -> None:
    """
    Configure JSON logging for the application.

    Args:
        level: Logging level (default: INFO)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Reconfigure stdout to be line-buffered for immediate output
    sys.stdout.reconfigure(line_buffering=True)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(CloudWatchJsonFormatter())

    # Force immediate flush after each log record
    original_emit = console_handler.emit
    def flush_emit(record):
        original_emit(record)
        console_handler.flush()
    console_handler.emit = flush_emit

    logging.basicConfig(level=log_level, handlers=[console_handler], force=True)
    logging.captureWarnings(True)

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Force immediate flush after setup
    sys.stdout.flush()


def generate_session_id() -> str:
    """
    Generate a unique session ID for tracking requests.

    Returns:
        UUID string for session tracking
    """
    return str(uuid.uuid4())


def set_session_context(session_id: str) -> None:
    """
    Set the session ID in the context variable.

    Args:
        session_id: Session ID to set in context
    """
    session_context.set(session_id)


def get_session_context() -> Optional[str]:
    """
    Get the current session ID from context.

    Returns:
        Session ID or None if not set
    """
    return session_context.get()


def clear_session_context() -> None:
    """Clear the session ID from context."""
    session_context.set(None)


class SessionLogger:
    """Logger wrapper that automatically includes session context."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize session logger.

        Args:
            logger: Base logger instance
        """
        self.logger = logger

    def _log(
        self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        """
        Internal log method that adds session context.

        Args:
            level: Log level
            msg: Log message
            extra: Extra fields to include
            **kwargs: Additional keyword arguments
        """
        if extra is None:
            extra = {}

        # Add session ID from context if not already in extra
        if "session_id" not in extra:
            session_id = get_session_context()
            if session_id:
                extra["session_id"] = session_id

        self.logger.log(level, msg, extra=extra, **kwargs)

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log INFO level message."""
        self._log(logging.INFO, msg, extra, **kwargs)

    def warning(
        self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        """Log WARNING level message."""
        self._log(logging.WARNING, msg, extra, **kwargs)

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log ERROR level message."""
        self._log(logging.ERROR, msg, extra, **kwargs)

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log DEBUG level message."""
        self._log(logging.DEBUG, msg, extra, **kwargs)


def get_logger(name: str) -> SessionLogger:
    """
    Get a session-aware logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        SessionLogger instance
    """
    base_logger = logging.getLogger(name)
    return SessionLogger(base_logger)


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, logger: SessionLogger, operation: str, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize timing context.

        Args:
            logger: Logger instance
            operation: Operation name for logging
            extra: Extra fields to include in log
        """
        self.logger = logger
        self.operation = operation
        self.extra = extra or {}
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        self.logger.info(f"{self.operation} started", extra=self.extra)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        duration_ms = int((time.time() - self.start_time) * 1000)
        log_extra = {**self.extra, "duration_ms": duration_ms}

        if exc_type is None:
            self.logger.info(
                f"{self.operation} completed",
                extra=log_extra,
            )
        else:
            log_extra["error_type"] = exc_type.__name__ if exc_type else None
            self.logger.error(
                f"{self.operation} failed",
                extra=log_extra,
            )

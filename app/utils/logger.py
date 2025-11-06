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


class CloudWatchJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for CloudWatch with additional context fields."""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        """Add custom fields to the log record for CloudWatch."""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format for CloudWatch
        log_record["timestamp"] = self.formatTime(record, "%Y-%m-%dT%H:%M:%S.%fZ")

        # Add log level
        log_record["level"] = record.levelname

        # Add logger name
        log_record["logger"] = record.name

        # Add session ID from context if available
        session_id = session_context.get()
        if session_id:
            log_record["session_id"] = session_id

        # Add environment info
        log_record["environment"] = os.getenv("ENVIRONMENT", "development")

        # Add any extra fields passed via extra parameter
        if hasattr(record, "session_id"):
            log_record["session_id"] = record.session_id
        if hasattr(record, "stage"):
            log_record["stage"] = record.stage
        if hasattr(record, "agent"):
            log_record["agent"] = record.agent
        if hasattr(record, "tool"):
            log_record["tool"] = record.tool
        if hasattr(record, "duration_ms"):
            log_record["duration_ms"] = record.duration_ms
        if hasattr(record, "iteration"):
            log_record["iteration"] = record.iteration
        if hasattr(record, "max_iterations"):
            log_record["max_iterations"] = record.max_iterations
        if hasattr(record, "transcript_length"):
            log_record["transcript_length"] = record.transcript_length
        if hasattr(record, "num_claims"):
            log_record["num_claims"] = record.num_claims
        if hasattr(record, "num_quotes"):
            log_record["num_quotes"] = record.num_quotes
        if hasattr(record, "num_topics"):
            log_record["num_topics"] = record.num_topics
        if hasattr(record, "search_results"):
            log_record["search_results"] = record.search_results
        if hasattr(record, "verification_status"):
            log_record["verification_status"] = record.verification_status
        if hasattr(record, "error_type"):
            log_record["error_type"] = record.error_type


def setup_json_logging(level: str = "INFO") -> None:
    """
    Configure JSON logging for the application.

    Args:
        level: Logging level (default: INFO)
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Remove any existing handlers
    root_logger.handlers = []

    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Create JSON formatter
    formatter = CloudWatchJsonFormatter(
        "%(timestamp)s %(level)s %(logger)s %(message)s",
        rename_fields={
            "levelname": "level",
            "name": "logger",
            "asctime": "timestamp",
        },
    )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


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

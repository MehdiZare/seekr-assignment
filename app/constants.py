"""Application constants for the Podcast Agent.

This module provides easy access to configuration constants from config.yaml.
"""

from app.config import get_config

# Lazy-load configuration
_config = None


def _get_config():
    """Get configuration instance (lazy-loaded)."""
    global _config
    if _config is None:
        _config = get_config()
    return _config


# Agent iteration limits
def get_max_supervisor_iterations() -> int:
    """Maximum iterations for supervisor agent."""
    return _get_config().app_settings.get("max_supervisor_iterations", 15)


def get_max_fact_check_iterations() -> int:
    """Maximum iterations for fact-checking agent."""
    return _get_config().app_settings.get("max_fact_check_iterations", 10)


# Network timeouts
def get_url_validation_timeout() -> int:
    """Timeout in seconds for URL validation requests."""
    return _get_config().app_settings.get("url_validation_timeout", 3)


# Workflow version
def get_workflow_version() -> str:
    """Current workflow version identifier."""
    return _get_config().app_settings.get("workflow_version", "v2")


# Model keys (for convenience)
SUPERVISOR_MODEL_KEY = "model_c"
FACT_CHECK_MODEL_KEY = "model_d"

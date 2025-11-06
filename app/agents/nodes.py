"""Node implementations for the LangGraph workflow."""

import json
import logging
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import ValidationError

from app.utils.logger import get_logger

logger = get_logger(__name__)

from app.config import get_config


@traceable(name="create_llm")
def _create_llm(model_key: str) -> ChatAnthropic | ChatOpenAI:
    """Create an LLM instance based on model configuration."""
    config = get_config()
    model_config = config.get_model_config(model_key)

    provider = model_config["provider"]
    model_name = model_config["name"]
    temperature = model_config.get("temperature", 0.3)
    max_tokens = model_config.get("max_tokens", 2000)

    if provider == "anthropic":
        return ChatAnthropic(
            api_key=config.get_api_key("anthropic"),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "llama":
        # Use OpenAI SDK with Llama API endpoint
        return ChatOpenAI(
            api_key=config.get_api_key("llama"),
            base_url="https://api.llama.com/compat/v1/",
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "openai":
        return ChatOpenAI(
            api_key=config.get_api_key("openai"),
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


@traceable(name="create_llm_with_fallback")
def _create_llm_with_fallback(model_key: str) -> tuple[ChatAnthropic | ChatOpenAI, ChatAnthropic | ChatOpenAI | None]:
    """Create primary and fallback LLM instances based on model configuration.

    Args:
        model_key: Model configuration key (e.g., "model_c", "model_d")

    Returns:
        Tuple of (primary_llm, fallback_llm or None if no fallback configured)
    """
    config = get_config()
    model_config = config.get_model_config(model_key)

    # Create primary LLM
    primary_provider = model_config["provider"]
    primary_model_name = model_config["name"]
    temperature = model_config.get("temperature", 0.3)
    max_tokens = model_config.get("max_tokens", 2000)

    logger.info(
        "Creating primary LLM",
        extra={
            "model_key": model_key,
            "provider": primary_provider,
            "model_name": primary_model_name,
            "stage": "llm_creation",
        },
    )

    if primary_provider == "anthropic":
        primary_llm = ChatAnthropic(
            api_key=config.get_api_key("anthropic"),
            model=primary_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif primary_provider == "llama":
        primary_llm = ChatOpenAI(
            api_key=config.get_api_key("llama"),
            base_url="https://api.llama.com/compat/v1/",
            model=primary_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif primary_provider == "openai":
        primary_llm = ChatOpenAI(
            api_key=config.get_api_key("openai"),
            model=primary_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown provider: {primary_provider}")

    # Create fallback LLM if configured
    fallback_llm = None
    if "fallback" in model_config:
        fallback_config = model_config["fallback"]
        fallback_provider = fallback_config["provider"]
        fallback_model_name = fallback_config["name"]
        fallback_temperature = fallback_config.get("temperature", temperature)
        fallback_max_tokens = fallback_config.get("max_tokens", max_tokens)

        logger.info(
            "Creating fallback LLM",
            extra={
                "model_key": model_key,
                "fallback_provider": fallback_provider,
                "fallback_model_name": fallback_model_name,
                "stage": "llm_creation_fallback",
            },
        )

        if fallback_provider == "anthropic":
            fallback_llm = ChatAnthropic(
                api_key=config.get_api_key("anthropic"),
                model=fallback_model_name,
                temperature=fallback_temperature,
                max_tokens=fallback_max_tokens,
            )
        elif fallback_provider == "llama":
            fallback_llm = ChatOpenAI(
                api_key=config.get_api_key("llama"),
                base_url="https://api.llama.com/compat/v1/",
                model=fallback_model_name,
                temperature=fallback_temperature,
                max_tokens=fallback_max_tokens,
            )
        elif fallback_provider == "openai":
            fallback_llm = ChatOpenAI(
                api_key=config.get_api_key("openai"),
                model=fallback_model_name,
                temperature=fallback_temperature,
                max_tokens=fallback_max_tokens,
            )
        else:
            raise ValueError(f"Unknown fallback provider: {fallback_provider}")

    return primary_llm, fallback_llm


@traceable(name="parse_json_response")
def _parse_json_response(response: str, model_class: type) -> Any:
    """Parse JSON response and validate with Pydantic model."""
    original_response = response

    try:
        # Try to extract JSON from markdown code blocks if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        # Sanitize the response to fix common JSON issues
        response = response.strip()

        # Try parsing with strict=False first (more lenient)
        try:
            data = json.loads(response, strict=False)
            return model_class.model_validate(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {e}. Attempting sanitization...")

            # Attempt to fix common issues
            # Replace control characters with escaped versions
            sanitized = response
            # Fix unescaped newlines in strings
            sanitized = re.sub(r'(?<!\\)\n', r'\\n', sanitized)
            # Fix unescaped tabs
            sanitized = re.sub(r'(?<!\\)\t', r'\\t', sanitized)
            # Fix unescaped carriage returns
            sanitized = re.sub(r'(?<!\\)\r', r'\\r', sanitized)

            # Try parsing sanitized version
            try:
                data = json.loads(sanitized, strict=False)
                logger.info("JSON parsing succeeded after sanitization")
                return model_class.model_validate(data)
            except json.JSONDecodeError as e2:
                logger.error(f"Sanitization failed: {e2}")
                logger.error(f"Original response (first 500 chars): {original_response[:500]}")
                logger.error(f"Sanitized response (first 500 chars): {sanitized[:500]}")
                raise ValueError(
                    f"Failed to parse JSON response even after sanitization. "
                    f"Error: {e2}. Check logs for full response."
                )

    except Exception as e:
        logger.error(f"Error parsing JSON response: {e}")
        logger.error(f"Full original response: {original_response}")
        raise


@traceable(name="format_validation_error")
def _format_validation_error(error: ValidationError) -> str:
    """Format a Pydantic ValidationError into a human-readable message for the LLM."""
    error_messages = []

    for err in error.errors():
        field = ".".join(str(x) for x in err["loc"])
        error_type = err["type"]
        msg = err["msg"]

        # Extract the actual and expected values if available
        ctx = err.get("ctx", {})

        if error_type == "string_too_long":
            max_length = ctx.get("max_length", "unknown")
            actual_value = err.get("input", "")
            actual_length = len(str(actual_value))
            error_messages.append(
                f"- Field '{field}': String is too long ({actual_length} characters). "
                f"Maximum allowed is {max_length} characters. Please shorten this field."
            )
        elif error_type == "string_too_short":
            min_length = ctx.get("min_length", "unknown")
            actual_length = len(str(err.get("input", "")))
            error_messages.append(
                f"- Field '{field}': String is too short ({actual_length} characters). "
                f"Minimum required is {min_length} characters. Please expand this field."
            )
        elif error_type == "too_short":
            min_length = ctx.get("min_length", "unknown")
            actual_length = len(err.get("input", []))
            error_messages.append(
                f"- Field '{field}': List has {actual_length} items. "
                f"Minimum required is {min_length} items. Please add more items."
            )
        else:
            error_messages.append(f"- Field '{field}': {msg}")

    return "\n".join(error_messages)


@traceable(name="auto_fix_validation")
def _auto_fix_validation(data: dict, error: ValidationError, model_class: type) -> Any:
    """
    Attempt to automatically fix validation errors by truncating or modifying fields.
    This is a fallback when all retry attempts fail.
    """
    logger.warning("Attempting auto-fix for validation errors")

    # Make a copy to avoid modifying original
    fixed_data = data.copy()

    for err in error.errors():
        field = err["loc"][0] if err["loc"] else None
        error_type = err["type"]

        if not field or field not in fixed_data:
            continue

        if error_type == "string_too_long":
            # Intelligently truncate at word boundary
            max_length = err.get("ctx", {}).get("max_length", 100)
            original_value = fixed_data[field]

            if len(original_value) > max_length:
                # Truncate at word boundary
                truncated = original_value[:max_length]
                last_space = truncated.rfind(" ")

                if last_space > max_length * 0.8:  # If we can save at least 80% of content
                    truncated = truncated[:last_space]

                # Add ellipsis if we truncated
                if len(truncated) < len(original_value):
                    truncated = truncated.rstrip() + "..."

                fixed_data[field] = truncated
                logger.warning(
                    f"Auto-truncated field '{field}' from {len(original_value)} "
                    f"to {len(truncated)} characters"
                )

        elif error_type == "string_too_short":
            # For too-short strings, we can't really auto-fix meaningfully
            # Just log a warning
            logger.warning(f"Cannot auto-fix string_too_short for field '{field}'")

        elif error_type == "too_short":
            # For lists that are too short, we can't add meaningful items
            logger.warning(f"Cannot auto-fix too_short list for field '{field}'")

    # Try to validate the fixed data
    try:
        return model_class.model_validate(fixed_data)
    except ValidationError as e:
        logger.error("Auto-fix failed, re-raising original error")
        raise error  # Raise original error if fix didn't work


@traceable(name="invoke_llm_with_validation_retry")
def _invoke_llm_with_validation_retry(
    llm: ChatAnthropic | ChatOpenAI,
    messages: list,
    model_class: type,
    max_retries: int = 3,
) -> Any:
    """
    Invoke LLM with automatic retry on Pydantic validation errors.

    Args:
        llm: The LLM instance to invoke
        messages: List of messages to send
        model_class: Pydantic model class for validation
        max_retries: Maximum number of retry attempts

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If all retries fail and auto-fix also fails
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            # Invoke LLM
            response = llm.invoke(messages)

            # Parse and validate response
            validated_output = _parse_json_response(response.content, model_class)

            # Success!
            if attempt > 0:
                logger.info(
                    f"Validation succeeded on attempt {attempt + 1}/{max_retries + 1}"
                )

            return validated_output

        except ValidationError as e:
            logger.warning(
                f"Validation error on attempt {attempt + 1}/{max_retries + 1}: "
                f"{len(e.errors())} error(s) found"
            )

            # Log structured validation error details
            for err in e.errors():
                field = ".".join(str(x) for x in err["loc"])
                logger.warning(f"  - Field '{field}': {err['msg']} (type: {err['type']})")

            # If this was the last attempt, try auto-fix
            if attempt >= max_retries:
                logger.warning(
                    f"All {max_retries + 1} attempts failed. Attempting auto-fix..."
                )

                # Try to extract the parsed data from the last response
                try:
                    # Re-parse without validation to get the raw data
                    response_text = response.content
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()

                    data = json.loads(response_text.strip(), strict=False)
                    return _auto_fix_validation(data, e, model_class)
                except Exception as fix_error:
                    logger.error(f"Auto-fix failed: {fix_error}")
                    raise e  # Re-raise original validation error

            # Format error message for LLM
            error_message = _format_validation_error(e)

            # Add error feedback to conversation
            error_prompt = f"""
Your previous response had validation errors. Please fix these issues and try again:

{error_message}

IMPORTANT: Make sure your response is valid JSON and follows ALL the schema constraints exactly.
"""

            messages.append(response)  # Add the failed response
            messages.append(HumanMessage(content=error_prompt))  # Add error feedback

            logger.info(f"Retrying with error feedback (attempt {attempt + 2}/{max_retries + 1})...")


@traceable(name="invoke_llm_with_failover")
def _invoke_llm_with_failover(
    model_key: str,
    messages: list,
    model_class: type,
    max_retries: int = 3,
) -> Any:
    """
    Invoke LLM with automatic failover from primary to fallback model on ANY error.

    This wraps _invoke_llm_with_validation_retry and adds provider-level failover.
    If the primary LLM fails for any reason (API error, rate limit, network error, etc.),
    it automatically retries with the fallback LLM if configured.

    Args:
        model_key: Model configuration key (e.g., "model_c", "model_d")
        messages: List of messages to send
        model_class: Pydantic model class for validation
        max_retries: Maximum number of retry attempts per model

    Returns:
        Validated Pydantic model instance

    Raises:
        Exception: If both primary and fallback models fail
    """
    # Create primary and fallback LLMs
    primary_llm, fallback_llm = _create_llm_with_fallback(model_key)

    # Get model config for logging
    config = get_config()
    model_config = config.get_model_config(model_key)
    primary_model_name = model_config["name"]
    primary_provider = model_config["provider"]

    # Try primary LLM first
    try:
        logger.info(
            "Invoking primary LLM",
            extra={
                "model_key": model_key,
                "provider": primary_provider,
                "model_name": primary_model_name,
                "stage": "llm_invoke_primary",
            },
        )

        result = _invoke_llm_with_validation_retry(
            llm=primary_llm,
            messages=messages,
            model_class=model_class,
            max_retries=max_retries,
        )

        logger.info(
            "Primary LLM succeeded",
            extra={
                "model_key": model_key,
                "provider": primary_provider,
                "model_name": primary_model_name,
                "failover_triggered": False,
                "stage": "llm_success_primary",
            },
        )

        return result

    except Exception as primary_error:
        # Primary LLM failed - try fallback if available
        if fallback_llm is None:
            logger.error(
                "Primary LLM failed and no fallback configured",
                extra={
                    "model_key": model_key,
                    "provider": primary_provider,
                    "model_name": primary_model_name,
                    "error_type": type(primary_error).__name__,
                    "error_message": str(primary_error),
                    "failover_triggered": False,
                    "stage": "llm_error_no_fallback",
                },
            )
            raise

        # Fallback is configured - try it
        fallback_provider = model_config["fallback"]["provider"]
        fallback_model_name = model_config["fallback"]["name"]

        logger.warning(
            "Primary LLM failed, attempting fallback",
            extra={
                "model_key": model_key,
                "primary_provider": primary_provider,
                "primary_model": primary_model_name,
                "fallback_provider": fallback_provider,
                "fallback_model": fallback_model_name,
                "error_type": type(primary_error).__name__,
                "error_message": str(primary_error),
                "failover_triggered": True,
                "failover_reason": type(primary_error).__name__,
                "stage": "llm_failover_start",
            },
        )

        try:
            result = _invoke_llm_with_validation_retry(
                llm=fallback_llm,
                messages=messages,
                model_class=model_class,
                max_retries=max_retries,
            )

            logger.info(
                "Fallback LLM succeeded",
                extra={
                    "model_key": model_key,
                    "fallback_provider": fallback_provider,
                    "fallback_model": fallback_model_name,
                    "failover_triggered": True,
                    "stage": "llm_success_fallback",
                },
            )

            return result

        except Exception as fallback_error:
            logger.error(
                "Both primary and fallback LLMs failed",
                extra={
                    "model_key": model_key,
                    "primary_provider": primary_provider,
                    "primary_model": primary_model_name,
                    "primary_error_type": type(primary_error).__name__,
                    "primary_error_message": str(primary_error),
                    "fallback_provider": fallback_provider,
                    "fallback_model": fallback_model_name,
                    "fallback_error_type": type(fallback_error).__name__,
                    "fallback_error_message": str(fallback_error),
                    "failover_triggered": True,
                    "stage": "llm_error_both_failed",
                },
            )
            # Re-raise the fallback error as it's the most recent
            raise fallback_error

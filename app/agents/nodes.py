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

logger = logging.getLogger(__name__)

from app.agents.prompts import (
    CRITIC_PROMPT,
    FACT_CHECKER_PROMPT,
    IMPROVED_FACT_CHECKER_PROMPT,
    PARALLEL_ANALYSIS_PROMPT,
    SUPERVISOR_PROMPT,
)
from app.agents.tools import create_search_tools, get_tool_descriptions, validate_and_filter_search_results
from app.config import get_config
from app.models.outputs import (
    CriticFeedback,
    FactCheckOutput,
    ParallelAnalysis,
    SupervisorOutput,
)
from app.models.state import AgentState, ModelResponses, FactCheckIteration


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


# Parallel Processing Nodes
@traceable(name="model_a_parallel_analysis")
def model_a_node(state: AgentState) -> dict:
    """Process transcript with Model A (Claude Haiku)."""
    logger.info("=" * 80)
    logger.info("MODEL A (Claude Haiku) - Starting parallel analysis")
    logger.info(f"Transcript length: {len(state['transcript'])} characters")

    llm = _create_llm("model_a")
    config = get_config()
    max_retries = config.app_settings.get("max_retries", 3)

    prompt = PARALLEL_ANALYSIS_PROMPT.format(transcript=state["transcript"])
    messages = [HumanMessage(content=prompt)]

    logger.info("Invoking Model A with validation retry (max retries: %d)", max_retries)

    # Use validation retry wrapper
    analysis = _invoke_llm_with_validation_retry(
        llm=llm,
        messages=messages,
        model_class=ParallelAnalysis,
        max_retries=max_retries,
    )

    logger.info(
        "Model A completed: %d topics, %d key points",
        len(analysis.topics),
        len(analysis.key_points),
    )
    logger.info("Topics identified: %s", ", ".join(analysis.topics[:3]) + ("..." if len(analysis.topics) > 3 else ""))

    # Return only the fields this node updates (partial state)
    return {
        "model_responses": ModelResponses(model_a=analysis),
        "messages": [
            f"Model A (Claude Haiku) analysis complete: "
            f"{len(analysis.topics)} topics, {len(analysis.key_points)} key points identified"
        ],
    }


@traceable(name="model_b_parallel_analysis")
def model_b_node(state: AgentState) -> dict:
    """Process transcript with Model B (GPT-4o-mini)."""
    logger.info("=" * 80)
    logger.info("MODEL B (Llama 4 Maverick) - Starting parallel analysis")
    logger.info(f"Transcript length: {len(state['transcript'])} characters")

    llm = _create_llm("model_b")
    config = get_config()
    max_retries = config.app_settings.get("max_retries", 3)

    prompt = PARALLEL_ANALYSIS_PROMPT.format(transcript=state["transcript"])
    messages = [HumanMessage(content=prompt)]

    logger.info("Invoking Model B with validation retry (max retries: %d)", max_retries)

    # Use validation retry wrapper
    analysis = _invoke_llm_with_validation_retry(
        llm=llm,
        messages=messages,
        model_class=ParallelAnalysis,
        max_retries=max_retries,
    )

    logger.info(
        "Model B completed: %d topics, %d key points",
        len(analysis.topics),
        len(analysis.key_points),
    )
    logger.info("Topics identified: %s", ", ".join(analysis.topics[:3]) + ("..." if len(analysis.topics) > 3 else ""))

    # Return only the fields this node updates (partial state)
    return {
        "model_responses": ModelResponses(model_b=analysis),
        "messages": [
            f"Model B (Llama 4 Maverick) analysis complete: "
            f"{len(analysis.topics)} topics, {len(analysis.key_points)} key points identified"
        ],
    }


# Supervisor Node
@traceable(name="supervisor_consolidation")
def supervisor_node(state: AgentState) -> dict:
    """Consolidate analyses with Model C (Claude Sonnet)."""
    logger.info("=" * 80)
    logger.info("SUPERVISOR (Claude Sonnet) - Consolidating parallel analyses")

    llm = _create_llm("model_c")
    config = get_config()
    max_retries = config.app_settings.get("max_retries", 3)

    # Get model responses
    model_responses = state["model_responses"]

    logger.info("Merging results from Model A and Model B")
    logger.info("Model A found %d topics, Model B found %d topics",
                len(model_responses.model_a.topics),
                len(model_responses.model_b.topics))

    # Convert Pydantic models to dict for formatting
    analysis_a_dict = model_responses.model_a.model_dump_json(indent=2)
    analysis_b_dict = model_responses.model_b.model_dump_json(indent=2)

    prompt = SUPERVISOR_PROMPT.format(
        analysis_a=analysis_a_dict,
        analysis_b=analysis_b_dict,
        transcript=state["transcript"],
    )
    messages = [HumanMessage(content=prompt)]

    logger.info("Invoking Supervisor with validation retry (max retries: %d)", max_retries)

    # Use validation retry wrapper to handle Pydantic validation errors
    supervisor_output = _invoke_llm_with_validation_retry(
        llm=llm,
        messages=messages,
        model_class=SupervisorOutput,
        max_retries=max_retries,
    )

    # Add detailed progress message
    num_topics = len(supervisor_output.main_topics)
    num_takeaways = len(supervisor_output.key_takeaways)
    num_claims = len(supervisor_output.claims_to_verify)
    num_quotes = len(supervisor_output.notable_quotes)

    logger.info(
        "Supervisor consolidation complete: %d topics, %d takeaways, %d claims to verify, %d quotes",
        num_topics, num_takeaways, num_claims, num_quotes
    )
    logger.info("Summary length: %d characters (max 400)", len(supervisor_output.final_summary))
    logger.info("Claims to verify: %s", ", ".join(supervisor_output.claims_to_verify[:2]) + ("..." if num_claims > 2 else ""))

    # Return partial state update
    return {
        "model_responses": ModelResponses(supervisor=supervisor_output),
        "current_stage": "supervisor_complete",
        "messages": [
            f"Supervisor (Claude Sonnet) consolidated analyses: "
            f"{num_topics} topics, {num_takeaways} key takeaways, "
            f"{num_quotes} notable quotes, {num_claims} claims identified for verification"
        ],
    }


# Fact Checker Node
@traceable(name="fact_checker_with_tools", run_type="llm")
def fact_checker_node(state: AgentState) -> dict:
    """Verify claims with Model D (GPT-4o) using search tools."""
    logger.info("=" * 80)

    llm = _create_llm("model_d")
    config = get_config()
    max_retries = config.app_settings.get("max_retries", 3)
    tools = create_search_tools()

    # Create LLM with tool binding
    llm_with_tools = llm.bind_tools(tools)

    # Get model responses
    model_responses = state["model_responses"]

    claims = model_responses.supervisor.claims_to_verify
    summary = model_responses.supervisor.final_summary

    # Determine if this is a re-verification based on critic feedback
    is_improvement = model_responses.critic_current is not None

    if is_improvement:
        iteration = state.get("critic_iterations", 0)
        logger.info("FACT-CHECKER (GPT-4o) - Re-verification iteration %d", iteration)
        logger.info("Applying critic feedback to improve fact-checking quality")
    else:
        logger.info("FACT-CHECKER (GPT-4o) - Initial fact-checking")
        logger.info("Number of claims to verify: %d", len(claims))
        logger.info("Available search tools: %s", ", ".join(tool.name for tool in tools))

    if is_improvement:
        # Use improved prompt with critic feedback
        previous_fact_check = model_responses.fact_check_current.model_dump_json(indent=2)
        critic_feedback = model_responses.critic_current.model_dump_json(indent=2)

        prompt = IMPROVED_FACT_CHECKER_PROMPT.format(
            previous_fact_check=previous_fact_check,
            critic_feedback=critic_feedback,
            claims="\n".join(f"- {claim}" for claim in claims),
            summary=summary,
            tools=get_tool_descriptions(),
        )
    else:
        # Initial fact-checking
        prompt = FACT_CHECKER_PROMPT.format(
            claims="\n".join(f"- {claim}" for claim in claims),
            summary=summary,
            tools=get_tool_descriptions(),
        )

    # Execute with tool calling loop
    messages = [HumanMessage(content=prompt)]

    # Tool calling loop (simplified - in production, use agent executor)
    max_iterations = 10
    tool_call_count = 0
    failed_tool_calls = 0

    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check if there are tool calls
        if not response.tool_calls:
            # No more tool calls, parse final response
            break

        # Execute tool calls with error handling
        for tool_call in response.tool_calls:
            tool_call_count += 1
            tool_name = tool_call.get("name", "unknown")

            # Find matching tool
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                logger.warning(f"Tool '{tool_name}' not found in available tools")
                failed_tool_calls += 1
                continue

            try:
                # Invoke tool with error handling
                tool_result = tool.invoke(tool_call["args"])

                # Validate URLs and filter out 404s
                tool_result = validate_and_filter_search_results(tool_result)

                # Add tool result to messages
                from langchain_core.messages import ToolMessage

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

                result_preview = str(tool_result)[:150] + ("..." if len(str(tool_result)) > 150 else "")
                logger.info(f"Successfully executed tool '{tool_name}' (call {tool_call_count})")
                logger.info(f"  Result preview: {result_preview}")

            except Exception as e:
                failed_tool_calls += 1
                error_msg = f"Tool '{tool_name}' failed: {str(e)}"
                logger.error(error_msg)

                # Send error message back to LLM so it can try alternative approaches
                from langchain_core.messages import ToolMessage

                messages.append(
                    ToolMessage(
                        content=f"Error: {str(e)}. Please try a different search query or tool.",
                        tool_call_id=tool_call["id"],
                    )
                )

    # Parse final response with validation retry
    # After tool calling is complete, handle validation errors
    fact_check_output = None
    for attempt in range(max_retries + 1):
        try:
            fact_check_output = _parse_json_response(response.content, FactCheckOutput)
            if attempt > 0:
                logger.info(f"Fact-check validation succeeded on attempt {attempt + 1}")
            break
        except ValidationError as e:
            logger.warning(
                f"Fact-check validation error on attempt {attempt + 1}/{max_retries + 1}"
            )

            if attempt >= max_retries:
                # Final attempt - try auto-fix
                logger.warning("Attempting auto-fix for fact-check output")
                try:
                    response_text = response.content
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    data = json.loads(response_text.strip(), strict=False)
                    fact_check_output = _auto_fix_validation(data, e, FactCheckOutput)
                    break
                except Exception:
                    raise e

            # Retry with error feedback (no tool calls needed)
            error_message = _format_validation_error(e)
            error_prompt = f"""
Your previous response had validation errors. Please fix the JSON output:

{error_message}

Return ONLY the corrected JSON, no tool calls needed.
"""
            messages.append(HumanMessage(content=error_prompt))
            response = llm.invoke(messages)  # Use regular LLM without tools
            messages.append(response)

    # Add detailed progress message with statistics
    num_verified = sum(
        1 for claim in fact_check_output.verified_claims
        if claim.verification_status == "verified"
    )
    num_partial = sum(
        1 for claim in fact_check_output.verified_claims
        if claim.verification_status == "partially_verified"
    )
    num_unverified = sum(
        1 for claim in fact_check_output.verified_claims
        if claim.verification_status in ["unverified", "false"]
    )
    successful_tools = tool_call_count - failed_tool_calls

    logger.info(
        "Fact-checking complete: %d/%d tool calls succeeded",
        successful_tools, tool_call_count
    )
    logger.info(
        "Results: %d verified, %d partially verified, %d unverified/false",
        num_verified, num_partial, num_unverified
    )
    logger.info(
        "Quality metrics: research_quality=%.2f, overall_reliability=%.2f",
        fact_check_output.research_quality, fact_check_output.overall_reliability
    )

    # Return partial state update
    return {
        "model_responses": ModelResponses(fact_check_current=fact_check_output),
        "current_stage": "fact_check_complete",
        "messages": [
            f"Fact-checking complete: {successful_tools}/{tool_call_count} tool calls succeeded, "
            f"verified {num_verified} claims, {num_partial} partially verified, "
            f"{num_unverified} unverified/false. "
            f"Research quality: {fact_check_output.research_quality:.2f}, "
            f"Overall reliability: {fact_check_output.overall_reliability:.2f}"
        ],
    }


# Critic Node
@traceable(name="critic_quality_review")
def critic_node(state: AgentState) -> dict:
    """Evaluate fact-checking quality with independent verification."""
    logger.info("=" * 80)

    llm = _create_llm("model_c")  # Use Sonnet for critic too
    config = get_config()
    max_retries = config.app_settings.get("max_retries", 3)
    tools = create_search_tools()

    # Create LLM with tool binding
    llm_with_tools = llm.bind_tools(tools)

    # Get model responses
    model_responses = state["model_responses"]

    # Update iteration counter
    current_iteration = state.get("critic_iterations", 0)
    max_iterations = state.get("max_critic_iterations", 2)

    logger.info("CRITIC (Claude Sonnet) - Iteration %d/%d", current_iteration + 1, max_iterations)
    logger.info("Evaluating fact-checking quality and research thoroughness")
    logger.info("Available search tools for independent verification: %s", ", ".join(tool.name for tool in tools))

    fact_check_output = model_responses.fact_check_current.model_dump_json(indent=2)
    claims = model_responses.supervisor.claims_to_verify

    prompt = CRITIC_PROMPT.format(
        fact_check_output=fact_check_output,
        claims="\n".join(f"- {claim}" for claim in claims),
        tools=get_tool_descriptions(),
    )
    messages = [HumanMessage(content=prompt)]

    logger.info("Invoking Critic with tool access (max retries: %d)", max_retries)

    # Execute with tool calling loop for independent verification
    max_tool_iterations = 10
    tool_call_count = 0
    failed_tool_calls = 0

    for i in range(max_tool_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check if there are tool calls
        if not response.tool_calls:
            # No more tool calls, parse final response
            break

        # Execute tool calls with error handling
        for tool_call in response.tool_calls:
            tool_call_count += 1
            tool_name = tool_call.get("name", "unknown")

            # Find matching tool
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                logger.warning(f"Tool '{tool_name}' not found in available tools")
                failed_tool_calls += 1
                continue

            try:
                # Invoke tool with error handling
                tool_result = tool.invoke(tool_call["args"])

                # Validate URLs and filter out 404s
                tool_result = validate_and_filter_search_results(tool_result)

                # Add tool result to messages
                from langchain_core.messages import ToolMessage

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

                result_preview = str(tool_result)[:150] + ("..." if len(str(tool_result)) > 150 else "")
                logger.info(f"Critic executed tool '{tool_name}' (call {tool_call_count})")
                logger.info(f"  Result preview: {result_preview}")

            except Exception as e:
                failed_tool_calls += 1
                error_msg = f"Tool '{tool_name}' failed: {str(e)}"
                logger.error(error_msg)

                # Send error message back to LLM so it can try alternative approaches
                from langchain_core.messages import ToolMessage

                messages.append(
                    ToolMessage(
                        content=f"Error: {str(e)}. Please try a different search query or tool.",
                        tool_call_id=tool_call["id"],
                    )
                )

    # Parse final response with validation retry
    critic_feedback = None
    for attempt in range(max_retries + 1):
        try:
            critic_feedback = _parse_json_response(response.content, CriticFeedback)
            if attempt > 0:
                logger.info(f"Critic validation succeeded on attempt {attempt + 1}")
            break
        except ValidationError as e:
            logger.warning(
                f"Critic validation error on attempt {attempt + 1}/{max_retries + 1}"
            )

            if attempt >= max_retries:
                # Final attempt - try auto-fix
                logger.warning("Attempting auto-fix for critic output")
                try:
                    response_text = response.content
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    data = json.loads(response_text.strip(), strict=False)
                    critic_feedback = _auto_fix_validation(data, e, CriticFeedback)
                    break
                except Exception:
                    raise e

            # Retry with error feedback (no tool calls needed)
            error_message = _format_validation_error(e)
            error_prompt = f"""
Your previous response had validation errors. Please fix the JSON output:

{error_message}

Return ONLY the corrected JSON, no tool calls needed.
"""
            messages.append(HumanMessage(content=error_prompt))
            response = llm.invoke(messages)  # Use regular LLM without tools
            messages.append(response)

    successful_tools = tool_call_count - failed_tool_calls
    if tool_call_count > 0:
        logger.info(
            "Critic tool usage: %d/%d tool calls succeeded",
            successful_tools, tool_call_count
        )

    # Create iteration history entry
    iteration = FactCheckIteration(
        iteration=current_iteration,
        fact_check=model_responses.fact_check_current,
        critic_feedback=critic_feedback,
    )

    # Determine if we should continue
    should_continue = (
        not critic_feedback.research_is_sufficient
        and current_iteration < max_iterations - 1
    )

    logger.info("Critic evaluation: quality_score=%.2f, research_sufficient=%s",
                critic_feedback.quality_score, critic_feedback.research_is_sufficient)
    logger.info("Missing verifications: %d, Suggested improvements: %d",
                len(critic_feedback.missing_verifications),
                len(critic_feedback.suggested_improvements))

    # Add detailed progress messages
    num_missing = len(critic_feedback.missing_verifications)
    num_improvements = len(critic_feedback.suggested_improvements)

    if should_continue:
        logger.info("Decision: CONTINUE critic loop (research needs improvement)")
        tool_info = f" ({successful_tools}/{tool_call_count} tool calls)" if tool_call_count > 0 else ""
        message = (
            f"Critic iteration {current_iteration + 1}{tool_info}: Research needs improvement - "
            f"identified {num_missing} missing verifications, "
            f"{num_improvements} suggested improvements. "
            f"Quality score: {critic_feedback.quality_score:.2f}"
        )
    else:
        reason = "Critic satisfied" if critic_feedback.research_is_sufficient else "Max iterations reached"
        logger.info("Decision: END critic loop (%s)", reason)
        tool_info = f" Used {successful_tools}/{tool_call_count} tools." if tool_call_count > 0 else ""
        message = (
            f"{reason}: Quality score {critic_feedback.quality_score:.2f}.{tool_info} "
            f"Final review complete with {num_missing} remaining gaps."
        )

    # Build updated model responses with history
    updated_responses = ModelResponses(
        critic_current=critic_feedback,
        fact_check_iterations=[iteration],  # Append to existing iterations
    )

    # Return partial state update
    return {
        "model_responses": updated_responses,
        "current_stage": "critic_complete",
        "critic_iterations": current_iteration + 1,
        "should_continue": should_continue,
        "messages": [message],
    }

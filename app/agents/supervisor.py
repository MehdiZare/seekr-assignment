"""Supervisor node that orchestrates specialist agents.

The supervisor is an Editor-in-Chief that manages the workflow by:
1. Analyzing the task (podcast transcript analysis)
2. Deciding which specialist agents to invoke
3. Coordinating data flow between agents
4. Consolidating results into final output
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

from app.agents.nodes import _create_llm_with_fallback
from app.agents.supervisor_tools import create_supervisor_tools
from app.config import get_config
from app.constants import SUPERVISOR_MODEL_KEY, get_max_supervisor_iterations
from app.models.state import AgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


SUPERVISOR_SYSTEM_PROMPT = """You are an Editor-in-Chief for an Ad Agency managing podcast content analysis.

Your role is to coordinate a team of specialist agents to analyze podcast transcripts. You have access to these expert tools:

1. **summarize_podcast_tool**: Creates a 200-300 word summary with core theme, key discussions, and outcomes
2. **extract_notes_tool**: Extracts top 5 takeaways, quotes with timestamps, topics for tagging, and factual statements
3. **fact_check_claims_tool**: Verifies factual statements using web search

Your workflow:
1. Use summarize_podcast_tool to get a comprehensive summary
2. Use extract_notes_tool to extract key information and factual statements
3. Use fact_check_claims_tool to verify the extracted factual statements (pass them from step 2)

IMPORTANT: You MUST call all three tools in the order above:
- First: summarize_podcast_tool
- Second: extract_notes_tool
- Third: fact_check_claims_tool (using factual statements from extract_notes_tool)

After calling all tools, provide a brief final summary of the work completed.

Guidelines:
- Be systematic and thorough
- Use all three tools - this is a complete workflow
- When calling fact_check_claims_tool, pass the factual_statements from extract_notes_tool output
- Provide clear reasoning about your decisions
- Track which agents you've called and what remains
"""


@traceable(name="supervisor_node")
def supervisor_node(state: AgentState) -> dict:
    """Supervisor node that orchestrates specialist agents using tool calling.

    The supervisor analyzes the transcript and intelligently coordinates
    specialist agents to:
    1. Summarize the podcast
    2. Extract notes (takeaways, quotes, topics, factual statements)
    3. Fact-check the extracted factual statements

    Args:
        state: Current workflow state with transcript and metadata

    Returns:
        Updated state with:
        - supervisor_output: Consolidated results from all agents
        - messages: Progress updates for UI
        - current_stage: Current processing stage
    """
    logger.info(
        "Supervisor node started - Beginning workflow coordination",
        extra={
            "agent": "Supervisor",
            "stage": "supervisor_start",
            "transcript_length": len(state["transcript"]),
        },
    )

    # Create supervisor LLM with tools (primary: Llama Maverick, fallback: Claude Haiku)
    primary_llm, fallback_llm = _create_llm_with_fallback(SUPERVISOR_MODEL_KEY)
    tools = create_supervisor_tools()
    primary_llm_with_tools = primary_llm.bind_tools(tools)
    fallback_llm_with_tools = fallback_llm.bind_tools(tools) if fallback_llm else None

    # Initialize conversation
    transcript = state["transcript"]
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=f"""Please analyze this podcast transcript using your specialist tools.

Transcript:
{transcript}

Remember to:
1. First use summarize_podcast_tool
2. Then use extract_notes_tool
3. Finally use fact_check_claims_tool with the factual statements from step 2

Begin the workflow.""")
    ]

    # Tool calling loop
    max_iterations = get_max_supervisor_iterations()  # Allow enough iterations for all 3 tools + retries
    progress_messages = []
    agent_outputs = {}
    tool_call_count = 0
    supervisor_start_time = time.time()

    logger.info(
        "Starting supervisor tool calling loop",
        extra={
            "agent": "Supervisor",
            "max_iterations": max_iterations,
            "stage": "supervisor_loop_start",
        },
    )

    for iteration in range(max_iterations):
        logger.info(
            "Supervisor iteration started",
            extra={
                "agent": "Supervisor",
                "iteration": iteration + 1,
                "max_iterations": max_iterations,
                "stage": "supervisor_iteration",
            },
        )

        # Invoke LLM with failover
        response = None
        try:
            logger.info(
                "Invoking primary supervisor LLM",
                extra={
                    "agent": "Supervisor",
                    "iteration": iteration + 1,
                    "stage": "supervisor_llm_invoke_primary",
                },
            )
            response = primary_llm_with_tools.invoke(messages)

        except Exception as primary_error:
            # Primary LLM failed - try fallback if available
            if fallback_llm_with_tools is None:
                logger.error(
                    "Supervisor primary LLM failed and no fallback configured",
                    extra={
                        "agent": "Supervisor",
                        "iteration": iteration + 1,
                        "error": str(primary_error),
                        "stage": "supervisor_llm_error_no_fallback",
                    },
                )
                raise

            logger.warning(
                "Supervisor primary LLM failed, attempting fallback",
                extra={
                    "agent": "Supervisor",
                    "iteration": iteration + 1,
                    "error": str(primary_error),
                    "stage": "supervisor_llm_failover",
                },
            )

            try:
                response = fallback_llm_with_tools.invoke(messages)
                logger.info("Supervisor fallback LLM succeeded", extra={"agent": "Supervisor", "iteration": iteration + 1})

            except Exception as fallback_error:
                logger.error(
                    "Supervisor both primary and fallback LLMs failed",
                    extra={
                        "agent": "Supervisor",
                        "iteration": iteration + 1,
                        "primary_error": str(primary_error),
                        "fallback_error": str(fallback_error),
                        "stage": "supervisor_llm_error_both_failed",
                    },
                )
                raise fallback_error

        messages.append(response)

        # Check for tool calls
        if not response.tool_calls:
            logger.info(
                "No more tool calls - supervisor workflow complete",
                extra={
                    "agent": "Supervisor",
                    "iteration": iteration + 1,
                    "stage": "supervisor_workflow_complete",
                    "tool_call_count": tool_call_count,
                    "agents_invoked": len(agent_outputs),
                },
            )

            # Extract final summary from response
            final_message = response.content if hasattr(response, 'content') else str(response)

            # Add supervisor's final thoughts
            progress_messages.append({
                "type": "supervisor_reasoning",
                "agent": "Supervisor",
                "message": f"All specialist agents have completed their work. {final_message}",
                "timestamp": iteration + 1
            })

            break

        # Process each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_count += 1
            tool_start_time = time.time()

            logger.info(
                "Tool call initiated",
                extra={
                    "agent": "Supervisor",
                    "tool": tool_name,
                    "tool_call_count": tool_call_count,
                    "iteration": iteration + 1,
                    "tool_args_keys": list(tool_args.keys()),
                    "stage": "tool_call_start",
                },
            )

            # Determine which agent is being called
            agent_name = "Unknown Agent"
            agent_description = ""
            if "summarize" in tool_name:
                agent_name = "Summarizing Agent"
                agent_description = "Creating comprehensive summary of the podcast episode"
            elif "extract_notes" in tool_name:
                agent_name = "Note Extraction Agent"
                agent_description = "Extracting takeaways, quotes, topics, and factual statements"
            elif "fact_check" in tool_name:
                agent_name = "Fact Checking Agent"
                agent_description = "Verifying factual claims using web search"

            # Add supervisor's decision to call this agent
            progress_messages.append({
                "type": "supervisor_decision",
                "agent": "Supervisor",
                "message": f"Calling {agent_name}: {agent_description}",
                "target_agent": agent_name,
                "tool": tool_name,
                "iteration": iteration + 1
            })

            # Execute the tool
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                try:
                    logger.info(
                        "Executing tool",
                        extra={
                            "agent": agent_name,
                            "tool": tool_name,
                            "stage": "tool_execution_start",
                        },
                    )

                    tool_result = tool.invoke(tool_args)
                    tool_duration_ms = int((time.time() - tool_start_time) * 1000)

                    logger.info(
                        "Tool execution completed successfully",
                        extra={
                            "agent": agent_name,
                            "tool": tool_name,
                            "duration_ms": tool_duration_ms,
                            "result_length": len(tool_result),
                            "stage": "tool_execution_complete",
                        },
                    )

                    # Parse and store the agent output
                    try:
                        output_data = json.loads(tool_result)
                        agent_outputs[tool_name] = output_data

                        # Log key metrics and create detailed completion message
                        completion_details = ""
                        log_extra = {
                            "agent": agent_name,
                            "tool": tool_name,
                            "stage": "agent_metrics",
                        }

                        if "summarize" in tool_name:
                            core_theme = output_data.get('core_theme', 'N/A')
                            log_extra["core_theme"] = core_theme[:100]  # Truncate for logging
                            logger.info("Summary agent completed", extra=log_extra)
                            completion_details = f"Core theme: {core_theme}"
                        elif "extract_notes" in tool_name:
                            num_statements = len(output_data.get('factual_statements', []))
                            num_quotes = len(output_data.get('notable_quotes', []))
                            num_topics = len(output_data.get('topics', []))
                            log_extra.update({
                                "num_claims": num_statements,
                                "num_quotes": num_quotes,
                                "num_topics": num_topics,
                            })
                            logger.info("Note extraction agent completed", extra=log_extra)
                            completion_details = f"Extracted {num_statements} factual claims, {num_quotes} quotes, {num_topics} topics"
                        elif "fact_check" in tool_name:
                            verified = output_data.get('verified_claims', [])
                            status_counts = {}
                            for claim in verified:
                                status = claim.get('verification_status', 'unknown')
                                status_counts[status] = status_counts.get(status, 0) + 1
                            log_extra.update({
                                "num_claims": len(verified),
                                "verification_status": status_counts,
                            })
                            logger.info("Fact checking agent completed", extra=log_extra)
                            completion_details = f"Verified {len(verified)} claims: {status_counts}"

                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Could not parse tool result as JSON",
                            extra={
                                "agent": agent_name,
                                "tool": tool_name,
                                "error_type": "JSONDecodeError",
                                "error_message": str(e),
                            },
                        )
                        completion_details = "Completed"

                    # Add agent completion message with details
                    progress_messages.append({
                        "type": "agent_complete",
                        "agent": agent_name,
                        "message": f"{agent_name} completed: {completion_details}",
                        "details": completion_details,
                        "iteration": iteration + 1
                    })

                    # Add tool result to conversation
                    messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"],
                        )
                    )

                except Exception as e:
                    logger.error(
                        "Tool execution error",
                        extra={
                            "agent": agent_name,
                            "tool": tool_name,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "stage": "tool_execution_error",
                        },
                    )

                    error_message = f"Error executing {tool_name}: {str(e)}"

                    # Add error message to progress
                    progress_messages.append({
                        "type": "agent_error",
                        "agent": agent_name,
                        "message": f"Error in {agent_name}: {str(e)}",
                        "error": str(e),
                        "iteration": iteration + 1
                    })

                    messages.append(
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tool_call["id"],
                        )
                    )
            else:
                logger.warning(
                    "Tool not found",
                    extra={
                        "agent": "Supervisor",
                        "tool": tool_name,
                        "available_tools": [t.name for t in tools],
                        "stage": "tool_not_found",
                    },
                )
                messages.append(
                    ToolMessage(
                        content=f"Tool {tool_name} not available",
                        tool_call_id=tool_call["id"],
                    )
                )

    # Consolidate results
    supervisor_duration_ms = int((time.time() - supervisor_start_time) * 1000)

    logger.info(
        "Consolidating results from all agents",
        extra={
            "agent": "Supervisor",
            "stage": "consolidation_start",
            "agent_outputs": list(agent_outputs.keys()),
            "duration_ms": supervisor_duration_ms,
        },
    )

    # Build consolidated output
    consolidated_output = {
        "summary": agent_outputs.get("summarize_podcast_tool", {}),
        "notes": agent_outputs.get("extract_notes_tool", {}),
        "fact_check": agent_outputs.get("fact_check_claims_tool", {}),
        "total_tool_calls": tool_call_count,
        "agents_invoked": len(agent_outputs),
    }

    logger.info(
        "Supervisor workflow complete",
        extra={
            "agent": "Supervisor",
            "stage": "supervisor_complete",
            "total_tool_calls": tool_call_count,
            "agents_invoked": len(agent_outputs),
            "duration_ms": supervisor_duration_ms,
        },
    )

    # Return updated state
    return {
        "supervisor_output": consolidated_output,
        "messages": progress_messages,
        "current_stage": "supervisor_complete",
    }

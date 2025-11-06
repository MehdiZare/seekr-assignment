"""Specialist agent implementations for the new supervisor-based workflow.

These agents are called by the supervisor agent as tools to perform specialized tasks:
- Summarizing: Create 200-300 word summaries
- Note Extraction: Extract takeaways, quotes, topics, and factual statements
- Fact Checking: Verify factual claims using search tools
"""

import time
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langsmith import traceable

from app.agents.nodes import _create_llm_with_fallback, _invoke_llm_with_failover
from app.agents.tools import create_search_tools, validate_and_filter_search_results
from app.config import get_config
from app.constants import FACT_CHECK_MODEL_KEY, get_max_fact_check_iterations
from app.models.outputs import (
    FactCheckOutput,
    NotesOutput,
    SummaryOutput,
    VerifiedClaim,
    Source,
    FactualStatement,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# AGENT PROMPTS
# ============================================================================

SUMMARIZER_PROMPT = """You are a podcast summarizing expert working for an Ad Agency.

Your task is to create a comprehensive summary of this podcast episode.

Podcast Transcript:
{transcript}

You must provide:
1. A 200-300 word summary that captures the essence of the episode
2. The core theme or central topic
3. Key discussions and conversation points
4. Outcomes, conclusions, or opinions shared by the participants

Your summary should be clear, engaging, and suitable for publishing on social media or marketing materials.

IMPORTANT: Respond with valid JSON matching this exact schema:
{{
    "summary": "A comprehensive 200-300 word summary of the episode...",
    "core_theme": "The central theme or main topic",
    "key_discussions": ["Discussion point 1", "Discussion point 2", ...],
    "outcomes_and_opinions": ["Outcome or opinion 1", "Outcome or opinion 2", ...],
    "reasoning": "Your thought process for creating this summary"
}}

Guidelines:
- The summary MUST be between 1200-2400 characters (approximately 200-400 words)
- Focus on what makes this episode valuable and interesting
- Capture the tone and style of the conversation
- Include at least 2 key discussions
- Include at least 1 outcome or opinion
- Be specific and avoid generic statements
- Only return the JSON object, no additional text
"""

NOTE_EXTRACTOR_PROMPT = """You are a content extraction expert working for an Ad Agency.

Your task is to extract key information from this podcast episode for publishing and marketing purposes.

Podcast Transcript:
{transcript}

You must extract:
1. Top 5 takeaways - The most valuable insights from the episode
2. Notable quotes with timestamps - Memorable, shareable quotes
3. Topics for tagging - Keywords and themes for categorization (at least 3)
4. Factual statements - Specific claims that can be verified (e.g., "I started company X and sold it for Y$")

IMPORTANT: Respond with valid JSON matching this exact schema:
{{
    "top_takeaways": [
        "Takeaway 1",
        "Takeaway 2",
        "Takeaway 3",
        "Takeaway 4",
        "Takeaway 5"
    ],
    "notable_quotes": [
        {{
            "text": "The actual quote",
            "speaker": "Speaker name or null",
            "timestamp": "00:15:23 or null",
            "context": "Brief context around the quote"
        }}
    ],
    "topics": ["topic1", "topic2", "topic3", ...],
    "factual_statements": [
        {{
            "statement": "The factual claim",
            "speaker": "Who made the claim or null",
            "context": "Context around the statement",
            "timestamp": "00:10:30 or null"
        }}
    ],
    "reasoning": "Your thought process for extracting this information"
}}

Guidelines:
- EXACTLY 5 takeaways (no more, no less)
- Extract timestamps from the transcript when available (format: "MM:SS" or "HH:MM:SS")
- Topics should be single words or short phrases (e.g., "remote work", "AI", "productivity")
- Include at least 3 topics
- Factual statements should be specific, verifiable claims (statistics, dates, company names, scientific facts)
- Focus on factual statements that are central to the episode's message
- Notable quotes should be memorable and shareable
- Only return the JSON object, no additional text
"""

FACT_CHECKER_PROMPT = """You are a fact-checking expert with access to search tools.

You must verify the following factual statements from a podcast episode:

Factual Statements to Verify:
{factual_statements}

Context:
{context}

Available Tools:
{tool_descriptions}

Your task:
1. For each factual statement, use the search tools to find credible sources
2. Search for top 10 results and evaluate their credibility
3. Classify each statement as:
   - "fact-checked": Verified with credible sources (provide links)
   - "unverified": No credible sources found to confirm or deny
   - "declined": Credible evidence contradicts the claim (provide links)

Credible sources include: reputable news organizations, academic institutions, official company websites, government websites, peer-reviewed publications.

IMPORTANT: You MUST use the search tools available to you. Call the search tool multiple times (once for each claim).

After you've gathered search results, respond with valid JSON matching this schema:
{{
    "verified_claims": [
        {{
            "claim": "The original statement",
            "verification_status": "fact-checked" | "unverified" | "declined",
            "confidence": 0.85,
            "sources": [
                {{
                    "url": "https://credible-source.com/article",
                    "title": "Article title or description",
                    "relevance": 0.9
                }}
            ],
            "reasoning": "Detailed explanation of verification process and findings",
            "additional_context": "Any nuance or additional information discovered"
        }}
    ],
    "overall_reliability": 0.8,
    "research_quality": 0.85,
    "reasoning": "Overall assessment of the fact-checking process and findings"
}}

Guidelines:
- Use search tools extensively - don't guess or use prior knowledge
- For each claim, search with different query variations
- Evaluate source credibility carefully
- Confidence should reflect certainty (0.0-1.0)
- Overall reliability is the average verification confidence
- Research quality reflects how thorough your search was
- Only return the JSON object after you've used the search tools
"""


# ============================================================================
# SPECIALIST AGENT FUNCTIONS
# ============================================================================

@traceable(name="summarizer_agent")
def summarize_podcast(transcript: str) -> dict[str, Any]:
    """Summarizing Agent: Create a 200-300 word summary of the podcast episode.

    Args:
        transcript: The full podcast transcript text

    Returns:
        Dictionary with:
        - output: SummaryOutput object
        - reasoning: Agent's reasoning process
        - agent_name: "Summarizing Agent"
    """
    agent_start_time = time.time()

    logger.info(
        "Summarizing Agent started",
        extra={
            "agent": "Summarizing Agent",
            "stage": "agent_start",
            "transcript_length": len(transcript),
        },
    )

    # Get max retries from config
    config = get_config()
    max_retries = config.app_settings.get("max_retries", 3)

    # Format prompt
    prompt = SUMMARIZER_PROMPT.format(transcript=transcript)
    messages = [HumanMessage(content=prompt)]

    logger.info(
        "Invoking Summarizer with failover",
        extra={
            "agent": "Summarizing Agent",
            "stage": "llm_invoke",
            "max_retries": max_retries,
        },
    )

    # Invoke LLM with failover (Primary: Llama Maverick, Fallback: Claude Haiku)
    summary_output = _invoke_llm_with_failover(
        model_key="model_c",
        messages=messages,
        model_class=SummaryOutput,
        max_retries=max_retries,
    )

    agent_duration_ms = int((time.time() - agent_start_time) * 1000)

    logger.info(
        "Summarizing Agent completed",
        extra={
            "agent": "Summarizing Agent",
            "stage": "agent_complete",
            "duration_ms": agent_duration_ms,
            "summary_length": len(summary_output.summary),
            "num_key_discussions": len(summary_output.key_discussions),
            "num_outcomes": len(summary_output.outcomes_and_opinions),
            "core_theme": summary_output.core_theme[:100],  # Truncate for logging
        },
    )

    return {
        "output": summary_output,
        "reasoning": summary_output.reasoning,
        "agent_name": "Summarizing Agent",
    }


@traceable(name="note_extractor_agent")
def extract_notes(transcript: str) -> dict[str, Any]:
    """Note Extraction Agent: Extract takeaways, quotes, topics, and factual statements.

    Args:
        transcript: The full podcast transcript text

    Returns:
        Dictionary with:
        - output: NotesOutput object
        - reasoning: Agent's reasoning process
        - agent_name: "Note Extraction Agent"
    """
    agent_start_time = time.time()

    logger.info(
        "Note Extraction Agent started",
        extra={
            "agent": "Note Extraction Agent",
            "stage": "agent_start",
            "transcript_length": len(transcript),
        },
    )

    # Get max retries from config
    config = get_config()
    max_retries = config.app_settings.get("max_retries", 3)

    # Format prompt
    prompt = NOTE_EXTRACTOR_PROMPT.format(transcript=transcript)
    messages = [HumanMessage(content=prompt)]

    logger.info(
        "Invoking Note Extractor with failover",
        extra={
            "agent": "Note Extraction Agent",
            "stage": "llm_invoke",
            "max_retries": max_retries,
        },
    )

    # Invoke LLM with failover (Primary: Llama Maverick, Fallback: Claude Haiku)
    notes_output = _invoke_llm_with_failover(
        model_key="model_c",
        messages=messages,
        model_class=NotesOutput,
        max_retries=max_retries,
    )

    agent_duration_ms = int((time.time() - agent_start_time) * 1000)

    logger.info(
        "Note Extraction Agent completed",
        extra={
            "agent": "Note Extraction Agent",
            "stage": "agent_complete",
            "duration_ms": agent_duration_ms,
            "num_takeaways": len(notes_output.top_takeaways),
            "num_quotes": len(notes_output.notable_quotes),
            "num_topics": len(notes_output.topics),
            "num_claims": len(notes_output.factual_statements),
            "topics": notes_output.topics[:5],  # First 5 topics
        },
    )

    return {
        "output": notes_output,
        "reasoning": notes_output.reasoning,
        "agent_name": "Note Extraction Agent",
    }


@traceable(name="fact_checker_agent")
def fact_check_claims(
    factual_statements: list[FactualStatement],
    context: str,
) -> dict[str, Any]:
    """Fact Checking Agent: Verify factual claims using search tools.

    Note: Empty lists are handled gracefully by the tool wrapper (fact_check_claims_tool),
    which returns a default response before calling this function.

    Args:
        factual_statements: List of FactualStatement objects to verify
        context: Context/summary for better search queries

    Returns:
        Dictionary with:
        - output: FactCheckOutput object
        - reasoning: Agent's reasoning process
        - agent_name: "Fact Checking Agent"
        - tool_calls: List of tool call details
    """
    agent_start_time = time.time()

    logger.info(
        "Fact Checking Agent started",
        extra={
            "agent": "Fact Checking Agent",
            "stage": "agent_start",
            "num_claims": len(factual_statements),
        },
    )

    # Create LLM instance with tools and failover (Primary: Llama Maverick, Fallback: Claude Haiku)
    primary_llm, fallback_llm = _create_llm_with_fallback(FACT_CHECK_MODEL_KEY)
    config = get_config()
    max_retries = config.app_settings.get("max_retries", 3)

    # Create search tools
    tools = create_search_tools()
    tool_descriptions = "\n".join(
        f"- {tool.name}: {tool.description}" for tool in tools
    )
    primary_llm_with_tools = primary_llm.bind_tools(tools)
    fallback_llm_with_tools = fallback_llm.bind_tools(tools) if fallback_llm else None

    # Format factual statements for the prompt
    statements_text = "\n".join(
        f"{i+1}. {stmt.statement} (Speaker: {stmt.speaker or 'Unknown'}, Context: {stmt.context})"
        for i, stmt in enumerate(factual_statements)
    )

    # Format prompt
    prompt = FACT_CHECKER_PROMPT.format(
        factual_statements=statements_text,
        context=context,
        tool_descriptions=tool_descriptions,
    )

    messages = [HumanMessage(content=prompt)]

    # Tool calling loop
    max_iterations = get_max_fact_check_iterations()

    logger.info(
        "Starting fact-checking with search tools",
        extra={
            "agent": "Fact Checking Agent",
            "stage": "tool_calling_start",
            "max_iterations": max_iterations,
        },
    )
    tool_call_log = []

    for iteration in range(max_iterations):
        logger.info(
            "Fact-checking tool calling iteration",
            extra={
                "agent": "Fact Checking Agent",
                "iteration": iteration + 1,
                "max_iterations": max_iterations,
                "stage": "tool_calling_iteration",
            },
        )

        # Invoke LLM with failover
        try:
            response = primary_llm_with_tools.invoke(messages)
            messages.append(response)

        except Exception as primary_error:
            # Primary LLM failed - try fallback if available
            if fallback_llm_with_tools is None:
                logger.error(
                    "Fact-checking primary LLM failed and no fallback configured",
                    extra={
                        "agent": "Fact Checking Agent",
                        "iteration": iteration + 1,
                        "error_type": type(primary_error).__name__,
                        "error_message": str(primary_error),
                        "failover_triggered": False,
                        "stage": "fact_check_llm_error_no_fallback",
                    },
                )
                raise

            logger.warning(
                "Fact-checking primary LLM failed, attempting fallback",
                extra={
                    "agent": "Fact Checking Agent",
                    "iteration": iteration + 1,
                    "error_type": type(primary_error).__name__,
                    "error_message": str(primary_error),
                    "failover_triggered": True,
                    "failover_reason": type(primary_error).__name__,
                    "stage": "fact_check_llm_failover",
                },
            )

            try:
                response = fallback_llm_with_tools.invoke(messages)
                messages.append(response)

                logger.info(
                    "Fact-checking fallback LLM succeeded",
                    extra={
                        "agent": "Fact Checking Agent",
                        "iteration": iteration + 1,
                        "failover_triggered": True,
                        "stage": "fact_check_llm_success_fallback",
                    },
                )

            except Exception as fallback_error:
                logger.error(
                    "Fact-checking both primary and fallback LLMs failed",
                    extra={
                        "agent": "Fact Checking Agent",
                        "iteration": iteration + 1,
                        "primary_error_type": type(primary_error).__name__,
                        "primary_error_message": str(primary_error),
                        "fallback_error_type": type(fallback_error).__name__,
                        "fallback_error_message": str(fallback_error),
                        "failover_triggered": True,
                        "stage": "fact_check_llm_error_both_failed",
                    },
                )
                raise fallback_error

        # Check if there are tool calls
        if not response.tool_calls:
            logger.info(
                "No more tool calls - finalizing fact-check results",
                extra={
                    "agent": "Fact Checking Agent",
                    "iteration": iteration + 1,
                    "stage": "tool_calling_complete",
                },
            )
            break

        # Process each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            query = tool_args.get('query', tool_args)

            logger.info(
                "Executing search tool",
                extra={
                    "agent": "Fact Checking Agent",
                    "tool": tool_name,
                    "query": str(query)[:100],  # Truncate for logging
                    "stage": "search_tool_call",
                },
            )

            # Execute tool
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                try:
                    tool_result = tool.invoke(tool_args)

                    # Validate and filter search results
                    results_count = 0
                    if tool_name in ["tavily_search", "brave_search", "serper_search"]:
                        tool_result = validate_and_filter_search_results(tool_result)
                        results_count = len(tool_result) if isinstance(tool_result, list) else 0

                        logger.info(
                            "Search tool completed",
                            extra={
                                "agent": "Fact Checking Agent",
                                "tool": tool_name,
                                "search_results": results_count,
                                "stage": "search_results",
                            },
                        )

                    # Log tool call
                    tool_call_log.append({
                        "iteration": iteration + 1,
                        "tool": tool_name,
                        "query": tool_args.get('query', str(tool_args)),
                        "results_count": results_count
                    })

                    # Add tool result to messages
                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"],
                        )
                    )

                except Exception as e:
                    logger.error(
                        "Search tool execution error",
                        extra={
                            "agent": "Fact Checking Agent",
                            "tool": tool_name,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "stage": "search_tool_error",
                        },
                    )
                    messages.append(
                        ToolMessage(
                            content=f"Error executing tool: {str(e)}",
                            tool_call_id=tool_call["id"],
                        )
                    )
            else:
                logger.warning(
                    "Search tool not found",
                    extra={
                        "agent": "Fact Checking Agent",
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

    # Parse final response
    logger.info(
        "Parsing fact-check results with validation retry",
        extra={
            "agent": "Fact Checking Agent",
            "stage": "parsing_results",
            "max_retries": max_retries,
        },
    )

    # The last message should be the LLM's final response (after all tool calls)
    # If the last message has content, it's the final JSON response
    if hasattr(messages[-1], 'content') and messages[-1].content:
        final_response_text = messages[-1].content

        # Parse and validate the JSON response
        from app.agents.nodes import _parse_json_response
        try:
            fact_check_output = _parse_json_response(final_response_text, FactCheckOutput)
        except Exception as e:
            logger.warning(
                "Failed to parse final response directly",
                extra={
                    "agent": "Fact Checking Agent",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "stage": "parse_error",
                },
            )
            # Ask the LLM to format it properly
            messages.append(HumanMessage(content="Please provide your fact-check results in valid JSON format as specified in the schema."))
            fact_check_output = _invoke_llm_with_failover(
                model_key="model_d",
                messages=messages,
                model_class=FactCheckOutput,
                max_retries=max_retries,
            )
    else:
        # If no final content, ask for the results
        messages.append(HumanMessage(content="Based on your search results, please provide the final fact-check results in JSON format."))
        fact_check_output = _invoke_llm_with_failover(
            model_key="model_d",
            messages=messages,
            model_class=FactCheckOutput,
            max_retries=max_retries,
        )

    agent_duration_ms = int((time.time() - agent_start_time) * 1000)

    # Count verification statuses
    status_counts = {}
    for claim in fact_check_output.verified_claims:
        status_counts[claim.verification_status] = status_counts.get(claim.verification_status, 0) + 1

    logger.info(
        "Fact Checking Agent completed",
        extra={
            "agent": "Fact Checking Agent",
            "stage": "agent_complete",
            "duration_ms": agent_duration_ms,
            "num_claims": len(fact_check_output.verified_claims),
            "verification_status": status_counts,
            "overall_reliability": fact_check_output.overall_reliability,
            "research_quality": fact_check_output.research_quality,
            "total_tool_calls": len(tool_call_log),
        },
    )

    return {
        "output": fact_check_output,
        "reasoning": fact_check_output.reasoning,
        "agent_name": "Fact Checking Agent",
        "tool_calls": tool_call_log,
    }

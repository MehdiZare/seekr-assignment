"""Tool wrappers for specialist agents to be used by the supervisor.

The supervisor agent uses these tools to delegate work to specialist agents.
Each tool wraps a specialist agent function and handles parameter conversion.
"""

import json
from typing import Any

from langchain_core.tools import tool
from pydantic import ValidationError

from app.agents.specialist_agents import (
    summarize_podcast,
    extract_notes,
    fact_check_claims,
)
from app.models.outputs import FactualStatement
from app.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def summarize_podcast_tool(transcript: str) -> str:
    """Summarize a podcast episode with a 200-300 word summary.

    This tool analyzes the full podcast transcript and creates:
    - A comprehensive 200-300 word summary
    - The core theme or central topic
    - Key discussions and conversation points
    - Outcomes, conclusions, or opinions shared

    Args:
        transcript: The full podcast transcript text

    Returns:
        JSON string containing the summary output with fields:
        - summary: The 200-300 word summary
        - core_theme: Central theme of the episode
        - key_discussions: List of main discussion points
        - outcomes_and_opinions: List of outcomes or opinions
        - reasoning: Agent's thought process
    """
    logger.info("Supervisor calling Summarizing Agent tool")

    try:
        # Call the specialist agent
        result = summarize_podcast(transcript)

        # Convert output to JSON string
        output_dict = result["output"].model_dump()
        output_dict["reasoning"] = result["reasoning"]

        return json.dumps(output_dict, indent=2)

    except ValidationError as e:
        # Format validation errors for the LLM
        error_messages = []
        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])
            error_messages.append(f"{field}: {err['msg']}")

        logger.error(f"Validation error in summarize_podcast_tool: {error_messages}")

        return json.dumps({
            "error": "ValidationError - The summary output did not meet requirements",
            "error_type": "ValidationError",
            "error_details": error_messages,
            "reasoning": "The summary generation failed validation. Please review the requirements and try again.",
            "summary": None,
            "core_theme": None,
            "key_discussions": [],
            "outcomes_and_opinions": []
        }, indent=2)

    except Exception as e:
        logger.error(f"Unexpected error in summarize_podcast_tool: {type(e).__name__}: {e}")

        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "reasoning": "An unexpected error occurred during summary generation.",
            "summary": None,
            "core_theme": None,
            "key_discussions": [],
            "outcomes_and_opinions": []
        }, indent=2)


@tool
def extract_notes_tool(transcript: str) -> str:
    """Extract key information from a podcast episode.

    This tool analyzes the podcast transcript and extracts:
    - Top 5 takeaways (most valuable insights)
    - Notable quotes with timestamps
    - Topics for tagging (keywords and themes)
    - Factual statements that can be verified

    Args:
        transcript: The full podcast transcript text

    Returns:
        JSON string containing the notes output with fields:
        - top_takeaways: List of exactly 5 key takeaways
        - notable_quotes: List of quote objects with text, speaker, timestamp, context
        - topics: List of topic keywords for tagging
        - factual_statements: List of factual claim objects
        - reasoning: Agent's thought process
    """
    logger.info("Supervisor calling Note Extraction Agent tool")

    try:
        # Call the specialist agent
        result = extract_notes(transcript)

        # Convert output to JSON string
        output_dict = result["output"].model_dump()
        output_dict["reasoning"] = result["reasoning"]

        return json.dumps(output_dict, indent=2)

    except ValidationError as e:
        # Format validation errors for the LLM
        error_messages = []
        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])
            error_messages.append(f"{field}: {err['msg']}")

        logger.error(f"Validation error in extract_notes_tool: {error_messages}")

        return json.dumps({
            "error": "ValidationError - The notes extraction output did not meet requirements",
            "error_type": "ValidationError",
            "error_details": error_messages,
            "reasoning": "The notes extraction failed validation. Please review the requirements and try again.",
            "top_takeaways": [],
            "notable_quotes": [],
            "topics": [],
            "factual_statements": []
        }, indent=2)

    except Exception as e:
        logger.error(f"Unexpected error in extract_notes_tool: {type(e).__name__}: {e}")

        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "reasoning": "An unexpected error occurred during notes extraction.",
            "top_takeaways": [],
            "notable_quotes": [],
            "topics": [],
            "factual_statements": []
        }, indent=2)


@tool
def fact_check_claims_tool(factual_statements_json: str, context: str) -> str:
    """Verify factual claims using search tools.

    This tool takes factual statements and verifies them using web search:
    - Searches for top 10 results per claim
    - Evaluates source credibility
    - Classifies as: fact-checked, unverified, or declined
    - Provides credible source links

    Args:
        factual_statements_json: JSON string containing list of factual statements.
            Each statement should have: statement, speaker, context, timestamp.
            Example:
            [
                {
                    "statement": "I started company X in 2020",
                    "speaker": "John Doe",
                    "context": "Discussion about entrepreneurship",
                    "timestamp": "00:15:30"
                }
            ]
        context: Context or summary to help with search queries

    Returns:
        JSON string containing fact-check results with fields:
        - verified_claims: List of verified claim objects with status, sources, reasoning
        - overall_reliability: Overall reliability score (0-1)
        - research_quality: Quality of research conducted (0-1)
        - reasoning: Agent's assessment of the fact-checking process
    """
    logger.info("Supervisor calling Fact Checking Agent tool")

    # Parse the factual statements JSON
    try:
        statements_data = json.loads(factual_statements_json)

        # Convert to FactualStatement objects
        factual_statements = [
            FactualStatement(**stmt) for stmt in statements_data
        ]

        logger.info(f"Parsed {len(factual_statements)} factual statements for verification")

        # Handle empty list - return default message when no facts to verify
        if len(factual_statements) == 0:
            logger.info("No factual statements to verify - returning default response")
            return json.dumps({
                "verified_claims": [{
                    "claim": "No factual statements found to verify",
                    "verification_status": "unverified",
                    "confidence": 1.0,
                    "sources": [],
                    "reasoning": "No factual statements were extracted from the podcast episode for verification.",
                    "additional_context": "This episode may be primarily opinion-based, conversational, or does not contain specific verifiable claims."
                }],
                "overall_reliability": 1.0,
                "research_quality": 1.0,
                "reasoning": "No factual statements were provided for verification. This indicates that the podcast content is primarily opinion-based or conversational without specific verifiable claims."
            }, indent=2)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse factual_statements_json: {e}")
        return json.dumps({
            "error": f"Invalid JSON in factual_statements_json: {str(e)}",
            "verified_claims": [],
            "overall_reliability": 0.0,
            "research_quality": 0.0,
            "reasoning": "Failed to parse input data"
        })
    except Exception as e:
        logger.error(f"Failed to create FactualStatement objects: {e}")
        return json.dumps({
            "error": f"Invalid factual statement data: {str(e)}",
            "verified_claims": [],
            "overall_reliability": 0.0,
            "research_quality": 0.0,
            "reasoning": "Failed to validate input data"
        })

    # Call the specialist agent
    try:
        result = fact_check_claims(factual_statements, context)

        # Convert output to JSON string
        output_dict = result["output"].model_dump()
        output_dict["reasoning"] = result["reasoning"]
        output_dict["tool_calls_summary"] = {
            "total_searches": len(result.get("tool_calls", [])),
            "tools_used": list(set(tc["tool"] for tc in result.get("tool_calls", [])))
        }

        return json.dumps(output_dict, indent=2)

    except ValidationError as e:
        # Format validation errors for the LLM
        error_messages = []
        for err in e.errors():
            field = ".".join(str(x) for x in err["loc"])
            error_messages.append(f"{field}: {err['msg']}")

        logger.error(f"Validation error in fact_check_claims_tool: {error_messages}")

        return json.dumps({
            "error": "ValidationError - The fact-checking output did not meet requirements",
            "error_type": "ValidationError",
            "error_details": error_messages,
            "reasoning": "The fact-checking process failed validation. Please review the requirements and try again.",
            "verified_claims": [],
            "overall_reliability": 0.0,
            "research_quality": 0.0
        }, indent=2)

    except Exception as e:
        logger.error(f"Unexpected error in fact_check_claims_tool: {type(e).__name__}: {e}")

        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "reasoning": "An unexpected error occurred during fact-checking.",
            "verified_claims": [],
            "overall_reliability": 0.0,
            "research_quality": 0.0
        }, indent=2)


def create_supervisor_tools() -> list:
    """Create the list of tools available to the supervisor agent.

    Returns:
        List of tool objects that can be bound to an LLM
    """
    return [
        summarize_podcast_tool,
        extract_notes_tool,
        fact_check_claims_tool,
    ]

"""LangGraph state definition for the podcast analysis workflow."""

from operator import add
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph workflow.

    This state is shared across all nodes in the graph and accumulates
    results as the workflow progresses.

    Uses TypedDict for LangGraph compatibility, but nests Pydantic models
    for structured, validated data.
    """

    # Input
    transcript: str
    metadata: dict | None

    # Supervisor workflow output
    supervisor_output: dict  # Contains: summary, notes, fact_check, metadata

    # Progress tracking for SSE (accumulates messages from all nodes)
    current_stage: str
    messages: Annotated[list[str], add]

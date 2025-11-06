"""LangGraph state definition for the podcast analysis workflow."""

from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph workflow.

    This state is shared across all nodes in the graph and accumulates
    results as the workflow progresses.

    Uses TypedDict for LangGraph compatibility, but nests Pydantic models
    for structured, validated data.
    """

    # Input
    transcript: str
    metadata: dict[str, Any] | None
    session_id: str  # Session ID for tracking and logging

    # Supervisor workflow output
    supervisor_output: dict[str, Any]  # Contains: summary, notes, fact_check, metadata

    # Progress tracking for SSE (accumulates messages from all nodes)
    current_stage: str
    progress_messages: Annotated[list[str], add]  # UI progress tracking

    # LangGraph message passing (for agent-tool communication)
    messages: Annotated[list[BaseMessage], add]  # LangGraph conversation history

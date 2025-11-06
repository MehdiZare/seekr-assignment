"""LangGraph workflow definition for podcast analysis."""

import sys
from langgraph.graph import END, StateGraph

# Import supervisor-based workflow
from app.agents.supervisor import supervisor_node

from app.models.state import AgentState
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_workflow() -> StateGraph:
    """Create and compile the NEW supervisor-based workflow.

    New workflow: START → Supervisor (coordinates specialist agents) → END

    The supervisor intelligently calls specialist agents as tools:
    - Summarizing Agent
    - Note Extraction Agent
    - Fact Checking Agent

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info(
        "Creating supervisor-based workflow graph",
        extra={
            "stage": "workflow_creation",
            "workflow_type": "supervisor-based",
        },
    )

    # Create graph
    workflow = StateGraph(AgentState)

    # Add supervisor node (the only node in the workflow)
    workflow.add_node("supervisor", supervisor_node)

    # Simple linear flow: START → Supervisor → END
    workflow.add_edge("__start__", "supervisor")
    workflow.add_edge("supervisor", END)

    # Compile and return
    logger.info("About to compile workflow graph")

    try:
        compiled_workflow = workflow.compile()
        logger.info(
            "Workflow graph compiled successfully",
            extra={
                "stage": "workflow_compiled",
                "nodes": ["supervisor"],
            },
        )
        return compiled_workflow
    except Exception as e:
        logger.error(
            "CRITICAL: Workflow compilation failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        raise


async def stream_analysis(
    transcript: str,
    metadata: dict | None = None,
    session_id: str | None = None,
):
    """Stream the podcast analysis workflow with real-time updates.

    Uses LangGraph's astream_events() for fine-grained streaming that captures
    tool calls in real-time, allowing the UI to show progress as agents execute.

    Args:
        transcript: The podcast transcript text
        metadata: Optional metadata about the podcast
        session_id: Session ID for tracking and logging

    Yields:
        Events from the workflow including tool calls, completions, and final state
    """
    # Set session context early so all logs include session_id
    if session_id:
        from app.utils.logger import set_session_context
        set_session_context(session_id)

    logger.info(
        "Starting streaming analysis workflow",
        extra={
            "stage": "streaming_workflow_start",
            "transcript_length": len(transcript),
            "has_metadata": metadata is not None,
        },
    )

    # Initialize state with session_id
    initial_state: AgentState = {
        "transcript": transcript,
        "metadata": metadata,
        "session_id": session_id or "",
        "current_stage": "initialized",
        "messages": [],
    }

    workflow = create_workflow()

    # Stream fine-grained events (tool calls, LLM calls, etc.)
    # version="v2" is the current stable event streaming format
    event_count = 0

    try:
        async for event in workflow.astream_events(initial_state, version="v2"):
            event_type = event.get("event")
            event_name = event.get("name", "")

            # Filter and yield events we want to send to the UI:
            # - on_tool_start: When specialist agents begin execution
            # - on_tool_end: When specialist agents complete
            # - on_chain_end (name=LangGraph): When workflow completes with final state

            if event_type in ["on_tool_start", "on_tool_end"]:
                event_count += 1
                yield event
            elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
                # Workflow complete - yield the final state
                event_count += 1
                logger.info(
                    "Streaming analysis workflow completed",
                    extra={
                        "stage": "streaming_workflow_complete",
                        "total_events_streamed": event_count,
                    },
                )
                yield event

    except Exception as e:
        error_msg = f"CRITICAL ERROR in stream_analysis: {type(e).__name__}: {str(e)}"
        logger.error(
            error_msg,
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "stage": "streaming_workflow_error",
            },
        )
        raise

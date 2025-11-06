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
    print("[DEBUG] graph.py: About to compile workflow")
    sys.stdout.flush()

    try:
        compiled_workflow = workflow.compile()
        logger.info(
            "Workflow graph compiled successfully",
            extra={
                "stage": "workflow_compiled",
                "nodes": ["supervisor"],
            },
        )
        print("[DEBUG] graph.py: Workflow compiled successfully")
        sys.stdout.flush()
        return compiled_workflow
    except Exception as e:
        logger.error(
            "CRITICAL: Workflow compilation failed",
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
            },
        )
        print(f"[ERROR] graph.py: Workflow compilation failed: {e}")
        sys.stdout.flush()
        raise


async def run_analysis(
    transcript: str,
    metadata: dict | None = None,
) -> AgentState:
    """Run the complete podcast analysis workflow.

    Args:
        transcript: The podcast transcript text
        metadata: Optional metadata about the podcast

    Returns:
        Final agent state with all results from specialist agents
    """
    logger.info(
        "Starting analysis workflow",
        extra={
            "stage": "workflow_start",
            "transcript_length": len(transcript),
            "has_metadata": metadata is not None,
        },
    )

    # Initialize state
    initial_state: AgentState = {
        "transcript": transcript,
        "metadata": metadata,
        "current_stage": "initialized",
        "messages": [],
    }

    # Create and run workflow
    workflow = create_workflow()
    final_state = await workflow.ainvoke(initial_state)

    logger.info(
        "Analysis workflow completed",
        extra={
            "stage": "workflow_complete",
            "final_stage": final_state.get("current_stage"),
            "message_count": len(final_state.get("messages", [])),
        },
    )

    return final_state


async def stream_analysis(
    transcript: str,
    metadata: dict | None = None,
):
    """Stream the podcast analysis workflow with real-time updates.

    Uses LangGraph's astream_events() for fine-grained streaming that captures
    tool calls in real-time, allowing the UI to show progress as agents execute.

    Args:
        transcript: The podcast transcript text
        metadata: Optional metadata about the podcast

    Yields:
        Events from the workflow including tool calls, completions, and final state
    """
    # IMMEDIATE entry logging - this should appear FIRST
    print(f"[DEBUG] stream_analysis: ENTERED - transcript_length={len(transcript)}")
    sys.stdout.flush()

    logger.info(
        "Starting streaming analysis workflow",
        extra={
            "stage": "streaming_workflow_start",
            "transcript_length": len(transcript),
            "has_metadata": metadata is not None,
        },
    )
    print("[DEBUG] stream_analysis: After logger.info")
    sys.stdout.flush()

    # Initialize state
    initial_state: AgentState = {
        "transcript": transcript,
        "metadata": metadata,
        "current_stage": "initialized",
        "messages": [],
    }

    print("[DEBUG] stream_analysis: Calling create_workflow()")
    sys.stdout.flush()

    workflow = create_workflow()

    print("[DEBUG] stream_analysis: Workflow created, starting astream_events()")
    sys.stdout.flush()

    # Stream fine-grained events (tool calls, LLM calls, etc.)
    # version="v2" is the current stable event streaming format
    event_count = 0

    try:
        print("[DEBUG] stream_analysis: Entering astream_events loop")
        sys.stdout.flush()

        async for event in workflow.astream_events(initial_state, version="v2"):
            event_type = event.get("event")
            event_name = event.get("name", "")
            print(f"[DEBUG] stream_analysis: Event received - type={event_type}, name={event_name}")
            sys.stdout.flush()

            # Filter and yield events we want to send to the UI:
            # - on_tool_start: When specialist agents begin execution
            # - on_tool_end: When specialist agents complete
            # - on_chain_end (name=LangGraph): When workflow completes with final state

            if event_type in ["on_tool_start", "on_tool_end"]:
                event_count += 1
                print(f"[DEBUG] stream_analysis: Yielding {event_type} event (count={event_count})")
                sys.stdout.flush()
                yield event
            elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
                # Workflow complete - yield the final state
                event_count += 1
                print(f"[DEBUG] stream_analysis: Workflow complete, yielding final state (count={event_count})")
                sys.stdout.flush()
                logger.info(
                    "Streaming analysis workflow completed",
                    extra={
                        "stage": "streaming_workflow_complete",
                        "total_events_streamed": event_count,
                    },
                )
                yield event

        print(f"[DEBUG] stream_analysis: Exited astream_events loop - total events: {event_count}")
        sys.stdout.flush()

    except Exception as e:
        error_msg = f"CRITICAL ERROR in stream_analysis: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        sys.stdout.flush()
        logger.error(
            error_msg,
            extra={
                "error": str(e),
                "error_type": type(e).__name__,
                "stage": "streaming_workflow_error",
            },
        )
        raise

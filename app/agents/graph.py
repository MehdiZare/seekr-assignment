"""LangGraph workflow definition for podcast analysis."""

from typing import Literal

from langgraph.graph import END, StateGraph

from app.agents.nodes import (
    critic_node,
    fact_checker_node,
    model_a_node,
    model_b_node,
    supervisor_node,
)
from app.config import get_config
from app.models.state import AgentState, ModelResponses


def should_continue_critic_loop(state: AgentState) -> Literal["fact_checker", "end"]:
    """Determine whether to continue the critic loop or end.

    Args:
        state: Current agent state

    Returns:
        "fact_checker" to re-run fact checking, or "end" to finish
    """
    return "fact_checker" if state.get("should_continue", False) else "end"


def create_workflow() -> StateGraph:
    """Create and compile the LangGraph workflow.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("model_a", model_a_node)
    workflow.add_node("model_b", model_b_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("critic", critic_node)

    # Set both models as entry points - they run in TRUE parallel
    workflow.add_edge("__start__", "model_a")
    workflow.add_edge("__start__", "model_b")

    # Both models connect to supervisor (supervisor waits for both to complete)
    workflow.add_edge("model_a", "supervisor")
    workflow.add_edge("model_b", "supervisor")

    # Supervisor -> Fact Checker
    workflow.add_edge("supervisor", "fact_checker")

    # Fact Checker -> Critic
    workflow.add_edge("fact_checker", "critic")

    # Critic decides whether to loop back or end
    workflow.add_conditional_edges(
        "critic",
        should_continue_critic_loop,
        {
            "fact_checker": "fact_checker",  # Loop back for improvement
            "end": END,  # Finish processing
        },
    )

    # Compile and return
    return workflow.compile()


async def run_analysis(
    transcript: str,
    metadata: dict | None = None,
    max_critic_iterations: int | None = None,
) -> AgentState:
    """Run the complete podcast analysis workflow.

    Args:
        transcript: The podcast transcript text
        metadata: Optional metadata about the podcast
        max_critic_iterations: Maximum number of critic loop iterations (default from config)

    Returns:
        Final agent state with all results
    """
    config = get_config()

    # Get max iterations from config if not provided
    if max_critic_iterations is None:
        max_critic_iterations = config.app_settings.get("critic_loops", 2)

    # Initialize state
    initial_state: AgentState = {
        "transcript": transcript,
        "metadata": metadata,
        "model_responses": ModelResponses(),
        "critic_iterations": 0,
        "max_critic_iterations": max_critic_iterations,
        "current_stage": "initialized",
        "messages": [],
        "should_continue": False,
    }

    # Create and run workflow
    workflow = create_workflow()
    final_state = await workflow.ainvoke(initial_state)

    return final_state


async def stream_analysis(
    transcript: str,
    metadata: dict | None = None,
    max_critic_iterations: int | None = None,
):
    """Stream the podcast analysis workflow with real-time updates.

    Args:
        transcript: The podcast transcript text
        metadata: Optional metadata about the podcast
        max_critic_iterations: Maximum number of critic loop iterations

    Yields:
        State updates after each node execution
    """
    config = get_config()

    if max_critic_iterations is None:
        max_critic_iterations = config.app_settings.get("critic_loops", 2)

    initial_state: AgentState = {
        "transcript": transcript,
        "metadata": metadata,
        "model_responses": ModelResponses(),
        "critic_iterations": 0,
        "max_critic_iterations": max_critic_iterations,
        "current_stage": "initialized",
        "messages": [],
        "should_continue": False,
    }

    workflow = create_workflow()

    # Stream events
    async for event in workflow.astream(initial_state):
        yield event

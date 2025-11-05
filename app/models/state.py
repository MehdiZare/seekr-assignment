"""LangGraph state definition for the podcast analysis workflow."""

from typing import TypedDict

from app.models.outputs import (
    CriticFeedback,
    FactCheckOutput,
    ParallelAnalysis,
    SupervisorOutput,
)


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph workflow.

    This state is shared across all nodes in the graph and accumulates
    results as the workflow progresses.
    """

    # Input
    transcript: str
    metadata: dict | None

    # Parallel processing results (Models A & B)
    analysis_a: ParallelAnalysis | None
    analysis_b: ParallelAnalysis | None

    # Supervisor consolidation (Model C)
    supervisor_output: SupervisorOutput | None

    # Fact checking (Model D)
    fact_check_output: FactCheckOutput | None

    # Critic loop
    critic_feedback: CriticFeedback | None
    critic_iterations: int
    max_critic_iterations: int

    # Progress tracking for SSE
    current_stage: str
    messages: list[str]

    # Final output flag
    should_continue: bool

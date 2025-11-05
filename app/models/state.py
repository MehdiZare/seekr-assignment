"""LangGraph state definition for the podcast analysis workflow."""

from datetime import datetime
from operator import add
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from app.models.outputs import (
    CriticFeedback,
    FactCheckOutput,
    ParallelAnalysis,
    SupervisorOutput,
)


def merge_model_responses(left: "ModelResponses | None", right: "ModelResponses") -> "ModelResponses":
    """Merge two ModelResponses objects from parallel nodes.

    This allows model_a and model_b nodes to run in parallel and return separate
    ModelResponses objects that get merged into a single state.

    Args:
        left: Existing ModelResponses (or None if first update)
        right: New ModelResponses to merge in

    Returns:
        Merged ModelResponses with combined fields
    """
    if left is None:
        return right

    # Merge fields: later non-None values override earlier ones
    # Lists get concatenated
    return ModelResponses(
        model_a=right.model_a or left.model_a,
        model_b=right.model_b or left.model_b,
        supervisor=right.supervisor or left.supervisor,
        fact_check_current=right.fact_check_current or left.fact_check_current,
        critic_current=right.critic_current or left.critic_current,
        fact_check_iterations=left.fact_check_iterations + right.fact_check_iterations,
    )


class FactCheckIteration(BaseModel):
    """Single iteration in the fact-checking improvement loop.

    This tracks each round of fact-checking and critic feedback,
    allowing us to see how the verification improves over iterations.
    """

    iteration: int = Field(..., ge=0, description="Iteration number (0-indexed)")
    fact_check: FactCheckOutput = Field(..., description="Fact-checking results for this iteration")
    critic_feedback: CriticFeedback = Field(..., description="Critic's feedback on this iteration")
    timestamp: datetime = Field(default_factory=datetime.now, description="When this iteration completed")

    class Config:
        frozen = False  # Allow mutations if needed


class ModelResponses(BaseModel):
    """Container for all model outputs throughout the workflow.

    This provides clean separation of data from different processing paths:
    - Parallel analysis (Model A & B)
    - Supervisor consolidation (Model C)
    - Fact-checking (Model D with iteration history)
    - Critic feedback
    """

    # Parallel processing results (Models A & B)
    model_a: ParallelAnalysis | None = Field(
        None, description="Analysis from Model A (Claude Haiku)"
    )
    model_b: ParallelAnalysis | None = Field(
        None, description="Analysis from Model B (Llama Maverick)"
    )

    # Supervisor consolidation (Model C)
    supervisor: SupervisorOutput | None = Field(
        None, description="Consolidated analysis from Supervisor (Claude Sonnet)"
    )

    # Current fact-checking output (Model D)
    fact_check_current: FactCheckOutput | None = Field(
        None, description="Most recent fact-checking results"
    )

    # Current critic feedback
    critic_current: CriticFeedback | None = Field(
        None, description="Most recent critic feedback"
    )

    # Full history of fact-checking iterations
    fact_check_iterations: list[FactCheckIteration] = Field(
        default_factory=list,
        description="Complete history of all fact-checking improvement iterations"
    )

    class Config:
        frozen = False  # Allow mutations
        validate_assignment = True  # Validate when fields are assigned


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

    # NEW: Unified model responses container (Pydantic)
    # Use custom reducer to merge parallel node outputs
    model_responses: Annotated[ModelResponses, merge_model_responses]

    # Critic loop control
    critic_iterations: int
    max_critic_iterations: int

    # Progress tracking for SSE
    # Use Annotated with operator.add to merge messages from parallel nodes
    current_stage: str
    messages: Annotated[list[str], add]

    # Final output flag
    should_continue: bool

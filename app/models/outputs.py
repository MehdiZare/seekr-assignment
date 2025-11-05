"""Output models for all agent stages."""

from pydantic import BaseModel, Field


# Parallel Processing Outputs (Models A & B)
class ParallelAnalysis(BaseModel):
    """Output from parallel processing agents (Model A or B)."""

    summary: str = Field(
        ...,
        description="Initial summary of the podcast content",
    )
    key_points: list[str] = Field(
        ...,
        description="List of key points identified in the transcript",
        min_length=3,
    )
    topics: list[str] = Field(
        ...,
        description="Main topics discussed",
        min_length=1,
    )
    confidence: float = Field(
        ...,
        description="Confidence score in the analysis (0-1)",
        ge=0.0,
        le=1.0,
    )


# Supervisor Output (Model C)
class Quote(BaseModel):
    """A notable quote from the podcast."""

    text: str = Field(..., description="The quote text")
    speaker: str | None = Field(None, description="Who said it (if identifiable)")
    context: str = Field(..., description="Brief context around the quote")


class SupervisorOutput(BaseModel):
    """Consolidated output from the supervisor agent."""

    final_summary: str = Field(
        ...,
        description="Comprehensive 200-300 word summary consolidating both analyses",
        min_length=200,
        max_length=400,
    )
    main_topics: list[str] = Field(
        ...,
        description="Consolidated list of main topics",
        min_length=1,
    )
    key_takeaways: list[str] = Field(
        ...,
        description="Key takeaways from the podcast",
        min_length=3,
    )
    notable_quotes: list[Quote] = Field(
        default_factory=list,
        description="Notable quotes from the podcast",
    )
    claims_to_verify: list[str] = Field(
        ...,
        description="Factual claims that need verification",
        min_length=1,
    )
    reasoning: str = Field(
        ...,
        description="Reasoning behind the consolidation decisions",
    )


# Fact Checker Output (Model D)
class Source(BaseModel):
    """A source used for verification."""

    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title/description")
    relevance: float = Field(
        ...,
        description="Relevance score (0-1)",
        ge=0.0,
        le=1.0,
    )


class VerifiedClaim(BaseModel):
    """A verified factual claim."""

    claim: str = Field(..., description="The original claim")
    verification_status: str = Field(
        ...,
        description="Status: 'verified', 'partially_verified', 'unverified', or 'false'",
    )
    confidence: float = Field(
        ...,
        description="Confidence in verification (0-1)",
        ge=0.0,
        le=1.0,
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="Sources used for verification",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning behind the verification result",
    )
    additional_context: str | None = Field(
        None,
        description="Additional context or nuance discovered during verification",
    )


class FactCheckOutput(BaseModel):
    """Output from the fact-checking agent."""

    verified_claims: list[VerifiedClaim] = Field(
        ...,
        description="All verified claims",
        min_length=1,
    )
    overall_reliability: float = Field(
        ...,
        description="Overall reliability score of the podcast content (0-1)",
        ge=0.0,
        le=1.0,
    )
    research_quality: float = Field(
        ...,
        description="Quality of the research conducted (0-1)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        ...,
        description="Overall reasoning about the fact-checking process",
    )


# Critic Output
class CriticFeedback(BaseModel):
    """Feedback from the critic agent."""

    research_is_sufficient: bool = Field(
        ...,
        description="Whether the research quality is sufficient",
    )
    missing_verifications: list[str] = Field(
        default_factory=list,
        description="Claims that need better verification",
    )
    suggested_improvements: list[str] = Field(
        default_factory=list,
        description="Specific improvements to make",
    )
    quality_score: float = Field(
        ...,
        description="Overall quality score (0-1)",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        ...,
        description="Reasoning behind the feedback",
    )


# Final Output
class FinalOutput(BaseModel):
    """Final output combining all stages."""

    summary: SupervisorOutput = Field(
        ...,
        description="Consolidated summary from supervisor",
    )
    fact_check: FactCheckOutput = Field(
        ...,
        description="Fact-checking results",
    )
    confidence_in_analysis: float = Field(
        ...,
        description="Overall confidence in the complete analysis (0-1)",
        ge=0.0,
        le=1.0,
    )
    critic_iterations: int = Field(
        ...,
        description="Number of critic loop iterations performed",
        ge=0,
    )
    processing_notes: str = Field(
        default="",
        description="Any additional notes about the processing",
    )

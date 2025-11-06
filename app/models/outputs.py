"""Output models for all agent stages."""

from pydantic import BaseModel, Field


# ============================================================================
# SPECIALIST AGENT OUTPUTS (New Architecture)
# ============================================================================

class SummaryOutput(BaseModel):
    """Output from the Summarizing Agent."""

    summary: str = Field(
        ...,
        description="Comprehensive 200-300 word summary of the podcast episode",
        min_length=200,
        max_length=400,
    )
    core_theme: str = Field(
        ...,
        description="The central theme or main topic of the episode",
    )
    key_discussions: list[str] = Field(
        ...,
        description="Main discussion points covered in the episode",
        min_length=2,
    )
    outcomes_and_opinions: list[str] = Field(
        ...,
        description="Key outcomes, conclusions, or opinions shared",
        min_length=1,
    )
    reasoning: str = Field(
        ...,
        description="Agent's reasoning process for creating this summary",
    )


class QuoteWithTimestamp(BaseModel):
    """A notable quote with timestamp from the podcast."""

    text: str = Field(..., description="The quote text")
    speaker: str | None = Field(None, description="Who said it")
    timestamp: str | None = Field(None, description="Timestamp in the transcript (e.g., '00:15:23')")
    context: str = Field(..., description="Brief context around the quote")


class FactualStatement(BaseModel):
    """A factual claim that needs verification."""

    statement: str = Field(..., description="The factual claim or statement")
    speaker: str | None = Field(None, description="Who made the claim")
    context: str = Field(..., description="Context around the statement")
    timestamp: str | None = Field(None, description="When in the episode this was said")


class NotesOutput(BaseModel):
    """Output from the Note Extraction Agent."""

    top_takeaways: list[str] = Field(
        ...,
        description="Top 5 key takeaways from the episode",
        min_length=5,
        max_length=5,
    )
    notable_quotes: list[QuoteWithTimestamp] = Field(
        ...,
        description="Notable quotes with timestamps",
        min_length=1,
    )
    topics: list[str] = Field(
        ...,
        description="Topics for tagging the podcast (e.g., 'remote work', 'productivity', 'technology')",
        min_length=3,
    )
    factual_statements: list[FactualStatement] = Field(
        ...,
        description="Factual statements extracted for verification",
        min_length=1,
    )
    reasoning: str = Field(
        ...,
        description="Agent's reasoning process for extracting these notes",
    )


# ============================================================================
# FACT CHECKING OUTPUTS
# ============================================================================

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
        description="Status: 'fact-checked' (verified with credible sources), 'unverified' (no credible sources), or 'declined' (contradicted by credible evidence)",
    )
    confidence: float = Field(
        ...,
        description="Confidence in verification (0-1)",
        ge=0.0,
        le=1.0,
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="Sources used for verification (credible websites)",
    )
    reasoning: str = Field(
        ...,
        description="Reasoning behind the verification result, including search process",
    )
    additional_context: str | None = Field(
        None,
        description="Additional context or nuance discovered during verification",
    )


class FactCheckOutput(BaseModel):
    """Output from the fact-checking agent."""

    verified_claims: list[VerifiedClaim] = Field(
        ...,
        description="All verified claims (empty if no factual statements to verify)",
        min_length=0,
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

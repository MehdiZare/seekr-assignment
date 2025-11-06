"""Input models for podcast transcripts."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TranscriptInput(BaseModel):
    """Input model for podcast transcript analysis."""

    transcript: str = Field(
        ...,
        description="The full podcast transcript text",
        min_length=100,
    )
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional metadata about the podcast (title, date, speakers, etc.)",
    )

    @field_validator("transcript", mode="before")
    @classmethod
    def convert_transcript_array(cls, v: Any) -> str:
        """Convert transcript array to string if needed."""
        if isinstance(v, list):
            # Assume it's an array of transcript segments
            # Each segment should have 'speaker' and 'text' fields
            lines = []
            for segment in v:
                if isinstance(segment, dict):
                    speaker = segment.get("speaker", "Speaker")
                    text = segment.get("text", "")
                    if text:
                        lines.append(f"{speaker}: {text}")
            return "\n".join(lines)
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transcript": "Speaker 1: Welcome to Tech Talk...",
                "metadata": {
                    "title": "Tech Talk Episode 42",
                    "date": "2024-01-15",
                    "speakers": ["Alice", "Bob"],
                },
            }
        }
    )

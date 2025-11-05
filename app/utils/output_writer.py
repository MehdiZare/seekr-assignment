"""Output file generation for podcast analysis results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.models.outputs import FinalOutput


def _format_fact_check_table_markdown(fact_check: Any) -> str:
    """Format fact-check results as a Markdown table.

    Args:
        fact_check: FactCheckOutput object

    Returns:
        Markdown formatted table string
    """
    if not fact_check or not fact_check.claims:
        return "No fact-check claims available.\n"

    table = "| Claim | Status | Confidence | Evidence |\n"
    table += "|-------|--------|------------|----------|\n"

    for claim in fact_check.claims:
        # Escape pipe characters in content
        claim_text = claim.claim.replace("|", "\\|")
        evidence = claim.evidence.replace("|", "\\|") if claim.evidence else "N/A"

        table += f"| {claim_text} | {claim.status} | {claim.confidence:.1%} | {evidence} |\n"

    return table


def _format_fact_check_table_json(fact_check: Any) -> list[dict]:
    """Format fact-check results as a list of dicts for JSON.

    Args:
        fact_check: FactCheckOutput object

    Returns:
        List of claim dictionaries
    """
    if not fact_check or not fact_check.claims:
        return []

    return [
        {
            "claim": claim.claim,
            "status": claim.status,
            "confidence": round(claim.confidence, 3),
            "evidence": claim.evidence or "N/A",
        }
        for claim in fact_check.claims
    ]


def generate_json_output(
    final_output: FinalOutput,
    output_dir: str | Path = "output",
    filename: str | None = None,
) -> Path:
    """Generate JSON output file with analysis results.

    Args:
        final_output: FinalOutput object with all results
        output_dir: Directory to save the output file
        filename: Optional custom filename (defaults to timestamp-based name)

    Returns:
        Path to the generated file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.json"

    file_path = output_path / filename

    # Build JSON structure
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "critic_iterations": final_output.critic_iterations,
            "confidence_in_analysis": round(final_output.confidence_in_analysis, 3),
        },
        "summary": {
            "main_topics": final_output.summary.main_topics if final_output.summary else [],
            "key_takeaways": final_output.summary.key_takeaways if final_output.summary else [],
            "final_summary": final_output.summary.final_summary if final_output.summary else "",
        },
        "notes": final_output.processing_notes or "",
        "fact_check": {
            "overall_reliability": round(final_output.fact_check.overall_reliability, 3)
            if final_output.fact_check
            else 0.0,
            "research_quality": round(final_output.fact_check.research_quality, 3)
            if final_output.fact_check
            else 0.0,
            "verified_claims": _format_fact_check_table_json(final_output.fact_check),
        },
    }

    # Write JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return file_path


def generate_markdown_output(
    final_output: FinalOutput,
    output_dir: str | Path = "output",
    filename: str | None = None,
) -> Path:
    """Generate Markdown output file with analysis results.

    Args:
        final_output: FinalOutput object with all results
        output_dir: Directory to save the output file
        filename: Optional custom filename (defaults to timestamp-based name)

    Returns:
        Path to the generated file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.md"

    file_path = output_path / filename

    # Build Markdown content
    md_content = "# Podcast Analysis Report\n\n"

    # Metadata
    md_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"**Critic Iterations:** {final_output.critic_iterations}\n\n"
    md_content += f"**Overall Confidence:** {final_output.confidence_in_analysis:.1%}\n\n"
    md_content += "---\n\n"

    # Summary Section
    md_content += "## Summary\n\n"
    if final_output.summary:
        md_content += f"{final_output.summary.final_summary}\n\n"

        md_content += "### Topics Discussed\n\n"
        for topic in final_output.summary.main_topics:
            md_content += f"- {topic}\n"
        md_content += "\n"

        md_content += "### Key Takeaways\n\n"
        for takeaway in final_output.summary.key_takeaways:
            md_content += f"- {takeaway}\n"
        md_content += "\n"
    else:
        md_content += "*No summary available*\n\n"

    md_content += "---\n\n"

    # Notes Section
    md_content += "## Processing Notes\n\n"
    md_content += f"{final_output.processing_notes or '*No processing notes*'}\n\n"
    md_content += "---\n\n"

    # Fact-Check Section
    md_content += "## Fact-Check Results\n\n"
    if final_output.fact_check:
        md_content += f"**Overall Reliability:** {final_output.fact_check.overall_reliability:.1%}\n\n"
        md_content += f"**Research Quality:** {final_output.fact_check.research_quality:.1%}\n\n"

        md_content += "### Verified Claims\n\n"
        md_content += _format_fact_check_table_markdown(final_output.fact_check)
        md_content += "\n"
    else:
        md_content += "*No fact-check results available*\n\n"

    # Write Markdown file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return file_path


def generate_outputs(
    final_output: FinalOutput,
    output_dir: str | Path = "output",
    base_filename: str | None = None,
) -> tuple[Path, Path]:
    """Generate both JSON and Markdown output files.

    Args:
        final_output: FinalOutput object with all results
        output_dir: Directory to save the output files
        base_filename: Optional base filename (without extension)

    Returns:
        Tuple of (json_path, markdown_path)
    """
    if base_filename:
        json_filename = f"{base_filename}.json"
        md_filename = f"{base_filename}.md"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"analysis_{timestamp}.json"
        md_filename = f"analysis_{timestamp}.md"

    json_path = generate_json_output(final_output, output_dir, json_filename)
    md_path = generate_markdown_output(final_output, output_dir, md_filename)

    return json_path, md_path

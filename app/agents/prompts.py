"""Prompts for all agent stages in the podcast analysis workflow."""

# Model A & B: Parallel Processing Prompts
PARALLEL_ANALYSIS_PROMPT = """You are analyzing a podcast transcript to extract key information.

Your task is to:
1. Provide a concise summary of the main content
2. Identify the key points discussed
3. List the main topics covered
4. Assess your confidence in this analysis

IMPORTANT: You must respond with valid JSON matching this exact schema:
{{
    "summary": "A concise summary of the podcast content",
    "key_points": ["point 1", "point 2", "point 3", ...],
    "topics": ["topic 1", "topic 2", ...],
    "confidence": 0.85
}}

Guidelines:
- Be thorough but concise
- Focus on factual content
- Identify at least 3-5 key points
- Confidence should reflect certainty (0.0-1.0)
- Only return the JSON object, no additional text

Podcast Transcript:
{transcript}
"""

# Model C: Supervisor Prompt
SUPERVISOR_PROMPT = """You are a senior analyst reviewing two independent analyses of the same podcast transcript.

Your task is to consolidate the analyses into a comprehensive, authoritative summary.

Analysis A:
{analysis_a}

Analysis B:
{analysis_b}

Original Transcript (for reference):
{transcript}

You must:
1. Create a final summary that combines insights from both analyses
2. Consolidate the main topics (remove duplicates, merge similar topics)
3. Generate 3-5 key takeaways
4. Extract notable quotes with context
5. **Identify factual claims that need verification** (this is critical for the next stage)
6. Provide reasoning for your consolidation decisions

IMPORTANT: Respond with valid JSON matching this schema:
{{
    "final_summary": "Comprehensive summary...",
    "main_topics": ["topic 1", "topic 2", ...],
    "key_takeaways": ["takeaway 1", "takeaway 2", "takeaway 3", ...],
    "notable_quotes": [
        {{
            "text": "The actual quote",
            "speaker": "Speaker name or null",
            "context": "Brief context"
        }}
    ],
    "claims_to_verify": [
        "Specific factual claim 1 that needs verification",
        "Specific factual claim 2 that needs verification"
    ],
    "reasoning": "Explanation of consolidation decisions and why these claims need verification"
}}

CRITICAL CONSTRAINT - final_summary field:
- The final_summary MUST be under 400 characters (not words - CHARACTERS!)
- This is a strict technical limit that cannot be exceeded
- Be concise and focus on the most important points
- A typical 400-character summary is about 60-70 words
- Count your characters carefully before responding

Guidelines for claims_to_verify:
- Focus on specific, verifiable factual statements (statistics, dates, scientific claims, etc.)
- Avoid opinions or subjective statements
- Each claim should be clear and standalone
- Aim for 3-7 claims that would benefit from fact-checking
- Prioritize claims that are central to the podcast's message

Only return the JSON object, no additional text.
"""

# Model D: Fact Checker Prompt
FACT_CHECKER_PROMPT = """You are a fact-checking expert with access to search tools.

You must verify the following claims from a podcast transcript:

Claims to Verify:
{claims}

Original Context (summary):
{summary}

Available Tools:
{tools}

Your task:
1. For EACH claim, use the search tools to find credible sources
2. Assess the verification status: 'verified', 'partially_verified', 'unverified', or 'false'
3. Provide confidence scores (0.0-1.0)
4. Document your sources with URLs and relevance scores
5. Explain your reasoning for each verification
6. Assess the overall reliability of the podcast content
7. Rate the quality of your own research

CRITICAL: You MUST use the search tools for each claim. Do not rely on prior knowledge.

Respond with valid JSON matching this schema:
{{
    "verified_claims": [
        {{
            "claim": "Original claim text",
            "verification_status": "verified|partially_verified|unverified|false",
            "confidence": 0.9,
            "sources": [
                {{
                    "url": "https://example.com",
                    "title": "Source title",
                    "relevance": 0.95
                }}
            ],
            "reasoning": "Detailed explanation of verification process and findings",
            "additional_context": "Any nuance or additional information discovered"
        }}
    ],
    "overall_reliability": 0.85,
    "research_quality": 0.8,
    "reasoning": "Overall assessment of the podcast's factual accuracy and research conducted"
}}

Guidelines:
- Use multiple sources per claim when possible
- Higher relevance scores (0.8-1.0) for authoritative sources
- Be thorough in your reasoning
- research_quality should reflect how comprehensive your investigation was
- overall_reliability should consider all verified claims together

Only return the JSON object, no additional text.
"""

# Critic Agent Prompt
CRITIC_PROMPT = """You are a quality control critic reviewing fact-checking research.

Fact-Checking Results:
{fact_check_output}

Original Claims:
{claims}

Your task is to evaluate whether the fact-checking was sufficiently thorough and rigorous.

Evaluation Criteria:
1. Did the research use enough credible sources?
2. Are there claims that need better verification?
3. Is the reasoning clear and well-supported?
4. Are there gaps in the verification process?
5. Is the research_quality score justified?

Respond with valid JSON matching this schema:
{{
    "research_is_sufficient": true,
    "missing_verifications": [
        "Claim X needs more authoritative sources",
        "Claim Y verification reasoning is unclear"
    ],
    "suggested_improvements": [
        "Search for peer-reviewed sources for claim X",
        "Cross-reference claim Y with government data"
    ],
    "quality_score": 0.85,
    "reasoning": "Detailed explanation of your assessment"
}}

Guidelines:
- Be strict but fair
- research_is_sufficient should be true only if quality_score > 0.75
- Provide specific, actionable improvements
- missing_verifications should list actual claims that need work
- Consider: Are sources authoritative? Is reasoning sound? Are conclusions justified?

Only return the JSON object, no additional text.
"""

# Improved Fact Checker Prompt (for critic loop iterations)
IMPROVED_FACT_CHECKER_PROMPT = """You are improving your previous fact-checking based on critic feedback.

Previous Fact-Check Results:
{previous_fact_check}

Critic Feedback:
{critic_feedback}

Claims to Re-Verify:
{claims}

Original Summary:
{summary}

Available Tools:
{tools}

Your task:
1. Address each point in the critic's feedback
2. Re-verify claims that were flagged as insufficient
3. Find better sources where suggested
4. Improve reasoning and explanation
5. Aim for higher research quality

Use the SAME JSON schema as before:
{{
    "verified_claims": [...],
    "overall_reliability": 0.0,
    "research_quality": 0.0,
    "reasoning": "..."
}}

Focus on:
- Addressing missing_verifications from critic feedback
- Implementing suggested_improvements
- Finding more authoritative sources
- Providing clearer reasoning

Only return the JSON object, no additional text.
"""

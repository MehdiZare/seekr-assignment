# Podcast Agent - AI-Powered Podcast Analysis

A sophisticated multi-agent system built with LangGraph that analyzes podcast transcripts using parallel processing, supervisor consolidation, deep fact-checking, and critic loops.

## Features

- **Parallel LLM Processing**: Models A & B (Claude Haiku + GPT-4o-mini) analyze transcripts simultaneously
- **Supervisor Consolidation**: Model C (Claude Sonnet) reviews and merges analyses
- **Deep Fact-Checking**: Model D (GPT-4o) verifies claims using multiple search tools
- **Critic Loop**: Quality control with iterative improvement
- **Real-Time Progress**: Server-Sent Events (SSE) for live updates
- **Single Container Deployment**: Easy Docker deployment
- **Professional UI**: Clean, responsive interface built with Tailwind CSS

## Architecture

### Agent Workflow

```
START → Parallel(A,B) → Supervisor(C) → FactCheck(D) → Critic → [Loop or END]
```

### Models Used

- **Model A**: Claude 3 Haiku (fast, cost-effective parallel processing)
- **Model B**: GPT-4o-mini (diverse perspective for parallel processing)
- **Model C**: Claude 3.5 Sonnet (advanced reasoning for supervision)
- **Model D**: GPT-4o (comprehensive fact-checking with tool access)

### Search Tools

- **Tavily**: Advanced search with comprehensive filtering
- **Google Serper**: Current web search results
- **Brave Search**: Privacy-focused search engine

*(At least one search tool API key required)*

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose (for containerized deployment)
- API Keys:
  - Anthropic API key (for Claude models)
  - OpenAI API key (for GPT models)
  - At least one search tool API key (Tavily, Serper, or Brave)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd assignment
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

3. **Choose your deployment method**

#### Option A: Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application at http://localhost:8000
```

#### Option B: Local Development with uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration

### Environment Variables (.env)

```bash
# LLM API Keys (Required)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Search Tool API Keys (At least one required)
TAVILY_API_KEY=your_tavily_key_here
SERPER_API_KEY=your_serper_key_here
BRAVE_API_KEY=your_brave_key_here
```

### Application Settings (config.yaml)

```yaml
models:
  model_a:
    provider: "anthropic"
    name: "claude-3-haiku-20240307"
    temperature: 0.3
    max_tokens: 2000
  # ... (see config.yaml for full configuration)

search_tools:
  tavily:
    search_depth: "advanced"
    max_results: 10
  # ...

app:
  debug: false
  max_retries: 3
  critic_loops: 2  # Maximum critic loop iterations
  stream_delay_ms: 100
```

## Usage

### Web Interface

1. Navigate to `http://localhost:8000`
2. Choose input method:
   - **Sample Transcripts**: Select from pre-loaded examples
   - **Upload File**: Upload .txt or .json file
   - **Paste Text**: Directly paste transcript text
3. Click "Analyze" and watch real-time progress
4. Review results including summary, topics, and fact-checked claims

### API Endpoints

#### Analyze Transcript

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Your podcast transcript here...",
    "metadata": {
      "title": "Episode Title",
      "date": "2024-01-15"
    }
  }'
```

#### Upload File

```bash
curl -X POST http://localhost:8000/api/analyze/file \
  -F "file=@transcript.txt"
```

#### Health Check

```bash
curl http://localhost:8000/api/health
```

#### List Samples

```bash
curl http://localhost:8000/api/samples
```

## Project Structure

```
podcast-agent/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   ├── config.py                # Configuration management
│   ├── models/
│   │   ├── transcript.py        # Input models
│   │   ├── outputs.py           # Output schemas
│   │   └── state.py             # LangGraph state
│   ├── agents/
│   │   ├── graph.py             # LangGraph workflow
│   │   ├── nodes.py             # Agent node implementations
│   │   ├── tools.py             # Search tools
│   │   └── prompts.py           # Agent prompts
│   ├── static/
│   │   └── index.html           # Single-file UI
│   └── data/                    # Sample transcripts (user-provided)
├── config.yaml                  # Application configuration
├── pyproject.toml              # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## Output Format

```json
{
  "summary": {
    "final_summary": "200-300 word consolidated summary",
    "main_topics": ["topic1", "topic2"],
    "key_takeaways": ["takeaway1", "takeaway2"],
    "notable_quotes": [...]
  },
  "fact_check": {
    "verified_claims": [
      {
        "claim": "Specific claim",
        "verification_status": "verified",
        "confidence": 0.92,
        "sources": [
          {
            "url": "https://example.com",
            "title": "Source Title",
            "relevance": 0.95
          }
        ],
        "reasoning": "Detailed explanation"
      }
    ],
    "overall_reliability": 0.87,
    "research_quality": 0.85
  },
  "confidence_in_analysis": 0.87,
  "critic_iterations": 2
}
```

## Development

### Adding Sample Transcripts

Place JSON files in `app/data/`:

```json
{
  "transcript": "Full transcript text here...",
  "metadata": {
    "title": "Episode Title",
    "date": "2024-01-15",
    "speakers": ["Speaker 1", "Speaker 2"]
  }
}
```

### Running Tests

```bash
# Install test dependencies
uv pip install pytest pytest-asyncio

# Run tests
pytest
```

### Code Quality

```bash
# Install development tools
uv pip install ruff black mypy

# Format code
black app/

# Lint
ruff check app/

# Type check
mypy app/
```

## Performance Metrics

### Expected Processing Times

- Parallel Processing: 5-10 seconds
- Supervisor Review: 3-5 seconds
- Fact Checking: 10-20 seconds
- Critic Loop: 5-10 seconds per iteration
- **Total**: 30-60 seconds per transcript

### Token Usage (Estimated)

- Models A & B: ~2,000 tokens each
- Model C: ~4,000 tokens
- Model D: ~8,000 tokens (with tool calls)
- **Total**: ~16,000 tokens per analysis

## Deployment

### AWS App Runner

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t podcast-agent .
docker tag podcast-agent:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/podcast-agent:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/podcast-agent:latest

# Deploy via AWS Console or CLI
```

### Environment Variables in Production

Use AWS Secrets Manager or similar service for API keys:

```bash
aws secretsmanager create-secret --name podcast-agent/api-keys \
  --secret-string '{"ANTHROPIC_API_KEY":"...","OPENAI_API_KEY":"..."}'
```

## Troubleshooting

### Common Issues

**API Key Errors**
- Verify all required API keys are set in `.env`
- Check API key validity and rate limits

**Search Tool Errors**
- Ensure at least one search tool API key is configured
- Check search tool API quotas

**Docker Build Issues**
- Clear Docker cache: `docker-compose build --no-cache`
- Ensure sufficient disk space

**Import Errors**
- Reinstall dependencies: `uv pip install --force-reinstall -e .`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [FastAPI](https://fastapi.tiangolo.com/)
- UI styled with [Tailwind CSS](https://tailwindcss.com/)
- Package management with [uv](https://github.com/astral-sh/uv)

## Support

For issues, questions, or contributions, please open an issue on GitHub.

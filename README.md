# Podcast Agent - AI-Powered Podcast Analysis

A sophisticated multi-agent system built with LangGraph that analyzes podcast transcripts using parallel processing, supervisor consolidation, deep fact-checking, and critic loops.

## Features

- **True Parallel LLM Processing**: Models A & B (Claude Haiku + GPT-4o-mini) analyze transcripts simultaneously using LangGraph parallel execution
- **Supervisor Consolidation**: Model C (Claude Sonnet) reviews and merges analyses
- **Deep Fact-Checking**: Model D (GPT-4o) verifies claims using multiple search tools
- **Critic Loop**: Quality control with iterative improvement (configurable iterations)
- **Real-Time Progress**: Server-Sent Events (SSE) for live updates with detailed agent reasoning
- **Auto-Validation Retry**: LLM self-correction mechanism for Pydantic validation failures
- **Output Files**: Automatically generates JSON and Markdown reports with fact-check tables
- **Single Container Deployment**: Easy Docker deployment to Render.com or any Docker platform
- **Professional UI**: Clean, responsive interface built with Tailwind CSS
- **Comprehensive Logging**: Detailed logs showing agent reasoning steps and decision-making

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
│   │   ├── outputs.py           # Output schemas (Pydantic)
│   │   └── state.py             # LangGraph state (TypedDict + Pydantic)
│   ├── agents/
│   │   ├── graph.py             # LangGraph workflow definition
│   │   ├── nodes.py             # Agent node implementations
│   │   ├── tools.py             # Search tools (Tavily, Serper, Brave)
│   │   └── prompts.py           # Agent prompts
│   ├── utils/
│   │   └── output_writer.py     # JSON/Markdown output generation
│   ├── static/
│   │   └── index.html           # Single-file UI (Tailwind CSS)
│   └── data/                    # Sample transcripts (user-provided)
├── output/                      # Generated analysis files (auto-created)
│   ├── analysis_*.json
│   └── analysis_*.md
├── config.yaml                  # Application configuration
├── pyproject.toml              # Python dependencies (managed with uv)
├── Dockerfile                   # Docker build configuration
├── docker-compose.yml          # Local Docker Compose setup
├── render.yaml                 # Render.com deployment blueprint
├── .env.example                # Environment variables template
├── DEPLOY.md                   # Deployment guide
└── README.md                   # This file
```

## Output Files

The application automatically generates two output files for each analysis in the `output/` directory:

### 1. JSON Output (`analysis_YYYYMMDD_HHMMSS.json`)

```json
{
  "metadata": {
    "generated_at": "2024-11-05T14:30:45",
    "critic_iterations": 2,
    "confidence_in_analysis": 0.87
  },
  "summary": {
    "topics": ["topic1", "topic2"],
    "key_points": ["point1", "point2"],
    "final_summary": "Consolidated summary (under 400 characters)"
  },
  "notes": "Processing notes from all agents",
  "fact_check": {
    "overall_reliability": 0.87,
    "claims": [
      {
        "claim": "Specific claim made in podcast",
        "status": "verified",
        "confidence": 0.92,
        "evidence": "Supporting evidence from search results"
      }
    ],
    "issues_found": ["issue1", "issue2"],
    "recommendations": ["recommendation1"]
  }
}
```

### 2. Markdown Output (`analysis_YYYYMMDD_HHMMSS.md`)

A human-readable report containing:
- **Metadata**: Generation time, iterations, overall confidence
- **Summary**: Consolidated analysis with topics and key points
- **Processing Notes**: Agent reasoning and decision logs
- **Fact-Check Results**: Formatted table with claims, status, confidence, and evidence
- **Recommendations**: Suggested improvements or areas of concern

## Logging and Agent Reasoning

The application provides detailed logging to help you understand the agent decision-making process:

### Console Logs

The application logs show:
- **Agent Progress**: Each node execution with timing information
- **Model Invocations**: LLM calls with retry attempts (if validation fails)
- **Tool Usage**: Search tool invocations and results
- **State Updates**: Changes to the workflow state
- **Output Generation**: File paths for generated reports

Example log output:
```
INFO:app.agents.nodes:Model A analysis complete: 5 topics, 8 key points identified
INFO:app.agents.nodes:Model B analysis complete: 4 topics, 7 key points identified
INFO:app.agents.nodes:Supervisor consolidation complete: 400 char summary, 6 topics, 10 key points
INFO:app.agents.nodes:Fact-checker invoking search tools for 3 claims
INFO:app.agents.nodes:Critic iteration 1/2: Overall reliability 0.85, continuing iteration
INFO:app.main:Output files generated: output/analysis_20241105_143045.json, output/analysis_20241105_143045.md
```

### Real-Time Progress Updates

The UI displays detailed progress messages from each agent:
- Model A & B: Number of topics and key points identified
- Supervisor: Summary length and consolidated data
- Fact-Checker: Number of claims verified, tools used
- Critic: Reliability scores, iteration decisions

### Validation Retry Mechanism

When Pydantic validation fails, the system:
1. Logs the validation error details
2. Sends error feedback to the LLM
3. Retries up to 3 times (configurable in `config.yaml`)
4. Auto-truncates strings if retries fail
5. Logs each retry attempt with detailed error messages

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

### Render.com (Recommended)

Render.com provides easy Docker-based deployment with automatic HTTPS and custom domains.

#### Quick Deploy to Render

1. **Push your code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push
   ```

2. **Create Render Account**
   - Sign up at [render.com](https://render.com)
   - Connect your GitHub account

3. **Deploy via Blueprint**
   - Click "New +" → "Blueprint"
   - Connect your repository
   - Render will automatically detect `render.yaml`
   - Click "Apply"

4. **Set Environment Variables**
   - Go to your service in Render Dashboard
   - Navigate to "Environment" tab
   - Add required API keys:
     - `ANTHROPIC_API_KEY`
     - `OPENAI_API_KEY`
     - `TAVILY_API_KEY` (or `SERPER_API_KEY` or `BRAVE_API_KEY`)
   - Click "Save Changes"

5. **Access Your App**
   - First deploy takes ~5-10 minutes
   - Subsequent deploys are faster (cached layers)
   - Your app will be available at: `https://your-app-name.onrender.com`

#### Free Tier vs Production

**Free Tier:**
- 750 hours/month free
- Spins down after 15 minutes of inactivity
- First request after spin-down takes ~30 seconds

**Starter Plan ($7/month):**
```yaml
# In render.yaml
plan: starter  # Doesn't spin down
```

#### Monitoring

- View logs in Render Dashboard
- Auto-deploys on git push to main branch
- Health check available at `/api/health`

See [DEPLOY.md](DEPLOY.md) for detailed deployment instructions.

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

Use secrets management services for API keys:

**Render.com:** Use the Environment tab in the dashboard

**AWS:** Use AWS Secrets Manager
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

"""FastAPI application for podcast analysis."""

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.agents.graph import stream_analysis
from app.config import get_config
from app.models.outputs import FinalOutput
from app.models.transcript import TranscriptInput

app = FastAPI(
    title="Podcast Agent",
    description="AI-powered podcast analysis with multi-agent workflow",
    version="0.1.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    config = get_config()
    # Set up LangSmith tracing if enabled
    config.setup_langsmith()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed logging."""
    logger.error(f"Validation error for {request.url}: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/")
async def root():
    """Serve the main UI."""
    index_path = static_path / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")

    with open(index_path, "r") as f:
        from fastapi.responses import HTMLResponse

        return HTMLResponse(content=f.read())


@app.get("/api/samples")
async def list_samples():
    """List available sample transcripts."""
    data_path = Path(__file__).parent / "data"
    if not data_path.exists():
        return {"samples": []}

    samples = []
    for file in data_path.glob("*.json"):
        samples.append(
            {
                "id": file.stem,
                "name": file.stem.replace("_", " ").title(),
                "filename": file.name,
            }
        )

    return {"samples": samples}


@app.get("/api/samples/{sample_id}")
async def get_sample(sample_id: str):
    """Get a specific sample transcript."""
    data_path = Path(__file__).parent / "data" / f"{sample_id}.json"

    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")

    with open(data_path, "r") as f:
        return json.load(f)


async def generate_sse_events(
    transcript: str, metadata: dict | None = None
) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events for real-time progress updates.

    Args:
        transcript: The podcast transcript
        metadata: Optional metadata

    Yields:
        SSE formatted events
    """
    config = get_config()
    stream_delay = config.app_settings.get("stream_delay_ms", 100) / 1000.0

    try:
        # Send initial event
        yield f"data: {json.dumps({'stage': 'started', 'message': 'Analysis started'})}\n\n"
        await asyncio.sleep(stream_delay)

        # Stream the workflow
        async for event in stream_analysis(transcript, metadata):
            # event is a dict with node name as key
            for node_name, node_state in event.items():
                stage = node_state.get("current_stage", "processing")
                messages = node_state.get("messages", [])

                # Send stage update
                stage_data = {
                    "stage": stage,
                    "node": node_name,
                    "message": messages[-1] if messages else f"Processing {node_name}",
                }

                yield f"data: {json.dumps(stage_data)}\n\n"
                await asyncio.sleep(stream_delay)

        # Get final state (last event)
        final_state = list(event.values())[0] if event else {}

        # Construct final output
        final_output = FinalOutput(
            summary=final_state.get("supervisor_output"),
            fact_check=final_state.get("fact_check_output"),
            confidence_in_analysis=final_state.get("fact_check_output").overall_reliability
            if final_state.get("fact_check_output")
            else 0.0,
            critic_iterations=final_state.get("critic_iterations", 0),
            processing_notes="; ".join(final_state.get("messages", [])),
        )

        # Send final result
        yield f"data: {json.dumps({'stage': 'complete', 'result': final_output.model_dump()})}\n\n"

    except Exception as e:
        import traceback

        logger.error(f"Error in analysis workflow: {e}")
        logger.error(traceback.format_exc())
        error_data = {"stage": "error", "message": str(e), "traceback": traceback.format_exc()}
        yield f"data: {json.dumps(error_data)}\n\n"


@app.post("/api/analyze")
async def analyze_transcript(input_data: TranscriptInput):
    """Analyze a podcast transcript with SSE streaming.

    Args:
        input_data: Transcript input with text and optional metadata

    Returns:
        StreamingResponse with Server-Sent Events
    """
    logger.info(f"Starting analysis. Transcript length: {len(input_data.transcript)}")
    logger.info(f"Metadata: {input_data.metadata}")

    return StreamingResponse(
        generate_sse_events(input_data.transcript, input_data.metadata),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


@app.post("/api/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """Analyze a podcast transcript from uploaded file.

    Args:
        file: Uploaded file (.txt or .json)

    Returns:
        StreamingResponse with Server-Sent Events
    """
    # Read file content
    content = await file.read()

    try:
        # Try to parse as JSON first
        if file.filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))
            transcript = data.get("transcript", "")
            metadata = data.get("metadata", {})
        else:
            # Treat as plain text
            transcript = content.decode("utf-8")
            metadata = {"filename": file.filename}

        if not transcript or len(transcript) < 100:
            raise HTTPException(
                status_code=400, detail="Transcript too short (minimum 100 characters)"
            )

        return StreamingResponse(
            generate_sse_events(transcript, metadata),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File encoding not supported")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    config = get_config()
    return {
        "status": "healthy",
        "models_configured": len(config.models),
        "search_tools_available": sum(
            1
            for key in [
                config.settings.tavily_api_key,
                config.settings.serper_api_key,
                config.settings.brave_api_key,
            ]
            if key is not None
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

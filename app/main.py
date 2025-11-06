"""FastAPI application for podcast analysis."""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# CRITICAL: Load config and set up LangSmith BEFORE any imports that use langsmith
from app.config import get_config

# Initialize LangSmith tracing by setting environment variables BEFORE langsmith imports
config = get_config()
config.setup_langsmith()

# Debug: Confirm LangSmith env vars are set before langsmith imports
print(f"[DEBUG] LangSmith setup called early - LANGCHAIN_TRACING_V2={os.getenv('LANGCHAIN_TRACING_V2')}, project={os.getenv('LANGCHAIN_PROJECT')}")
sys.stdout.flush()

# Now set up JSON logging for CloudWatch
from app.utils.logger import (
    setup_json_logging,
    get_logger,
    generate_session_id,
    set_session_context,
    clear_session_context,
    TimingContext,
)

# Initialize JSON logging
setup_json_logging(level="INFO")
logger = get_logger(__name__)

# Log that JSON logging is active
logger.info("JSON logging initialized successfully")

# NOW import modules that use langsmith (AFTER environment variables are set)
from app.agents.graph import stream_analysis
from app.models.transcript import TranscriptInput


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown)."""
    # Re-apply logging configuration in case the host server overrides it (e.g., uvicorn reload)
    setup_json_logging(level="INFO")

    # Startup: Initialize application
    logger.info("Application startup initiated")

    # Note: config is already loaded at module level, and setup_langsmith() was already called
    # This ensures LangSmith environment variables are set before langsmith module imports

    # Log configuration details
    logger.info(
        "Configuration loaded",
        extra={
            "models_configured": len(config.models),
            "model_names": list(config.models.keys()),
            "search_tools_available": {
                "tavily": config.settings.tavily_api_key is not None,
                "brave": config.settings.brave_api_key is not None,
            },
        },
    )

    # Log LangSmith tracing status (already configured at module load time)
    if config.settings.langsmith_api_key:
        logger.info("LangSmith tracing enabled", extra={"project": config.settings.langsmith_project})
    else:
        logger.info("LangSmith tracing disabled (no API key configured)")

    logger.info("Application startup complete - ready to accept requests")

    yield  # Application runs here

    # Shutdown: Clean up resources (if needed)
    logger.info("Application shutdown")


app = FastAPI(
    title="Podcast Agent",
    description="AI-powered podcast analysis with multi-agent workflow",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed logging."""
    request_body = await request.body()
    logger.error(
        "Request validation error",
        extra={
            "error_type": "RequestValidationError",
            "url": str(request.url),
            "errors": exc.errors(),
            "request_body": request_body.decode("utf-8") if request_body else None,
        },
    )
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
    transcript: str, metadata: dict | None = None, session_id: str | None = None
) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events for real-time progress updates (NEW SUPERVISOR WORKFLOW).

    Args:
        transcript: The podcast transcript
        metadata: Optional metadata
        session_id: Session ID for tracking this analysis session

    Yields:
        SSE formatted events with detailed agent progress, reasoning, and tool calls
    """
    import sys
    print(f"[DEBUG] generate_sse_events: ENTERED - session_id={session_id}, transcript_length={len(transcript)}")
    sys.stdout.flush()

    config = get_config()
    stream_delay = config.app_settings.get("stream_delay_ms", 100) / 1000.0

    # Set session context for all logs in this coroutine
    if session_id:
        set_session_context(session_id)

    try:
        # Log workflow start with session info
        print("[DEBUG] generate_sse_events: About to log 'Analysis workflow started'")
        sys.stdout.flush()

        logger.info(
            "Analysis workflow started",
            extra={
                "stage": "started",
                "transcript_length": len(transcript),
                "metadata": metadata,
            },
        )

        # Send initial event
        print("[DEBUG] generate_sse_events: About to send 'started' event")
        sys.stdout.flush()
        logger.info("SSE: Sending 'started' event", extra={"stage": "started"})
        yield f"data: {json.dumps({'stage': 'started', 'message': 'Starting podcast analysis workflow...'})}\n\n"
        await asyncio.sleep(stream_delay)

        # Track accumulated state throughout streaming
        accumulated_state = {}
        workflow_start_time = asyncio.get_event_loop().time()

        # Map tool names to UI-friendly agent names
        tool_to_agent_map = {
            "summarize_podcast_tool": "Summarizing Agent",
            "extract_notes_tool": "Note Extraction Agent",
            "fact_check_claims_tool": "Fact Checking Agent",
        }

        # Stream the workflow (receives fine-grained events from astream_events)
        print("[DEBUG] generate_sse_events: About to call stream_analysis()")
        sys.stdout.flush()

        async for event in stream_analysis(transcript, metadata, session_id):
            event_type = event.get("event")
            event_name = event.get("name", "")
            print(f"[DEBUG] generate_sse_events: Received event - type={event_type}, name={event_name}")
            sys.stdout.flush()

            # Handle tool start events (agent beginning work)
            if event_type == "on_tool_start":
                tool_name = event_name
                agent_name = tool_to_agent_map.get(tool_name, tool_name)

                # Send supervisor_decision event
                sse_event = {
                    "stage": "supervisor_complete",
                    "node": "supervisor",
                    "type": "supervisor_decision",
                    "agent": "Supervisor",
                    "target_agent": agent_name,
                    "tool": tool_name,
                    "message": f"Calling {agent_name}...",
                    "action": "calling_agent"
                }

                try:
                    logger.info(f"SSE: Sending 'supervisor_decision' event → Calling {agent_name}")
                    yield f"data: {json.dumps(sse_event)}\n\n"
                    await asyncio.sleep(stream_delay)
                except TypeError as e:
                    logger.error(f"JSON serialization failed for supervisor_decision event: {e}")
                    logger.error(f"Event data: {sse_event}")

            # Handle tool end events (agent completed work)
            elif event_type == "on_tool_end":
                tool_name = event_name
                agent_name = tool_to_agent_map.get(tool_name, tool_name)

                # Extract result summary from event data
                event_data = event.get("data", {})
                output = event_data.get("output", "")

                # Try to parse details from output
                details = "Completed"
                if isinstance(output, str) and len(output) > 0:
                    try:
                        output_json = json.loads(output)
                        # Extract key metrics based on agent type
                        if "summarize" in tool_name and "core_theme" in output_json:
                            details = f"Core theme: {output_json['core_theme'][:80]}..."
                        elif "extract_notes" in tool_name:
                            num_claims = len(output_json.get("factual_statements", []))
                            num_quotes = len(output_json.get("notable_quotes", []))
                            num_topics = len(output_json.get("topics", []))
                            details = f"Extracted {num_claims} factual claims, {num_quotes} quotes, {num_topics} topics"
                        elif "fact_check" in tool_name:
                            claims = output_json.get("verified_claims", [])
                            status_counts = {}
                            for claim in claims:
                                status = claim.get("verification_status", "unknown")
                                status_counts[status] = status_counts.get(status, 0) + 1
                            details = f"Verified {len(claims)} claims: {status_counts}"
                    except:
                        pass

                # Send agent_complete event
                sse_event = {
                    "stage": "supervisor_complete",
                    "node": "supervisor",
                    "type": "agent_complete",
                    "agent": agent_name,
                    "details": details,
                    "message": f"{agent_name} completed: {details}",
                    "action": "agent_complete"
                }

                try:
                    logger.info(f"SSE: Sending 'agent_complete' event → {agent_name} completed: {details[:60]}...")
                    yield f"data: {json.dumps(sse_event)}\n\n"
                    await asyncio.sleep(stream_delay)
                except TypeError as e:
                    logger.error(f"JSON serialization failed for agent_complete event: {e}")
                    logger.error(f"Event data: {sse_event}")

            # Handle workflow completion (final state available)
            elif event_type == "on_chain_end" and event_name == "LangGraph":
                # Extract final state from event
                event_data = event.get("data", {})
                final_output = event_data.get("output", {})

                # Store in accumulated_state for results processing
                accumulated_state = final_output

                # Send supervisor reasoning event
                sse_event = {
                    "stage": "supervisor_complete",
                    "node": "supervisor",
                    "type": "supervisor_reasoning",
                    "agent": "Supervisor",
                    "message": "All specialist agents have completed their work.",
                    "action": "supervisor_thinking"
                }

                try:
                    logger.info("SSE: Sending 'supervisor_reasoning' event → All agents completed")
                    yield f"data: {json.dumps(sse_event)}\n\n"
                    await asyncio.sleep(stream_delay)
                except TypeError as e:
                    logger.error(f"JSON serialization failed for supervisor_reasoning event: {e}")
                    logger.error(f"Event data: {sse_event}")

        # Use accumulated state instead of last event
        final_state = accumulated_state
        workflow_duration_ms = int((asyncio.get_event_loop().time() - workflow_start_time) * 1000)

        # Get supervisor output from accumulated final state (NEW WORKFLOW)
        supervisor_output = final_state.get("supervisor_output", {})

        # Extract specialist agent outputs
        summary_data = supervisor_output.get("summary", {})
        notes_data = supervisor_output.get("notes", {})
        fact_check_data = supervisor_output.get("fact_check", {})

        # Build final result for the UI
        final_result = {
            "summary": summary_data,
            "notes": notes_data,
            "fact_check": fact_check_data,
            "metadata": {
                "total_tool_calls": supervisor_output.get("total_tool_calls", 0),
                "agents_invoked": supervisor_output.get("agents_invoked", 0),
                "processing_messages": final_state.get("messages", []),
            }
        }

        # Log workflow completion with comprehensive metrics
        logger.info(
            "Workflow completed successfully",
            extra={
                "stage": "complete",
                "duration_ms": workflow_duration_ms,
                "agents_invoked": supervisor_output.get("agents_invoked", 0),
                "total_tool_calls": supervisor_output.get("total_tool_calls", 0),
                "num_claims": len(fact_check_data.get("verified_claims", [])) if fact_check_data else 0,
                "num_quotes": len(notes_data.get("notable_quotes", [])) if notes_data else 0,
                "num_topics": len(notes_data.get("topics", [])) if notes_data else 0,
            }
        )

        # TODO: Generate output files (JSON and Markdown) - currently disabled for new workflow
        # Will be re-implemented after workflow is stable
        json_filename = None
        md_filename = None

        # Send final result with error handling
        try:
            result_data = {
                'stage': 'complete',
                'result': final_result,
                'output_files': {
                    'json': json_filename,
                    'markdown': md_filename,
                }
            }
            logger.info("SSE: Sending 'complete' event with final results")
            yield f"data: {json.dumps(result_data)}\n\n"
        except Exception as e:
            logger.error(f"Failed to serialize final output: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Send error event
            logger.info("SSE: Sending 'error' event due to serialization failure")
            yield f"data: {json.dumps({'stage': 'error', 'message': 'Failed to generate final output. Please try again.'})}\n\n"

    except Exception as e:
        import traceback

        error_msg = f"CRITICAL ERROR in generate_sse_events: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        sys.stdout.flush()

        logger.error(
            "Error in analysis workflow",
            extra={
                "stage": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        error_data = {"stage": "error", "message": str(e), "traceback": traceback.format_exc()}
        yield f"data: {json.dumps(error_data)}\n\n"
    finally:
        # Clear session context when done
        if session_id:
            clear_session_context()


@app.post("/api/analyze")
async def analyze_transcript(input_data: TranscriptInput):
    """Analyze a podcast transcript with SSE streaming.

    Args:
        input_data: Transcript input with text and optional metadata

    Returns:
        StreamingResponse with Server-Sent Events
    """
    # Generate unique session ID for this analysis
    session_id = generate_session_id()
    set_session_context(session_id)

    logger.info(
        "Analysis request received",
        extra={
            "session_id": session_id,
            "transcript_length": len(input_data.transcript),
            "metadata": input_data.metadata,
        },
    )

    return StreamingResponse(
        generate_sse_events(input_data.transcript, input_data.metadata, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
            "X-Session-ID": session_id,  # Include session ID in response headers
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
    # Generate unique session ID for this analysis
    session_id = generate_session_id()
    set_session_context(session_id)

    # Read file content
    content = await file.read()

    try:
        # Try to parse as JSON first
        if file.filename.endswith(".json"):
            data = json.loads(content.decode("utf-8"))

            # Handle transcript array format (like sample files)
            transcript_data = data.get("transcript", "")
            if isinstance(transcript_data, list):
                # Concatenate all text fields from transcript entries
                transcript = " ".join(
                    entry.get("text", "") for entry in transcript_data if isinstance(entry, dict)
                )

                # Extract metadata from JSON file
                metadata = {
                    "episode_id": data.get("episode_id", ""),
                    "title": data.get("title", ""),
                    "host": data.get("host", ""),
                    "guests": data.get("guests", []),
                    "filename": file.filename,
                }
            else:
                # Simple string format
                transcript = transcript_data
                metadata = data.get("metadata", {})
                if "filename" not in metadata:
                    metadata["filename"] = file.filename
        else:
            # Treat as plain text
            transcript = content.decode("utf-8")
            metadata = {"filename": file.filename}

        if not transcript or len(transcript) < 100:
            logger.warning(
                "Transcript too short",
                extra={
                    "session_id": session_id,
                    "filename": file.filename,
                    "transcript_length": len(transcript),
                },
            )
            raise HTTPException(
                status_code=400, detail="Transcript too short (minimum 100 characters)"
            )

        logger.info(
            "File upload analysis request received",
            extra={
                "session_id": session_id,
                "filename": file.filename,
                "transcript_length": len(transcript),
                "metadata": metadata,
            },
        )

        return StreamingResponse(
            generate_sse_events(transcript, metadata, session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Session-ID": session_id,  # Include session ID in response headers
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
                config.settings.brave_api_key,
            ]
            if key is not None
        ),
    }


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download a generated output file (JSON or Markdown).

    Args:
        filename: The filename to download (e.g., "analysis_20240101_120000.json")

    Returns:
        File download response
    """
    from fastapi.responses import FileResponse

    # Security: Only allow downloading from output directory and validate filename
    output_dir = Path(__file__).parent.parent / "output"
    file_path = output_dir / filename

    # Validate the file path to prevent directory traversal
    try:
        file_path = file_path.resolve()
        output_dir = output_dir.resolve()

        if not str(file_path).startswith(str(output_dir)):
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Determine media type
        if filename.endswith('.json'):
            media_type = 'application/json'
        elif filename.endswith('.md'):
            media_type = 'text/markdown'
        else:
            media_type = 'application/octet-stream'

        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename,
        )
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Error downloading file")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

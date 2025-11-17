#!/usr/bin/env python3
"""
Lightweight API server for running prediction samplers.

This server exposes a single endpoint that accepts question metadata
and returns a SingleEvalResult with the full conversation and trace data.
"""

import argparse
from typing import Any, Literal
from pathlib import Path
import importlib.util
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from gpt_oss.evals.flat_sampler import FlatSampler, AgentEvalContext
from gpt_oss.evals.recursive_sampler import RecursiveSampler
from gpt_oss.evals.types import SingleEvalResult
from gpt_oss.evals.utils import load_prompt

# Import centralized MCP server config without triggering package __init__.py
config_path = Path(__file__).parent.parent / "gpt_oss" / "evals" / "mcp_servers_config.py"
spec = importlib.util.spec_from_file_location("mcp_servers_config", config_path)
mcp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_config)

# Build MCP server mapping from config
MCP_SERVER_MAP = {
    server.short_name: (server.short_name, server.port)
    for server in mcp_config.MCP_SERVERS
}


# Pydantic models for request/response
class SamplerConfig(BaseModel):
    """Configuration for the sampler."""
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    max_tokens: int = 131_072
    reasoning_model: bool = False
    reasoning_effort: str | None = "medium"
    verbosity: str = "medium"
    base_url: str | None = None
    mcp_servers: list[str] | None = None
    enable_internal_browser: bool = False
    enable_internal_python: bool = False
    max_turns: int = 10
    max_depth: int | None = None  # For RecursiveSampler only


class PredictRequest(BaseModel):
    """Request schema matching metaculus_eval.py datapoint structure."""
    # Required fields
    question: str
    description: str
    resolution_criteria: str
    question_type: Literal["binary", "multiple_choice", "numeric"]
    cutoff_date: str

    # Optional fields for multiple_choice
    outcomes: list[str] | None = None

    # Optional fields for numeric
    range_min: float | None = None
    range_max: float | None = None
    zero_point: float | None = None
    open_lower_bound: bool = False
    open_upper_bound: bool = False
    unit: str = ""

    # Sampler configuration
    sampler_type: Literal["flat", "recursive"] = "flat"
    sampler_config: SamplerConfig = Field(default_factory=SamplerConfig)


class PredictResponse(BaseModel):
    """Response schema wrapping SingleEvalResult."""
    score: float | None
    metrics: dict[str, float]
    html: str | None
    convo: list[dict[str, Any]]
    spans: list[dict[str, Any]] | None
    example_level_metadata: dict[str, Any] | None


# Create FastAPI app
app = FastAPI(
    title="Sampler Prediction API",
    description="API for running prediction samplers on forecasting questions",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_user_message(request: PredictRequest) -> str:
    """Format the user message like metaculus_eval.py does."""
    user_message = load_prompt("user.md").format(
        question=request.question,
        description=request.description,
        resolution_criteria=request.resolution_criteria,
        question_type=request.question_type,
    )

    if request.question_type == "multiple_choice":
        user_message += f"\n\nPossible Options:\n{request.outcomes}"

    if request.question_type == "numeric":
        if not request.open_lower_bound:
            user_message += f"\n\nLower Bound:\n{request.range_min}"
        if not request.open_upper_bound:
            user_message += f"\n\nUpper Bound:\n{request.range_max}"

    return user_message


@app.post("/v1/eval/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Run a prediction sampler on a forecasting question.

    This endpoint accepts question metadata (like a metaculus_eval.py datapoint),
    runs the specified sampler, and returns the raw conversation and trace data.
    """
    sampler = None
    try:
        # Load the appropriate system prompt
        if request.sampler_type == "flat":
            developer_message = load_prompt("superforecaster.md")
        else:  # recursive
            developer_message = load_prompt("unified_forecaster.md")

        # Initialize the sampler
        sampler_kwargs = request.sampler_config.model_dump()
        sampler_kwargs["developer_message"] = developer_message

        if sampler_kwargs["mcp_servers"]:
            mcp_servers = [MCP_SERVER_MAP[mcp_server] for mcp_server in sampler_kwargs["mcp_servers"]]
            sampler_kwargs["mcp_servers"] = mcp_servers

        # Remove max_depth if not using RecursiveSampler
        max_depth = sampler_kwargs.pop("max_depth", None)

        if request.sampler_type == "flat":
            sampler = FlatSampler(**sampler_kwargs)
        else:  # recursive
            if max_depth is not None:
                sampler_kwargs["max_depth"] = max_depth
            sampler = RecursiveSampler(**sampler_kwargs)

        # Format the user message
        user_message = format_user_message(request)
        message_list = [{"role": "user", "content": user_message}]

        # Run the sampler (call async method directly to avoid event loop conflict)
        if request.sampler_type == "flat":
            context = AgentEvalContext(cutoff_date=request.cutoff_date)
            sampler_response = await sampler._execute_async(message_list, context)
        else:  # recursive
            sampler_response = await sampler._execute_async(message_list, request.cutoff_date)

        # Build the conversation (includes final assistant response)
        convo = sampler_response.actual_queried_message_list + [
            {"role": "assistant", "content": sampler_response.response_text}
        ]

        # Create SingleEvalResult (no scoring/extraction, just raw data)
        result = SingleEvalResult(
            score=None,
            metrics={},
            html=None,
            convo=convo,
            spans=sampler_response.spans,
            example_level_metadata=None,
        )

        # Convert to response format
        return PredictResponse(
            score=result.score,
            metrics=result.metrics,
            html=result.html,
            convo=result.convo,
            spans=result.spans,
            example_level_metadata=result.example_level_metadata,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Prompt file error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        # Clean up MCP connections to prevent context manager issues across tasks
        if sampler is not None and hasattr(sampler, '_thread_local'):
            if hasattr(sampler._thread_local, 'mcp_servers'):
                mcp_servers = sampler._thread_local.mcp_servers
                if mcp_servers:
                    for server in mcp_servers:
                        try:
                            await server.cleanup()
                        except Exception:
                            print(f"Error cleaning up MCP server {server.name}: {e}. Non-critical error.")
                            pass  # Ignore cleanup errors
                    sampler._thread_local.mcp_servers = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    """CLI entry point for serving the sampler API."""
    parser = argparse.ArgumentParser(description="Serve the prediction sampler API")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    print(f"Starting sampler API server on {args.host}:{args.port}")
    print(f"Endpoint: POST http://{args.host}:{args.port}/v1/eval/predict")
    print(f"Health check: GET http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

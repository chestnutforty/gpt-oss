"""A local JSON storage integration for OpenAI Agents.

This module provides a TracingProcessor implementation that logs OpenAI
Agent traces and spans to local JSON files.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from agents.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    MCPListToolsSpanData,
    ResponseSpanData,
    SpeechGroupSpanData,
    SpeechSpanData,
    Span,
    Trace,
    TracingProcessor,
    TranscriptionSpanData,
)


def _call_type(span: Span) -> str:
    """Determine the appropriate call type for a given OpenAI Agent span."""
    return span.span_data.type or "task"


def _call_name(span: Span) -> str:
    """Determine the name for a given OpenAI Agent span."""
    if name := getattr(span.span_data, "name", None):
        return name
    elif isinstance(span.span_data, ResponseSpanData):
        return "Response"
    elif isinstance(span.span_data, HandoffSpanData):
        return "Handoff"
    elif isinstance(span.span_data, MCPListToolsSpanData):
        return f"MCP Tools ({span.span_data.server or 'unknown server'})"
    elif isinstance(span.span_data, GenerationSpanData):
        return "Generation"
    elif isinstance(span.span_data, TranscriptionSpanData):
        return "Transcription"
    elif isinstance(span.span_data, SpeechSpanData):
        return "Speech"
    elif isinstance(span.span_data, SpeechGroupSpanData):
        return "Speech Group"
    else:
        return "Unknown"


class TraceDataDict(TypedDict):
    inputs: list[Any]
    outputs: list[Any]
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    error: dict[str, Any] | None


class LocalJSONTracingProcessor(TracingProcessor):
    """A TracingProcessor implementation that logs OpenAI Agent traces and spans to local JSON files.

    This processor captures different types of spans from OpenAI Agents (agent execution,
    function calls, LLM generations, etc.) and saves them as structured JSON data locally.
    Child spans are logged as separate entries but not redundantly included in the parent trace data.
    """

    def __init__(self, output_dir: str = "agent_traces") -> None:
        """Initialize the local JSON tracing processor.

        Args:
            output_dir: Directory to store trace JSON files. Defaults to "agent_traces".
        """
        self.output_dir = Path(output_dir)
        # self.output_dir.mkdir(parents=True, exist_ok=True)

        self._trace_data: dict[str, dict[str, Any]] = {}
        self._span_data: dict[str, dict[str, Any]] = {}
        self._ended_traces: set[str] = set()
        self._span_parents: dict[str, str] = {}
        self._trace_start_times: dict[str, float] = {}
        self._span_start_times: dict[str, float] = {}

    def on_trace_start(self, trace: Trace) -> None:
        """Called when a trace starts."""
        start_time = time.time()
        self._trace_start_times[trace.trace_id] = start_time

        # Set up basic trace data
        self._trace_data[trace.trace_id] = {
            "trace_id": trace.trace_id,
            "name": trace.name,
            "type": "task",
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": None,
            "status": "running",
            "metrics": {},
            "metadata": {},
            "spans": [],
        }

    def on_trace_end(self, trace: Trace) -> None:
        """Called when a trace ends."""
        tid = trace.trace_id
        if tid not in self._trace_data:
            return

        end_time = time.time()
        trace_data = self._trace_data[tid]
        self._ended_traces.add(tid)

        # Update trace data with completion info
        trace_data["end_time"] = datetime.fromtimestamp(end_time).isoformat()
        trace_data["status"] = "completed"
        if tid in self._trace_start_times:
            trace_data["duration_seconds"] = end_time - self._trace_start_times[tid]
            
        trace.trace_data = trace_data

        # Write trace to JSON file
        # self._write_trace_to_file(tid, trace_data)

    def _agent_log_data(self, span: Span[AgentSpanData]) -> TraceDataDict:
        """Extract log data from an agent span."""
        return TraceDataDict(
            inputs=[],
            outputs=[],
            metadata={
                "tools": span.span_data.tools,
                "handoffs": span.span_data.handoffs,
                "output_type": span.span_data.output_type,
            },
            metrics={},
            error=None,
        )

    def _response_log_data(self, span: Span[ResponseSpanData]) -> TraceDataDict:
        """Extract log data from a response span."""
        inputs = []
        outputs = []
        metadata: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        # Add input if available
        if span.span_data.input is not None:
            inputs.append(span.span_data.input)

        # Extract output and other details from response
        if span.span_data.response is not None:
            # Just get the plain output value
            outputs.append(span.span_data.response.output)

            # All other data goes into metadata
            metadata = span.span_data.response.metadata or {}

            # Add all other response fields to metadata
            additional_fields = span.span_data.response.model_dump(
                exclude={"input", "output", "metadata", "usage"}
            )
            metadata.update(additional_fields)

            # Add usage data to metrics if available
            if span.span_data.response.usage is not None:
                usage = span.span_data.response.usage
                metrics = {
                    "tokens": usage.total_tokens,
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                }

        return TraceDataDict(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            metrics=metrics,
            error=None,
        )

    def _function_log_data(self, span: Span[FunctionSpanData]) -> TraceDataDict:
        """Extract log data from a function span."""
        return TraceDataDict(
            inputs=span.span_data.input,
            outputs=span.span_data.output,
            metadata={},
            metrics={},
            error=None,
        )

    def _handoff_log_data(self, span: Span[HandoffSpanData]) -> TraceDataDict:
        """Extract log data from a handoff span."""
        return TraceDataDict(
            inputs=[],
            outputs=[],
            metadata={
                "from_agent": span.span_data.from_agent,
                "to_agent": span.span_data.to_agent,
            },
            metrics={},
            error=None,
        )

    def _guardrail_log_data(self, span: Span[GuardrailSpanData]) -> TraceDataDict:
        """Extract log data from a guardrail span."""
        return TraceDataDict(
            inputs=[],
            outputs=[],
            metadata={"triggered": span.span_data.triggered},
            metrics={},
            error=None,
        )

    def _custom_log_data(self, span: Span[CustomSpanData]) -> TraceDataDict:
        """Extract log data from a custom span."""
        # Prepare fields
        inputs = []
        outputs = []
        metadata: dict[str, Any] = {}
        metrics: dict[str, Any] = {}

        # Extract data from the custom span
        custom_data = span.span_data.data

        # Map custom data to the appropriate fields if possible
        if "input" in custom_data:
            inputs.append(custom_data["input"])

        if "output" in custom_data:
            outputs.append(custom_data["output"])

        if "metadata" in custom_data:
            metadata = custom_data["metadata"]

        if "metrics" in custom_data:
            metrics = custom_data["metrics"]

        # Add any remaining fields to metadata
        for key, value in custom_data.items():
            if key not in ["input", "output", "metadata", "metrics"]:
                metadata[key] = value

        return TraceDataDict(
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
            metrics=metrics,
            error=None,
        )

    def _mcp_tools_log_data(self, span: Span[MCPListToolsSpanData]) -> TraceDataDict:
        """Extract log data from an MCP tools span."""
        return TraceDataDict(
            inputs=[],
            outputs=span.span_data.result or [],
            metadata={
                "server": span.span_data.server,
            },
            metrics={},
            error=None,
        )

    def _generation_log_data(self, span: Span[GenerationSpanData]) -> TraceDataDict:
        """Extract log data from a generation span."""
        return TraceDataDict(
            inputs=list(span.span_data.input) if span.span_data.input else [],
            outputs=list(span.span_data.output) if span.span_data.output else [],
            metadata={
                "model": span.span_data.model,
                "model_config": span.span_data.model_config,
            },
            metrics=span.span_data.usage or {},
            error=None,
        )

    def _transcription_log_data(self, span: Span[TranscriptionSpanData]) -> TraceDataDict:
        """Extract log data from a transcription span."""
        return TraceDataDict(
            inputs=[{"data": span.span_data.input, "format": span.span_data.input_format}] if span.span_data.input else [],
            outputs=[span.span_data.output] if span.span_data.output else [],
            metadata={
                "model": span.span_data.model,
                "model_config": span.span_data.model_config,
            },
            metrics={},
            error=None,
        )

    def _speech_log_data(self, span: Span[SpeechSpanData]) -> TraceDataDict:
        """Extract log data from a speech span."""
        return TraceDataDict(
            inputs=[span.span_data.input] if span.span_data.input else [],
            outputs=[{"data": span.span_data.output, "format": span.span_data.output_format}] if span.span_data.output else [],
            metadata={
                "model": span.span_data.model,
                "model_config": span.span_data.model_config,
                "first_content_at": span.span_data.first_content_at,
            },
            metrics={},
            error=None,
        )

    def _speech_group_log_data(self, span: Span[SpeechGroupSpanData]) -> TraceDataDict:
        """Extract log data from a speech group span."""
        return TraceDataDict(
            inputs=[span.span_data.input] if span.span_data.input else [],
            outputs=[],
            metadata={},
            metrics={},
            error=None,
        )

    def _log_data(self, span: Span) -> TraceDataDict:
        """Extract the appropriate log data based on the span type."""
        if isinstance(span.span_data, AgentSpanData):
            return self._agent_log_data(span)
        elif isinstance(span.span_data, ResponseSpanData):
            return self._response_log_data(span)
        elif isinstance(span.span_data, FunctionSpanData):
            return self._function_log_data(span)
        elif isinstance(span.span_data, HandoffSpanData):
            return self._handoff_log_data(span)
        elif isinstance(span.span_data, GuardrailSpanData):
            return self._guardrail_log_data(span)
        elif isinstance(span.span_data, CustomSpanData):
            return self._custom_log_data(span)
        elif isinstance(span.span_data, MCPListToolsSpanData):
            return self._mcp_tools_log_data(span)
        elif isinstance(span.span_data, GenerationSpanData):
            return self._generation_log_data(span)
        elif isinstance(span.span_data, TranscriptionSpanData):
            return self._transcription_log_data(span)
        elif isinstance(span.span_data, SpeechSpanData):
            return self._speech_log_data(span)
        elif isinstance(span.span_data, SpeechGroupSpanData):
            return self._speech_group_log_data(span)
        else:
            return TraceDataDict(
                inputs=[],
                outputs=[],
                metadata={},
                metrics={},
                error=None,
            )

    def on_span_start(self, span: Span) -> None:
        """Called when a span starts."""
        tid = span.trace_id

        # Spans must be part of a trace
        if tid not in self._trace_data:
            return

        start_time = time.time()
        self._span_start_times[span.span_id] = start_time

        span_name = _call_name(span)
        span_type = _call_type(span)
        parent_span_id = getattr(span, "parent_id", None)

        # Create span data structure
        span_data = {
            "span_id": span.span_id,
            "name": span_name,
            "type": span_type,
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "end_time": None,
            "parent_span_id": parent_span_id,
            "inputs": [],
            "outputs": [],
            "metadata": {},
            "metrics": {},
            "error": None,
        }

        self._span_data[span.span_id] = span_data

        # Store parent relationship for hierarchy
        if parent_span_id is not None:
            self._span_parents[span.span_id] = parent_span_id

    def on_span_end(self, span: Span) -> None:
        """Called when a span ends."""
        trace_id = span.trace_id
        if trace_id not in self._trace_data:
            return

        # Get or create span data
        if span.span_id not in self._span_data:
            # Handle case where span_start wasn't called (e.g., Response spans)
            self.on_span_start(span)

        if span.span_id not in self._span_data:
            return

        end_time = time.time()
        log_data = self._log_data(span)
        span_data = self._span_data[span.span_id]

        # Update span with completion info
        span_data["end_time"] = datetime.fromtimestamp(end_time).isoformat()
        if span.span_id in self._span_start_times:
            span_data["duration_seconds"] = (
                end_time - self._span_start_times[span.span_id]
            )

        # Update with log data
        span_data["inputs"] = self._make_json_serializable(log_data["inputs"])
        span_data["outputs"] = self._make_json_serializable(log_data["outputs"])
        span_data["metadata"] = log_data["metadata"]
        span_data["metrics"] = log_data["metrics"]

        # Add error if present
        if span.error:
            span_data["error"] = str(span.error)
        elif log_data["error"]:
            span_data["error"] = log_data["error"]

        # Add span to trace's span list
        self._trace_data[trace_id]["spans"].append(span_data)

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert Pydantic BaseModels and other objects to JSON-serializable format."""
        # Handle Pydantic BaseModels
        if hasattr(obj, "model_dump"):
            return self._make_json_serializable(obj.model_dump())

        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}

        # Handle lists recursively
        if isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]

        # Return basic types as-is
        return obj

    def _write_trace_to_file(self, trace_id: str, trace_data: dict[str, Any]) -> None:
        """Write trace data to a JSON file."""
        # Create a safe filename from trace_id and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Take first 8 chars of trace_id for readability
        trace_id_short = trace_id[:8] if len(trace_id) > 8 else trace_id
        filename = f"trace_{timestamp}_{trace_id_short}.json"
        filepath = self.output_dir / filename

        # Make data JSON serializable (handles Pydantic models)
        serializable_data = self._make_json_serializable(trace_data)

        # Write the trace data to file
        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=2)

    def _finish_unfinished_calls(self, status: str) -> None:
        """Helper method for finishing unfinished traces on shutdown or flush."""
        end_time = time.time()

        # Finish any unfinished traces
        for trace_id, trace_data in self._trace_data.items():
            # Check if trace is already finished
            if trace_id not in self._ended_traces:
                # Set status based on shutdown/flush
                actual_status = "completed" if trace_id in self._ended_traces else status

                # Update trace data
                trace_data["end_time"] = datetime.fromtimestamp(end_time).isoformat()
                trace_data["status"] = actual_status
                if trace_id in self._trace_start_times:
                    trace_data["duration_seconds"] = (
                        end_time - self._trace_start_times[trace_id]
                    )

                # Write to file
                self._write_trace_to_file(trace_id, trace_data)

    def shutdown(self) -> None:
        """Called when the application stops."""
        self._finish_unfinished_calls("interrupted")

    def force_flush(self) -> None:
        """Forces an immediate flush of all queued traces."""
        self._finish_unfinished_calls("force_flushed")
        
    def export(self) -> list[dict[str, Any]]:
        """Exports all traces as a list of dictionaries."""
        return list(self._trace_data.values())


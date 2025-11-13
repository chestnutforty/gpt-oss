import asyncio
import json
import os
import time
import threading
from typing import Any

import openai
from openai import OpenAI

from .mcp_client_manager import MCPClientManager
from .types import MessageList, SamplerBase, SamplerResponse


class ResponsesSampler(SamplerBase):
    """
    Sample from OpenAI's responses API
    """

    def __init__(
        self,
        model: str,
        developer_message: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 131_072,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        mcp_servers: list[tuple[str, int]] | None = None,
        enable_internal_browser: bool = False,
        enable_internal_python: bool = False,
    ):
        self.client = OpenAI(base_url=base_url, timeout=24*60*60, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))
        self.model = model
        self.developer_message = developer_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.mcp_servers = mcp_servers
        # Use thread-local storage for MCP manager since it contains async objects
        # tied to a specific event loop. Each thread needs its own instance.
        self._thread_local = threading.local() if mcp_servers else None

        # Build internal tools list
        self.internal_tools = []
        if enable_internal_python:
            self.internal_tools.append({
                "type": "code_interpreter",
                "container": {
                    "type": "auto"
                }
            })
        if enable_internal_browser:
            self.internal_tools.append({
                "type": "web_search"
            })

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _run_async(self, coro):
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Cannot use _run_async from within an async context")
        except RuntimeError:
            pass

        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
            # Check if the loop is closed (can happen in multiprocessing workers)
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop in this thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coro)

    async def _ensure_mcp_connected(self):
        """Ensure MCP manager is connected. Uses thread-local storage for thread safety."""
        if not self.mcp_servers:
            return None

        # Each thread gets its own MCP manager via thread-local storage
        if not hasattr(self._thread_local, 'mcp_manager') or not self._thread_local.mcp_manager.connected:
            print(f"[MCP] Thread {threading.get_ident()}: Connecting to servers: {self.mcp_servers}")
            self._thread_local.mcp_manager = MCPClientManager(self.mcp_servers)
            await self._thread_local.mcp_manager.connect()
            print(f"[MCP] Thread {threading.get_ident()}: Connected successfully")

        return self._thread_local.mcp_manager

    async def _execute_tool_loop_async(self, initial_input: list[dict], tools: list[dict], mcp_manager: MCPClientManager | None = None) -> Any:
        """Execute tools in a loop until no more tool calls."""
        current_input = initial_input.copy()
        max_iterations = 50

        for iteration in range(max_iterations):
            request_kwargs = {
                "model": self.model,
                "input": current_input,
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "tools": tools,
                "extra_body": {"enable_response_messages": True},
            }
            if self.developer_message:
                request_kwargs["instructions"] = self.developer_message
            if self.reasoning_model:
                request_kwargs["reasoning"] = (
                    {"effort": self.reasoning_effort} if self.reasoning_effort else None
                )
            response = self.client.responses.create(**request_kwargs)

            # Check for function calls
            has_tool_calls = any(
                hasattr(item, "type") and item.type == "function_call"
                for item in response.output
            )

            # Check for final message output (terminal condition)
            has_message_output = any(
                hasattr(item, "type") and item.type == "message"
                for item in response.output
            )

            # print(f"Iteration {iteration}: has_tool_calls={has_tool_calls}, has_message_output={has_message_output}")
            # print(f"Output items: {[item.type for item in response.output if hasattr(item, 'type')]}")
            # print("Tool Calls:", [item for item in response.output if hasattr(item, 'type') and item.type == "function_call"])

            for output in response.output:
                current_input.append(output.model_dump(mode="json"))

            if not has_tool_calls and has_message_output:
                return response, current_input

            for item in response.output:
                if hasattr(item, "type") and item.type == "function_call":
                    try:
                        arguments = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                        result = await mcp_manager.call_tool(item.name, arguments)
                        current_input.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": result,
                        })
                    except Exception as e:
                        print(e)
                        # Add error as tool output
                        current_input.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": f"Error executing tool: {str(e)}",
                        })

        # Max iterations reached - return last response
        return response, current_input

    def __call__(self, message_list: MessageList, tools: list[dict[str, Any]] = []) -> SamplerResponse:
        # Get MCP tools if configured
        mcp_tools = None
        mcp_manager = None
        if self.mcp_servers:
            mcp_manager = self._run_async(self._ensure_mcp_connected())
            mcp_tools = self._run_async(mcp_manager.get_tools_schema(format="responses"))

        all_tools = self.internal_tools + tools
        if mcp_tools:
            all_tools = all_tools + mcp_tools

        trial = 0
        while True:
            try:
                response, message_list = self._run_async(
                    self._execute_tool_loop_async(message_list, all_tools, mcp_manager)
                )

                return SamplerResponse(
                    response_text=response.output_text,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                error_msg = e.body.get('error', {}).get('message', str(e))
                error_response = {
                    "role": "assistant",
                    "content": f"Error: {error_msg}\n\nPlease review your tool usage and try again with valid tool calls."
                }
                message_list.append(error_response)

                return SamplerResponse(
                    response_text=f"BadRequestError: {error_msg}",
                    response_metadata={"usage": None, "error": error_msg},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

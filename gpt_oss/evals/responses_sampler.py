import asyncio
import json
import os
import time
from typing import Any

import openai
from openai import OpenAI

from .mcp_client_manager import MCPClientManager
from harmony.python.openai_harmony import DeveloperContent, Message, Role
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
        self.mcp_manager: MCPClientManager | None = None

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
        """Run async code in sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    async def _ensure_mcp_connected(self):
        """Lazy connect to MCP servers."""
        if self.mcp_servers and not self.mcp_manager:
            self.mcp_manager = MCPClientManager(self.mcp_servers)
            await self.mcp_manager.connect()

    async def _execute_tool_loop_async(self, initial_input: list[dict], tools: list[dict]) -> Any:
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

            print(f"Iteration {iteration}: has_tool_calls={has_tool_calls}, has_message_output={has_message_output}")
            print(f"Output items: {[item.type for item in response.output if hasattr(item, 'type')]}")
            print("Tool Calls:", [item for item in response.output if hasattr(item, 'type') and item.type == "function_call"])

            for output in response.output:
                current_input.append(output.model_dump(mode="json"))
                
            if not has_tool_calls and has_message_output:
                return response, current_input

            for item in response.output:
                if hasattr(item, "type") and item.type == "function_call":
                    try:
                        arguments = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                        result = await self.mcp_manager.call_tool(item.name, arguments)
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
        message_list = [
            {"role": "user", "content": "Use the python tool first to compute the year that was 10 years before 2015"}
        ]

        # Get MCP tools if configured
        mcp_tools = None
        if self.mcp_servers:
            self._run_async(self._ensure_mcp_connected())
            mcp_tools = self._run_async(self.mcp_manager.get_tools_schema(format="responses"))

        all_tools = self.internal_tools + tools
        if mcp_tools:
            all_tools = all_tools + mcp_tools

        trial = 0
        while True:
            try:
                response, message_list = self._run_async(
                    self._execute_tool_loop_async(message_list, all_tools)
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

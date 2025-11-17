import asyncio
import json
import time
import threading
from typing import Any

import openai
from openai import OpenAI

from .mcp_client_manager import MCPClientManager
from .types import MessageList, SamplerBase, SamplerResponse


OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionsSampler(SamplerBase):
    """Sample from a Chat Completions compatible API."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        developer_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        mcp_servers: list[tuple[str, int]] | None = None,
        enable_internal_browser: bool = False,
        enable_internal_python: bool = False,
        **kwargs: Any,
    ):
        self.client = OpenAI(base_url=base_url, timeout=24 * 60 * 60)
        self.model = model
        self.developer_message = developer_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.image_format = "url"
        self.mcp_servers = mcp_servers
        # Use thread-local storage for MCP manager since it contains async objects
        # tied to a specific event loop. Each thread needs its own instance.
        self._thread_local = threading.local() if mcp_servers else None

    def _run_async(self, coro):
        """Run async code in sync context."""
        # Check if there's already a running loop
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Cannot use _run_async from within an async context")
        except RuntimeError:
            pass 

        # Try to get the current event loop, but be defensive about its state
        try:
            loop = asyncio.get_event_loop()
            # Check if the loop is closed (can happen in worker threads)
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

    async def _execute_tool_loop_async(self, initial_messages: list[dict], tools: list[dict], mcp_manager: MCPClientManager | None = None) -> Any:
        """Execute tools in a loop until no more tool calls."""
        current_messages = initial_messages.copy()
        max_iterations = 50

        for iteration in range(max_iterations):
            # Make API call
            if self.reasoning_model:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=current_messages,
                    reasoning_effort=self.reasoning_effort,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tools,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=current_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tools=tools,
                )

            message = response.choices[0].message

            # Check for tool calls
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return response, current_messages

            # Append assistant message with tool calls
            current_messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in tool_calls
                ],
            })

            # Execute tool calls
            for tool_call in tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments) if isinstance(tool_call.function.arguments, str) else tool_call.function.arguments
                    result = await mcp_manager.call_tool(tool_call.function.name, arguments)
                    current_messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                    })
                except Exception as e:
                    # Add error as tool output
                    current_messages.append({
                        "role": "tool",
                        "content": f"Error executing tool: {str(e)}",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                    })

        # Max iterations reached - return last response
        return response, current_messages

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.developer_message:
            message_list = [
                self._pack_message("system", self.developer_message)
            ] + message_list

        # Get MCP tools if configured
        mcp_tools = None
        mcp_manager = None
        if self.mcp_servers:
            mcp_manager = self._run_async(self._ensure_mcp_connected())
            mcp_tools = self._run_async(mcp_manager.get_tools_schema(format="chat_completions"))

        trial = 0
        while True:
            try:
                # Execute with tool loop if MCP tools present
                if mcp_tools:
                    response, message_list = self._run_async(
                        self._execute_tool_loop_async(message_list, mcp_tools, mcp_manager)
                    )
                else:
                    if self.reasoning_model:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=message_list,
                            reasoning_effort=self.reasoning_effort,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=message_list,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )

                choice = response.choices[0]
                content = choice.message.content
                if getattr(choice.message, "reasoning", None):
                    message_list.append(self._pack_message("assistant", choice.message.reasoning))

                if not content:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2 ** trial  # exponential back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

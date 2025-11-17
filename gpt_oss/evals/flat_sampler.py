import asyncio
import threading
from typing import Any
import os
import traceback
from datetime import datetime
from dataclasses import dataclass
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from agents import Agent, Runner, ModelSettings, set_default_openai_client, set_default_openai_api, set_tracing_disabled, add_trace_processor, trace
from agents.mcp import MCPServerSse

from .trace_processor import LocalJSONTracingProcessor
from .types import MessageList, SamplerBase, SamplerResponse

@dataclass
class AgentEvalContext:
    cutoff_date: str = datetime.now().strftime("%Y-%m-%d")


class FlatSampler(SamplerBase):

    def __init__(
        self,
        model: str,
        developer_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 131_072,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
        verbosity: str = "medium",
        base_url: str = "http://localhost:8000/v1",
        mcp_servers: list[tuple[str, int]] | list[tuple[str, dict[str, Any]]] | None = None,
        enable_internal_browser: bool = False,
        enable_internal_python: bool = False,
        max_turns: int = 10,
        **kwargs: Any,
    ):
        self.model = model
        self.developer_message = developer_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.max_turns = max_turns
        
        self.tools = []
        if enable_internal_browser:
            from agents import WebSearchTool
            self.tools.append(WebSearchTool())
        if enable_internal_python:
            from agents import CodeInterpreterTool
            self.tools.append(CodeInterpreterTool(
                tool_config={
                    "type": "code_interpreter",
                    "container": {"type": "auto"}}
            ))

        # Convert (name, port) format to (name, {"url": "..."}) format for SSE
        self.mcp_server_configs = []
        if mcp_servers:
            for server in mcp_servers:
                name, config = server
                if isinstance(config, int):
                    self.mcp_server_configs.append((name, {"url": f"http://localhost:{config}/sse"}))
                else:
                    self.mcp_server_configs.append((name, config))
        
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            base_url=None if base_url == "None" else base_url,
            timeout=24 * 60 * 60,
        )

        self._thread_local = threading.local() if mcp_servers else None

        set_tracing_disabled(False)
        set_default_openai_client(self.client)
        set_default_openai_api("responses")
        add_trace_processor(LocalJSONTracingProcessor(output_dir="agent_traces"))

    def _run_async(self, coro):
        """Run async code in sync context."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError("Cannot use _run_async from within an async context")
        except RuntimeError:
            pass

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coro)

    async def _ensure_mcp_connected(self) -> list[MCPServerSse]:
        """Ensure MCP servers are connected. Uses thread-local storage."""
        if not self.mcp_server_configs:
            return []

        if not hasattr(self._thread_local, 'mcp_servers') or not self._thread_local.mcp_servers:
            print(f"[MCP] Thread {threading.get_ident()}: Connecting to {len(self.mcp_server_configs)} servers")

            servers = []
            for name, params in self.mcp_server_configs:
                server = MCPServerSse(
                    name=name,
                    params=params,
                    cache_tools_list=True,
                )
                await server.connect()
                servers.append(server)

            self._thread_local.mcp_servers = servers
            print(f"[MCP] Thread {threading.get_ident()}: Connected successfully")

        return self._thread_local.mcp_servers

    def _convert_messages(self, message_list: MessageList, mcp_servers: list[MCPServerSse]) -> tuple[str | None, str]:
        instructions = self.developer_message
        user_prompt = ""

        # Get last user message as prompt
        for msg in reversed(message_list):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break
            
        server_instructions = ""
        for server in mcp_servers:
            server_instructions += f"\n\n## {server.name}\n\n{server.server_initialize_result.instructions}"
        instructions = instructions.format(server_instructions=server_instructions)

        return instructions, user_prompt

    async def _execute_async(self, message_list: MessageList, context: AgentEvalContext) -> SamplerResponse:
        mcp_servers = await self._ensure_mcp_connected()

        instructions, user_prompt = self._convert_messages(message_list, mcp_servers)
        
        # Reasoning model settings
        model_settings = ModelSettings(verbosity=self.verbosity, parallel_tool_calls=True)
        if self.reasoning_model and self.reasoning_effort:
            model_settings = ModelSettings(reasoning=Reasoning(effort=self.reasoning_effort, summary='detailed'), verbosity=self.verbosity, parallel_tool_calls=True)
            

        mcp_names = ','.join([name for name, _ in self.mcp_server_configs])
        agent = Agent(
            name=f'model_{self.model}_mcp_{mcp_names}',
            instructions=instructions,
            model=self.model,
            mcp_servers=mcp_servers,
            model_settings=model_settings,
            tools=self.tools,
        )
        
        try:
            with trace('Superforecaster') as t:
                result = await Runner.run(
                    agent,
                    input=f"{user_prompt}",
                    context=context,
                    max_turns=self.max_turns
                )
                t.finish()

                response_text = result.final_output or ""
                updated_messages = result.to_input_list()

                return SamplerResponse(
                    response_text=response_text,
                    response_metadata={"usage": None},
                    actual_queried_message_list=updated_messages,
                    spans=t.trace_data["spans"],
                )

        except Exception as e:
            print(f"Error during agent execution: {e}")
            print(traceback.format_exc())
            return SamplerResponse(
                response_text=f"Error: {str(e)}",
                response_metadata={"usage": None, "error": str(e)},
                actual_queried_message_list=message_list,
                spans=t.trace_data["spans"],
            )

    def __call__(self, message_list: MessageList, cutoff_date: str = None) -> SamplerResponse:
        if cutoff_date:
            context = AgentEvalContext(cutoff_date=cutoff_date)
        else:
            context = AgentEvalContext()
        return self._run_async(self._execute_async(message_list, context))

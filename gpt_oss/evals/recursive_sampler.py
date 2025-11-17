import asyncio
import traceback
import threading
from typing import Any
from dataclasses import dataclass, field
import os

from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from agents import trace, Agent, Runner, ModelSettings, function_tool, RunContextWrapper, set_default_openai_client, set_default_openai_api, set_tracing_disabled, add_trace_processor
from agents.mcp import MCPServerSse

from .trace_processor import LocalJSONTracingProcessor
from .types import MessageList, SamplerBase, SamplerResponse


@dataclass
class RecursiveForecastingContext:
    max_depth: int = 2
    current_depth: int = 0
    max_turns: int = 200,
    cutoff_date: str = ""
    original_question: str = ""
    question_path: list[str] = field(default_factory=list)
    base_agent: Agent = None
    instructions_template: str = ""
    mcp_servers: list[MCPServerSse] = field(default_factory=list)

    def increment_depth(self, new_subquestion: str) -> "RecursiveForecastingContext":
        return RecursiveForecastingContext(
            max_depth=self.max_depth,
            current_depth=self.current_depth + 1,
            max_turns=self.max_turns,
            cutoff_date=self.cutoff_date,
            original_question=self.original_question,
            question_path=self.question_path + [new_subquestion],
            base_agent=self.base_agent,
            instructions_template=self.instructions_template,
            mcp_servers=self.mcp_servers
        )

@function_tool
async def create_subagent(
    ctx: RunContextWrapper[RecursiveForecastingContext],
    subquestion: str,
    question_type: str = "binary",
) -> str:
    """Create a subagent to make a prediction for a subquestion. The subagent will return a full prediciton report including its final prediction in \prediction{} with the correct formatting depending on the question type in your output. The subagent has access to the same tools as you.

    Args:
        subquestion: Specific question to predict (make it clear and self-contained). Be concise in formulating the subquestion. Subquestions should be mutually exclusive and answer different causal components of the overall question.
        question_type: Type of question - one of "binary", "multiple_choice", or "numeric". This is used to instruct the subagent to format the final prediction in \prediction{} with the correct formatting.

    Returns:
        Prediction with reasoning and conclusions
    """
    if ctx.context.current_depth >= ctx.context.max_depth:
        return f"⚠️ Max depth ({ctx.context.max_depth}) reached. You must predict this directly: {subquestion}"

    new_context = ctx.context.increment_depth(subquestion)
    
    server_instructions = ""
    for server in ctx.context.mcp_servers:
        server_instructions += f"\n\n## {server.name}\n\n{server.server_initialize_result.instructions}"

    specialist = ctx.context.base_agent.clone(
        name="Subagent",
        instructions=ctx.context.instructions_template.format(server_instructions=server_instructions, max_depth=ctx.context.max_depth, current_depth=ctx.context.current_depth)
    )

    prompt = f"""**Original Root Question of the Overall Forecast:** {new_context.original_question}
**Question Path of the Subquestion:** {' → '.join(new_context.question_path)}
**Cutoff Date:** {new_context.cutoff_date}
**Question Type:** {question_type}

**YOUR Subquestion to Predict:**
{subquestion}
"""
    
    result = await Runner.run(specialist, input=prompt, context=new_context, max_turns=ctx.context.max_turns)

    return f"\n{'='*60}\n**Subquestion:** {subquestion}\n\n{result.final_output}\n{'='*60}\n"


class RecursiveSampler(SamplerBase):
    """
    Sampler using recursive agent pattern with subagent delegation.
    Leverages hierarchical decomposition for complex forecasting tasks.
    """

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
        max_depth: int = 2,
        max_turns: int = 200,
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
        self.max_depth = max_depth
        self.max_turns = max_turns

        # Convert MCP server configs
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

        return self._thread_local.mcp_servers

    def _convert_messages(self, message_list: MessageList, mcp_servers: list[MCPServerSse], max_depth: int) -> tuple[str | None, str]:
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
        instructions = instructions.format(server_instructions=server_instructions, max_depth=max_depth, current_depth=0)

        return instructions, user_prompt

    async def _execute_async(self, message_list: MessageList, cutoff_date: str) -> SamplerResponse:
        mcp_servers = await self._ensure_mcp_connected()

        instructions, user_prompt = self._convert_messages(message_list, mcp_servers, max_depth=self.max_depth)

        # Model settings
        model_settings = ModelSettings(verbosity=self.verbosity, parallel_tool_calls=True)
        if self.reasoning_model and self.reasoning_effort:
            model_settings = ModelSettings(
                reasoning=Reasoning(effort=self.reasoning_effort, summary='detailed'),
                verbosity=self.verbosity,
                parallel_tool_calls=True,
            )

        # Create base agent
        agent = Agent[RecursiveForecastingContext](
            model=self.model,
            name="Superforecaster",
            instructions=instructions,
            tools=[create_subagent],
            mcp_servers=mcp_servers,
            model_settings=model_settings,
        )

        # Create context
        context = RecursiveForecastingContext(
            max_depth=self.max_depth,
            current_depth=0,
            max_turns=self.max_turns,
            cutoff_date=cutoff_date,
            original_question=user_prompt,
            question_path=["Root"],
            base_agent=agent,
            instructions_template=self.developer_message,
            mcp_servers=mcp_servers,
        )

        try:
            with trace('Recursive Superforecaster') as t:
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
        return self._run_async(self._execute_async(message_list, cutoff_date or ""))

import argparse
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

from openai.types.shared import Reasoning
from openai import AsyncOpenAI
from agents import Agent, Runner, function_tool, RunContextWrapper, ModelSettings, ItemHelpers, trace, set_default_openai_client, set_default_openai_api, set_tracing_disabled, add_trace_processor, TResponseInputItem
from agents.mcp import MCPServerSse
import weave
from trace_processor import LocalJSONTracingProcessor

PROMPTS_DIR = Path(__file__).parent.parent.parent / "gpt_oss" / "prompts"
UNIFIED_PROMPT = (PROMPTS_DIR / "unified_forecaster.md").read_text(encoding="utf-8")
USER_QUESTION = (PROMPTS_DIR / "user.md").read_text(encoding="utf-8") + "CAll the search_wikipedia tool first!!!"

client = AsyncOpenAI(
    timeout=24 * 60 * 60,
)

set_tracing_disabled(False)
weave.init("openai_agents")
set_default_openai_client(client)
set_default_openai_api("responses")
add_trace_processor(LocalJSONTracingProcessor(output_dir="traces"))

@dataclass
class ForecastingContext:
    max_depth: int = 3
    current_depth: int = 0
    max_turns: int = 2
    cutoff_date: str = "2024-11-14"
    original_question: str = ""
    question_path: list[str] = field(default_factory=list)
    base_agent: Agent = None # main agent for cloning

    def increment_depth(self, new_subquestion: str) -> "ForecastingContext":
        return ForecastingContext(
            max_depth=self.max_depth,
            current_depth=self.current_depth + 1,
            max_turns=self.max_turns,
            cutoff_date=self.cutoff_date,
            original_question=self.original_question,
            question_path=self.question_path + [new_subquestion],
            base_agent=self.base_agent
        )

@function_tool
async def create_subagent(
    ctx: RunContextWrapper[ForecastingContext],
    subquestion: str,
    context_for_agent: str
) -> str:
    """Create a subagent to analyze a subquestion.

    The specialist will decide whether to further delegate or analyze directly based on question complexity.

    Args:
        subquestion: Specific question to analyze (make it clear and self-contained)
        context_for_agent: Background context explaining how this fits into the broader forecast

    Returns:
        Analysis with probability estimate, reasoning, and uncertainties
    """
    if ctx.context.current_depth >= ctx.context.max_depth:
        return f"⚠️ Max depth ({ctx.context.max_depth}) reached. You must analyze this directly: {subquestion}"

    depth = ctx.context.current_depth + 1
    new_context = ctx.context.increment_depth(subquestion)

    specialist = ctx.context.base_agent.clone(
        name=f"Forecaster", 
        instructions=UNIFIED_PROMPT.format(max_depth=ctx.context.max_depth, current_depth=ctx.context.current_depth)
    )

    prompt = f"""
**Main Root Question:** {new_context.original_question}
**Question Path:** {' → '.join(new_context.question_path)}
**Cutoff Date:** {new_context.cutoff_date}

**YOUR Subquestion:**
{subquestion}

**Context:**
{context_for_agent}
"""

    print(f"\n{'  ' * depth}🔄 Depth {depth}: {subquestion}")

    result = await Runner.run(specialist, input=prompt, context=new_context, max_turns=3)

    print(f"{'  ' * depth}✅ Done\n")

    return f"\n{'='*60}\n**Subquestion:** {subquestion}\n\n{result.final_output}\n{'='*60}\n"


def create_recursive_agent(model: str, reasoning_effort: str, summary: str, verbosity: str, max_depth: int, current_depth: int, mcp_servers=None):
    return Agent[ForecastingContext](
        model=model,
        model_settings=ModelSettings(reasoning=Reasoning(effort=reasoning_effort, summary=summary), verbosity=verbosity, parallel_tool_calls=True),
        name="Superforecaster",
        instructions=UNIFIED_PROMPT.format(max_depth=max_depth, current_depth=current_depth),
        tools=[create_subagent],
        mcp_servers=mcp_servers or []
    )

async def main(args):
    mcp_servers = []
    if args.mcp:
        mcp_mapping = {
            "wikipedia": ("wikipedia", 8003),
            "browser": ("browser", 8001),
            "python": ("python", 8002),
            "google-trends": ("google-trends", 8004),
            "metaculus": ("metaculus", 8005),
            "financial-datasets": ("financial-datasets", 8006),
            "datacommons": ("datacommons", 8007),
        }

        for mcp_name in args.mcp.split(","):
            if mcp_name not in mcp_mapping:
                print(f"Warning: Unknown MCP server '{mcp_name}', skipping")
                continue

            name, port = mcp_mapping[mcp_name]
            server = MCPServerSse(
                name=name,
                params={"url": f"http://localhost:{port}/sse"},
                cache_tools_list=True,
            )
            await server.connect()
            mcp_servers.append(server)

        print(f"Connected to MCP servers: {[s.name for s in mcp_servers]}")

    try:
        agent = create_recursive_agent(args.model, args.reasoning_effort, args.summary, args.verbosity, args.max_depth, 0, mcp_servers=mcp_servers)

        context = ForecastingContext(
            max_depth=args.max_depth,
            current_depth=0,
            max_turns=args.max_turns,
            cutoff_date=args.cutoff_date,
            original_question=USER_QUESTION.strip(),
            question_path=["Root"],
            base_agent=agent
        )

        print(f"\n{'='*80}\nRECURSIVE SUPERFORECASTER | Max Depth: {args.max_depth} | Cutoff: {args.cutoff_date}\n{'='*80}\n")

        prompt = f"""{USER_QUESTION}

**Cutoff Date:** {args.cutoff_date}"""

        with trace(args.trace_name):
            result = Runner.run_streamed(agent, input=prompt, context=context, max_turns=args.max_turns)

            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    continue
                elif event.type == "agent_updated_stream_event":
                    print(f"Agent updated: {event.new_agent.name}")
                    continue
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        print("-- Tool was called")
                    elif event.item.type == "tool_call_output_item":
                        print(f"-- Tool output: {event.item.output}")
                    elif event.item.type == "message_output_item":
                        print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
                    else:
                        pass  # Ignore other event types
        
        print(f"\n\n{'='*80}\nCOMPLETE\n{'='*80}\n")

    finally:
        for server in mcp_servers:
            try:
                await server.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up MCP server {server.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursive superforecasting agent")
    parser.add_argument("--trace-name", default="Recursive Superforecaster", help="Trace name")
    parser.add_argument("--model", default="gpt-5-mini", help="Model to use")
    parser.add_argument("--reasoning-effort", default="minimal", choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--summary", default="auto", choices=["auto", "concise", "detailed"])
    parser.add_argument("--verbosity", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--max-depth", type=int, default=2, help="Max recursion depth")
    parser.add_argument("--max-turns", type=int, default=2, help="Max turns per agent")
    parser.add_argument("--cutoff-date", default="2024-11-14", help="Forecast cutoff date")
    parser.add_argument("--mcp", default="", help="Comma-separated list of MCP servers (wikipedia,browser,python,google-trends,metaculus,financial-datasets,datacommons)")

    asyncio.run(main(parser.parse_args()))

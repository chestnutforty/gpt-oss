# `gpt_oss.evals`

This module is a reincarnation of [simple-evals](https://github.com/openai/simple-evals) adapted for gpt-oss. It lets you
run GPQA and HealthBench against a runtime that supports Responses API on `localhost:8080/v1`.

## Samplers

### ApiSampler
Standard sampler using OpenAI's Agents SDK with MCP server support.

### RecursiveSampler
Sampler using recursive agent pattern with hierarchical subagent delegation. Enables complex task decomposition where agents can spawn specialist subagents to analyze subquestions.

**Features:**
- Hierarchical task decomposition via `create_subagent` tool
- Configurable max recursion depth
- Tracks question path and context across delegation levels
- Compatible with MCP servers and reasoning models

**Usage:**
```python
from gpt_oss.evals import RecursiveSampler

sampler = RecursiveSampler(
    model="gpt-5-mini",
    developer_message="You are a forecasting agent...",
    mcp_servers=[("wikipedia", 8003), ("browser", 8001)],
    max_depth=2,
    max_turns=10,
    reasoning_model=True,
    reasoning_effort="medium"
)

response = sampler(message_list, cutoff_date="2024-11-14")
```
# GPT-OSS Superforecasting Agent

Fork of gpt-oss for running superforecasting agent evaluations on locally-hosted models.

## Quick Start

### 1. Installation

```bash
uv sync                    # Install dependencies from lockfile
uv pip install vllm==0.10.2 --torch-backend=auto
```

### 2. Serve Model

```bash
uv run serve-model openai/gpt-oss-20b 
```

Available at `http://localhost:8000/v1`

### 3. Set Up MCP Servers (Optional)

```bash
# Clone submodules
git submodule update --init --recursive

# Install all MCP servers
uv run install-servers

# Start servers (separate terminals)
uv run run-server wikipedia  # Port 8003
uv run run-server browser    # Port 8001
uv run run-server python     # Port 8002
```

**Required environment variables:**
Run each server in a separate tmux session to keep them running. Make sure to set env vars. 

### 4. Run Evaluations

```bash
# Basic Polymarket eval (no tools)
uv run python -m gpt_oss.evals --model openai/gpt-oss-20b --eval polymarket --reasoning-effort high --developer-message superforecaster

# With wikipedia mcp
python -m gpt_oss.evals \
  --model gpt-oss-20b \
  --reasoning-effort high \
  --eval polymarket \
  --mcp wikipedia \
  --developer-message superforecaster

# With internal tools (handled by API server)
python -m gpt_oss.evals \
  --model gpt-oss-20b \
  --reasoning-effort high \
  --eval polymarket \
  --enable-internal-browser \
  --enable-internal-python \
  --developer-message superforecaster

# Custom data and only 10 samples
python -m gpt_oss.evals \
  --eval polymarket \
  --polymarket-data-path data/my_data.jsonl \
  --examples 10
```

**Key options:**
- `--model`: Model name(s), comma-separated
- `--eval`: polymarket, gpqa, healthbench, aime25, basic
- `--reasoning-effort`: low, medium, high (comma-separated)
- `--sampler`: responses (default) or chat_completions
- `--mcp`: wikipedia,browser,python,google-trends,metaculus,financial-datasets
- `--examples`: Limit examples for testing

**Output:**

`results/<eval>_<model>_<timestamp>_allresults.json`

## MCP Servers

| Server | Port | Tools | Purpose |
|--------|------|-------|---------|
| Wikipedia | 8003 | search, get_article | Historical research, base rates |
| Browser | 8001 | search, open, find | Web research, current events |
| Python | 8002 | execute | Calculations, data analysis |
| Google Trends | 8004 | search_trends | Search popularity |
| Metaculus | 8005 | search_questions | Forecasting platform |
| Financial Datasets | 8006 | get_data | Market/economic data |

See [mcp-servers/README.md](mcp-servers/README.md) for details.

## Terminal Chat

```bash
# Basic
python -m gpt_oss.chat gpt-oss-20b/original/ --backend vllm

# With tools
python -m gpt_oss.chat gpt-oss-20b/original/ \
  --backend vllm \
  --browser \
  --python \
  --reasoning-effort medium
```
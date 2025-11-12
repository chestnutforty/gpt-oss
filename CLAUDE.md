# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains reference implementations for OpenAI's gpt-oss models (gpt-oss-120b and gpt-oss-20b), which are open-weight models designed for reasoning, agentic tasks, and developer use cases. The models use the Harmony response format and support native capabilities for function calling, web browsing, Python code execution, and Structured Outputs.

## Package Manager

This project uses standard `pip` for dependency management. Dependencies are defined in `pyproject.toml`.

## Key Commands

### Installation

```bash
# Install from PyPI
pip install gpt-oss                    # Tools only
pip install gpt-oss[torch]            # PyTorch implementation
pip install gpt-oss[triton]           # Triton implementation

# Install from source (for local development)
git clone https://github.com/openai/gpt-oss.git
pip install -e ".[triton]"            # Editable install with triton
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"  # Metal for Apple Silicon
```

### Model Download

```bash
# Download models from Hugging Face
hf download openai/gpt-oss-120b --include "original/*" --local-dir gpt-oss-120b/
hf download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/

# Download Metal-format weights
hf download openai/gpt-oss-120b --include "metal/*" --local-dir gpt-oss-120b/metal/
hf download openai/gpt-oss-20b --include "metal/*" --local-dir gpt-oss-20b/metal/
```

### Serving Models

```bash
# Serve a model using the serve-model script
serve-model openai/gpt-oss-20b                    # Serves on default port 8000
serve-model openai/gpt-oss-120b --port 8080       # Custom port
serve-model --list                                 # List available models

# Serve using vLLM directly
vllm serve openai/gpt-oss-20b

# Serve using Responses API
python -m gpt_oss.responses_api.serve --checkpoint openai/gpt-oss-20b --inference-backend vllm --port 8000
```

Available inference backends: `triton`, `torch`, `vllm`, `metal`, `ollama`, `transformers`

### Running Inference

```bash
# PyTorch implementation (requires 4xH100 for 120b)
torchrun --nproc-per-node=4 -m gpt_oss.generate gpt-oss-120b/original/

# Triton implementation (runs on single 80GB GPU)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gpt_oss.generate --backend triton gpt-oss-120b/original/

# Metal implementation (Apple Silicon)
python gpt_oss/metal/scripts/create-local-model.py -s <model_dir> -d <output_file>
python gpt_oss/metal/examples/generate.py gpt-oss-20b/metal/model.bin -p "your prompt"
```

### Terminal Chat

```bash
# Basic chat (torch/triton/vllm backends)
python -m gpt_oss.chat gpt-oss-20b/original/ --backend vllm

# With tools enabled
python -m gpt_oss.chat gpt-oss-20b/original/ \
  --backend vllm \
  --browser \
  --python \
  --reasoning-effort medium

# Options:
#   -r, --reasoning-effort {low,medium,high}
#   -b, --browser          Enable browser tool
#   -p, --python           Enable python tool
#   -a, --apply-patch      Enable apply_patch tool
#   --raw                  Raw mode (no Harmony encoding)
```

### Testing

```bash
# Run tests
pytest

# Run specific test files
pytest tests/test_responses_api.py
pytest tests/test_api_endpoints.py
pytest tests/gpt_oss/tools/simple_browser/test_backend.py

# Run MCP server tests
python gpt-oss-mcp-server/test_builtin_tools.py
python gpt-oss-mcp-server/test_mcp_servers.py
```

### Evaluations

```bash
# Run GPQA or HealthBench evaluations
# Requires a Responses API server running on localhost:8080/v1
python -m gpt_oss.evals.<eval_name>
```

## Architecture

### Core Components

**gpt_oss/torch/model.py**: Reference PyTorch implementation showing exact model architecture. Runs in BF16 with basic tensor parallelism for MoE. Educational purposes only, requires 4xH100 for the 120b model.

**gpt_oss/triton/model.py**: Optimized implementation using Triton kernels with MXFP4 quantization support. Runs 120b model on single 80GB GPU. Uses optimized attention and MoE kernels.

**gpt_oss/metal/**: Metal-specific implementation for Apple Silicon. Requires weight conversion from SafeTensor format.

**gpt_oss/vllm/token_generator.py**: vLLM backend integration for production inference with batching and optimization.

**gpt_oss/responses_api/**: Example Responses API-compatible server implementation with multiple inference backend support (triton, metal, ollama, vllm, transformers).

**gpt_oss/tools/**: Reference tool implementations:
- `simple_browser/`: Browser tool with search, open, and find capabilities. Supports YouCom and Exa backends.
- `python_docker/`: Stateless Python execution in Docker containers.
- `apply_patch.py`: File creation/update/deletion tool.

**gpt_oss/chat.py**: Terminal chat application demonstrating Harmony format usage with tools and reasoning effort control.

### Harmony Format

The models require the `openai-harmony` library for proper prompt formatting. Key concepts:

- **Conversation**: Structured message list with roles (SYSTEM, USER, ASSISTANT, DEVELOPER)
- **SystemContent**: Contains configuration (reasoning effort, tools, conversation date)
- **Encoding**: Converts conversations to/from token IDs
- **Tool Integration**: Tools defined in system message using `with_tools()` or `with_browser_tool()`/`with_python()`

### Tool System

Tools are integrated via the Harmony format:

1. Define tool configuration in system message
2. Model generates tool calls during inference
3. Parse tool calls from completion tokens
4. Execute tool and feed results back
5. Continue inference loop

Each tool implements:
- `tool_config`: Harmony-compatible tool definition
- `process(message)`: Async method to handle tool invocations

### MCP Server Integration

The `gpt-oss-mcp-server/` directory contains MCP (Model Context Protocol) server implementations for tools, enabling tool use via standardized protocols. Use `build-system-prompt.py` to generate system prompts from MCP service discovery.

### Inference Backends

- **triton**: Optimized for single-GPU with MXFP4, CUDA graphs, caching
- **torch**: Basic PyTorch, multi-GPU via tensor parallelism, educational
- **metal**: Apple Silicon via Metal Performance Shaders
- **vllm**: Production inference with batching and optimization
- **ollama**: Consumer hardware support via Ollama API
- **transformers**: Hugging Face Transformers library integration

## Environment Variables

Required for tools:
- `YDC_API_KEY`: For YouCom browser backend
- `EXA_API_KEY`: For Exa browser backend

Optional:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Recommended for triton backend to avoid OOM
- `VLLM_USE_FLASHINFER_SAMPLER=0`: Disable FlashInfer sampler in vLLM if needed
- `GPTOSS_BUILD_METAL=1`: Enable Metal build during installation

## Important Implementation Details

### MXFP4 Quantization

Models use native MXFP4 quantization for MoE linear projection weights:
- `tensor.blocks`: FP4 values packed as uint8 (2 values per byte)
- `tensor.scales`: Block-wise scaling factors
- All other tensors in BF16
- Recommended activation precision: BF16

### Sampling Parameters

Recommended: `temperature=1.0` and `top_p=1.0`

### Tool Usage Patterns

**Browser Tool**:
- Uses scrollable text window for context management
- Caches requests for revisiting pages
- Create new browser instance per request
- Model trained to provide citations from browser results

**Python Tool**:
- Reference implementation is stateless (model trained with stateful)
- Runs in permissive Docker container (implement restrictions for production)
- Tool definition overrides default in openai-harmony

### Harmony Encoding Flow

```python
# 1. Build conversation with system config
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
system_content = SystemContent.new().with_tools(tool_config)
messages = [Message.from_role_and_content(Role.SYSTEM, system_content), ...]
conversation = Conversation.from_messages(messages)

# 2. Convert to tokens
token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

# 3. Run inference
outputs = model.generate(prompt_token_ids=[token_ids], ...)

# 4. Parse structured output
parsed_messages = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
```

## Project Structure

- `gpt_oss/torch/`: Reference PyTorch implementation
- `gpt_oss/triton/`: Optimized Triton implementation
- `gpt_oss/metal/`: Metal implementation for Apple Silicon
- `gpt_oss/vllm/`: vLLM integration
- `gpt_oss/tools/`: Reference tool implementations (browser, python, apply_patch)
- `gpt_oss/responses_api/`: Responses API server with multiple backends
- `gpt_oss/evals/`: Evaluation harness (GPQA, HealthBench)
- `scripts/`: Utility scripts (serve_model, install_all, run_server)
- `tests/`: Test suite
- `gpt-oss-mcp-server/`: MCP server implementations for tools
- `harmony/`: Git submodule for Harmony library
- `mcp-servers/`: Git submodules for MCP server templates

## Common Workflows

### Adding a New Tool

1. Create tool class implementing `tool_config` property and `async process(message)` method
2. Add tool to imports in `gpt_oss/tools/__init__.py`
3. Update chat.py or Responses API to expose the tool
4. Define tool in system message using `.with_tools(tool.tool_config)`
5. Handle tool invocations in inference loop by parsing messages and calling `tool.process()`

### Running with Different Backends

The codebase supports flexible backend selection:

**Local GPU**: Use `--backend triton` or `--backend torch`
**Apple Silicon**: Use `--backend metal` (requires weight conversion)
**vLLM**: Use `--backend vllm` (requires vLLM installed)
**Ollama**: Use `--inference-backend ollama` in Responses API

### Creating MCP Servers for Tools

See `gpt-oss-mcp-server/` for reference implementations. MCP servers enable:
- Tool discovery via standardized protocol
- Integration with MCP-compatible clients
- Automatic system prompt generation from service discovery

## Development Notes

- Models require Harmony format; do not use standard chat templates
- Reference implementations are educational; optimize for production
- PyTorch implementation requires 4xH100 GPUs; use Triton for single GPU
- Create new browser tool instance per request for proper caching
- Python tool runs in permissive Docker; add security restrictions for production
- Triton backend requires building from source: `pip install -e . --no-build-isolation` in triton repo
- Metal weights must be converted from SafeTensor format or downloaded pre-converted
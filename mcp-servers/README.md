# GPT-OSS MCP Servers

This directory contains Model Context Protocol (MCP) servers for GPT-OSS. Each server is an independent git repository (added as a submodule) with its own dependencies and can be installed and run separately.

## Available Servers

### Wikipedia (Port 8003)
- **Repository**: `mcp-wikipedia/`
- **Description**: Search Wikipedia articles with time-travel capability
- **Key Features**: Historical article retrieval, multi-language support, revision history
- **Use Cases**: Historical research, fact-checking, reference statistics, base rates
- **Dependencies**: Self-contained (httpx, wikitextparser)

### Browser (Port 8001)
- **Repository**: `mcp-browser/`
- **Description**: Web browsing and search capabilities
- **Key Features**: Web search (Exa/YouCom), link navigation, pattern matching, session management
- **Use Cases**: Web research, information gathering, source citation
- **Dependencies**: Requires main `gpt-oss` package

### Python (Port 8002)
- **Repository**: `mcp-python/`
- **Description**: Secure Python code execution in Docker containers
- **Key Features**: Sandboxed execution, stateless containers, stdout capture
- **Use Cases**: Mathematical computations, data analysis, algorithm testing
- **Dependencies**: Requires main `gpt-oss` package and Docker

### Template
- **Repository**: `mcp-template/`
- **Description**: Minimal template for creating new MCP servers
- **Key Features**: Simple single-file structure, example tool, quick start guide
- **Use Cases**: Starting point for building custom MCP tools
- **Note**: This is a template, not a runnable server

## Installation

### Install All Servers

```bash
# From the gpt-oss root directory
./scripts/install_all.sh
```

This will run `uv sync` in each server directory to install dependencies.

### Install Individual Servers

```bash
# Navigate to the server directory
cd mcp-servers/mcp-wikipedia

# Install with uv
uv sync
```

## Running Servers

### Using the Helper Script

```bash
# From the gpt-oss root directory
python scripts/run_server.py <server_name>

# Examples:
python scripts/run_server.py wikipedia
python scripts/run_server.py browser
python scripts/run_server.py python
```

### Running Directly

```bash
# Navigate to the server directory and run
cd mcp-servers/mcp-wikipedia
python wikipedia_server.py
```

## Git Submodules

Each MCP server is maintained as a separate git repository and added to this repo as a submodule. This provides:
- Independent version control and release cycles
- Isolated dependencies per server
- Ability to use servers in other projects

### Cloning with Submodules

When cloning the main gpt-oss repository:

```bash
# Clone with all submodules
git clone --recurse-submodules <repo-url>

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### Updating Submodules

```bash
# Update all submodules to latest commits
git submodule update --remote

# Update specific submodule
git submodule update --remote mcp-servers/mcp-wikipedia
```

## Environment Variables

### Wikipedia Server
- `WIKIPEDIA_ACCESS_TOKEN` (optional): Personal access token for higher rate limits

### Browser Server
- `BROWSER_BACKEND` (optional): Choose "exa" (default) or "youcom"
- Backend-specific API keys (e.g., `EXA_API_KEY`)

### Python Server
- Requires Docker installed and running

## Port Assignments

| Server | Port | Protocol |
|--------|------|----------|
| Browser | 8001 | MCP/HTTP |
| Python | 8002 | MCP/HTTP |
| Wikipedia | 8003 | MCP/HTTP |

## Architecture

Each server uses the FastMCP framework and implements the Model Context Protocol (MCP) for tool integration with language models.

**Common patterns:**
- FastMCP for server setup and tool registration
- Async/await for non-blocking I/O
- Type hints with `Annotated` for parameter documentation
- Error handling and validation
- Structured logging

**Server types:**
- **Self-contained** (Wikipedia): All code and dependencies in the server repo
- **Main package dependent** (Browser, Python): Import tools from main `gpt-oss` package

## Adding New Servers

To add a new MCP server:

1. Create a new repository with the structure:
   ```
   mcp-<name>/
   ├── pyproject.toml
   ├── README.md
   ├── .gitignore
   └── <name>_server.py
   ```

2. Add as a submodule:
   ```bash
   git submodule add <repo-url> mcp-servers/mcp-<name>
   ```

3. Update `scripts/run_server.py` to include the new server

4. Update `scripts/install_all.sh` to include the new server

5. Document the server in this README

## Documentation

Each server has its own README.md with detailed information:
- [Wikipedia Server](mcp-wikipedia/README.md)
- [Browser Server](mcp-browser/README.md)
- [Python Server](mcp-python/README.md)

## License

See the main gpt-oss repository for license information.

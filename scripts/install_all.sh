#!/bin/bash
# Install all MCP servers using uv sync

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MCP_SERVERS_DIR="$REPO_ROOT/mcp-servers"

echo "Installing all MCP servers..."
echo "=============================="
echo

# Array of servers to install
SERVERS=("mcp-wikipedia" "mcp-browser" "mcp-python")

for server in "${SERVERS[@]}"; do
    server_dir="$MCP_SERVERS_DIR/$server"

    if [ ! -d "$server_dir" ]; then
        echo "Warning: Server directory not found: $server_dir"
        echo "Skipping $server..."
        echo
        continue
    fi

    echo "Installing $server..."
    cd "$server_dir"

    if [ -f "pyproject.toml" ]; then
        uv sync
        echo "✓ $server installed successfully"
    else
        echo "Warning: No pyproject.toml found in $server_dir"
        echo "Skipping $server..."
    fi

    echo
done

echo "=============================="
echo "Installation complete!"
echo
echo "To run a server, use:"
echo "  python scripts/run_server.py <server_name>"
echo
echo "Available servers:"
echo "  - wikipedia (port 8003)"
echo "  - browser (port 8001)"
echo "  - python (port 8002)"

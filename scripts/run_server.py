#!/usr/bin/env python3
"""
Helper script to run MCP servers.

Usage:
    python scripts/run_server.py <server_name>

Available servers are defined in gpt_oss/evals/mcp_servers_config.py
"""
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import importlib.util

# Import centralized MCP server config without triggering package __init__.py
config_path = Path(__file__).parent.parent / "gpt_oss" / "evals" / "mcp_servers_config.py"
spec = importlib.util.spec_from_file_location("mcp_servers_config", config_path)
mcp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_config)

MCP_SERVERS = mcp_config.MCP_SERVERS
get_server_by_name = mcp_config.get_server_by_name

load_dotenv(override=True)

# Build server maps from centralized config
SERVERS = {
    server.short_name: f"mcp-servers/{server.name}/{server.entry_script}"
    for server in MCP_SERVERS
}

PORT_INFO = {
    server.short_name: server.port
    for server in MCP_SERVERS
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <server_name>")
        print(f"\nAvailable servers:")
        for name, port in PORT_INFO.items():
            print(f"  - {name} (port {port})")
        sys.exit(1)

    server_name = sys.argv[1]

    if server_name not in SERVERS:
        print(f"Error: Unknown server '{server_name}'")
        print(f"\nAvailable servers: {', '.join(SERVERS.keys())}")
        sys.exit(1)

    # Get the script path relative to the repo root
    repo_root = Path(__file__).parent.parent
    server_script = repo_root / SERVERS[server_name]
    server_dir = server_script.parent

    if not server_script.exists():
        print(f"Error: Server script not found at {server_script}")
        sys.exit(1)

    # Check for venv in the server directory
    venv_mcp = server_dir / ".venv" / "bin" / "fastmcp"
    if not venv_mcp.exists():
        print(f"Error: MCP executable not found at {venv_mcp}")
        print(f"Please run './scripts/install_all.sh' first to install dependencies.")
        sys.exit(1)

    port = PORT_INFO[server_name]
    print(f"Starting {server_name} MCP server on port {port}...")
    print(f"Script: {server_script}")
    print(f"Using MCP: {venv_mcp}")
    print("-" * 50)

    # Run the server using fastmcp run -t sse --port <port>
    try:
        subprocess.run(
            [str(venv_mcp), "run", "-t", "sse", "--port", str(port), f"{server_script}"],
            check=True
        )
    except KeyboardInterrupt:
        print(f"\n{server_name} server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {server_name} server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

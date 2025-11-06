#!/usr/bin/env python3
"""
Helper script to run MCP servers.

Usage:
    python scripts/run_server.py <server_name>

Available servers:
    - wikipedia (port 8003)
    - browser (port 8001)
    - python (port 8002)
"""
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Map server names to their entry point scripts
SERVERS = {
    "wikipedia": "mcp-servers/mcp-wikipedia/wikipedia_server.py",
    "browser": "mcp-servers/mcp-browser/browser_server.py",
    "python": "mcp-servers/mcp-python/python_server.py",
}

PORT_INFO = {
    "wikipedia": 8003,
    "browser": 8001,
    "python": 8002,
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

    if not server_script.exists():
        print(f"Error: Server script not found at {server_script}")
        sys.exit(1)

    port = PORT_INFO[server_name]
    print(f"Starting {server_name} MCP server on port {port}...")
    print(f"Script: {server_script}")
    print("-" * 50)

    # Run the server
    try:
        subprocess.run([sys.executable, str(server_script)], check=True)
    except KeyboardInterrupt:
        print(f"\n{server_name} server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running {server_name} server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

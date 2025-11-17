#!/usr/bin/env python3
"""Install all MCP servers using uv sync."""
import sys
import subprocess
from pathlib import Path
import importlib.util

# Import centralized MCP server config without triggering package __init__.py
config_path = Path(__file__).parent.parent / "gpt_oss" / "evals" / "mcp_servers_config.py"
spec = importlib.util.spec_from_file_location("mcp_servers_config", config_path)
mcp_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_config)

MCP_SERVERS = mcp_config.MCP_SERVERS


def main():
    """Install all MCP servers."""
    # Get the repo root (parent of scripts directory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    mcp_servers_dir = repo_root / "mcp-servers"

    print("Installing all MCP servers...")
    print("=" * 30)
    print()

    # Get servers from centralized config
    servers = [server.name for server in MCP_SERVERS]

    for server in servers:
        server_dir = mcp_servers_dir / server

        if not server_dir.is_dir():
            print(f"Warning: Server directory not found: {server_dir}")
            print(f"Skipping {server}...")
            print()
            continue

        print(f"Installing {server}...")

        pyproject = server_dir / "pyproject.toml"
        if not pyproject.exists():
            print(f"Warning: No pyproject.toml found in {server_dir}")
            print(f"Skipping {server}...")
            print()
            continue

        try:
            subprocess.run(
                ["uv", "sync"],
                cwd=server_dir,
                check=True,
                capture_output=False
            )
            print(f"✓ {server} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing {server}: {e}")
            continue
        except FileNotFoundError:
            print("Error: 'uv' command not found. Please install uv first.")
            sys.exit(1)

        print()

    print("=" * 30)
    print("Installation complete!")
    print()
    print("To run a server, use:")
    print("  python scripts/run_server.py <server_name>")
    print()
    print("Available servers:")
    for server in MCP_SERVERS:
        print(f"  - {server.short_name} (port {server.port})")


if __name__ == "__main__":
    main()

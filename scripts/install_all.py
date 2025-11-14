#!/usr/bin/env python3
"""Install all MCP servers using uv sync."""
import sys
import subprocess
from pathlib import Path


def main():
    """Install all MCP servers."""
    # Get the repo root (parent of scripts directory)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    mcp_servers_dir = repo_root / "mcp-servers"

    print("Installing all MCP servers...")
    print("=" * 30)
    print()

    # Array of servers to install
    servers = ["mcp-wikipedia", "mcp-browser", "mcp-python", "mcp-google-trends", "mcp-metaculus", "mcp-financial-datasets", "mcp-datacommons"]

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
    print("  - wikipedia (port 8003)")
    print("  - browser (port 8001)")
    print("  - python (port 8002)")
    print("  - google-trends (port 8004)")
    print("  - metaculus (port 8005)")
    print("  - financial-datasets (port 8006)")
    print("  - datacommons (port 8007)")


if __name__ == "__main__":
    main()

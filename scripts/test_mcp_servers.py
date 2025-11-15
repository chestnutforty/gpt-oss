#!/usr/bin/env python3
"""
Test runner for MCP servers.

This script runs pytest for all MCP servers or a specific server.
Each MCP server must have a tests/ directory with test files.

Usage:
    # Test all MCP servers (auto-installs dependencies)
    python scripts/test_mcp_servers.py

    # Test specific server
    python scripts/test_mcp_servers.py --server mcp-browser

    # Test multiple servers
    python scripts/test_mcp_servers.py --server mcp-browser --server mcp-python

    # Run with verbose output
    python scripts/test_mcp_servers.py -v

    # Skip dependency installation (if already installed)
    python scripts/test_mcp_servers.py --no-install

    # Show available servers
    python scripts/test_mcp_servers.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)


# Available MCP servers (excluding mcp-template)
MCP_SERVERS = [
    "mcp-browser",
    "mcp-python",
    "mcp-wikipedia",
    "mcp-financial-datasets",
    "mcp-google-trends",
    "mcp-metaculus",
    "mcp-datacommons",
]


def get_mcp_servers_dir() -> Path:
    """Get the mcp-servers directory path."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    return repo_root / "mcp-servers"


def list_available_servers():
    """List all available MCP servers with test support."""
    print("Available MCP servers for testing:")
    print()
    mcp_dir = get_mcp_servers_dir()

    for server in MCP_SERVERS:
        server_path = mcp_dir / server
        tests_path = server_path / "tests"

        if server_path.exists():
            status = "✓" if tests_path.exists() else "✗ (no tests/)"
            print(f"  {status} {server}")
        else:
            print(f"  ✗ {server} (not found)")


def install_dependencies(server_name: str) -> tuple[bool, str]:
    """
    Install dependencies for a specific MCP server.

    Args:
        server_name: Name of the MCP server

    Returns:
        Tuple of (success: bool, error_message: str)
    """
    mcp_dir = get_mcp_servers_dir()
    server_path = mcp_dir / server_name
    pyproject_path = server_path / "pyproject.toml"

    if not pyproject_path.exists():
        return False, f"No pyproject.toml found in {server_path}"

    print(f"  Installing dependencies for {server_name}...")

    try:
        # Use uv pip install if available, otherwise fall back to pip
        # Try uv first
        result = subprocess.run(
            ["uv", "pip", "install", "-e", ".[test]"],
            cwd=server_path,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return True, ""

        # If uv fails, try regular pip
        result = subprocess.run(
            ["pip", "install", "-e", ".[test]"],
            cwd=server_path,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return True, ""
        else:
            return False, f"pip install failed: {result.stderr[:200]}"

    except FileNotFoundError:
        # Neither uv nor pip found
        return False, "Neither 'uv' nor 'pip' command found"
    except Exception as e:
        return False, f"Error installing dependencies: {str(e)}"


def run_pytest_for_server(server_name: str, verbose: bool = False) -> tuple[bool, str]:
    """
    Run pytest for a specific MCP server.

    Args:
        server_name: Name of the MCP server
        verbose: Whether to run with verbose output

    Returns:
        Tuple of (success: bool, output: str)
    """
    mcp_dir = get_mcp_servers_dir()
    server_path = mcp_dir / server_name
    tests_path = server_path / "tests"

    if not server_path.exists():
        return False, f"Server directory not found: {server_path}"

    if not tests_path.exists():
        return False, f"Tests directory not found: {tests_path}"

    # Build pytest command
    cmd = ["pytest"]
    if verbose:
        cmd.append("-v")
    cmd.append(str(tests_path))

    print(f"\n{'='*70}")
    print(f"Testing: {server_name}")
    print(f"{'='*70}")

    try:
        # Run pytest from the server directory
        result = subprocess.run(
            cmd,
            cwd=server_path,
            capture_output=False,
            text=True
        )

        success = result.returncode == 0
        return success, ""

    except FileNotFoundError:
        return False, "pytest not found. Install with: pip install pytest pytest-asyncio"
    except Exception as e:
        return False, f"Error running pytest: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for MCP servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--server",
        action="append",
        choices=MCP_SERVERS,
        help="Test specific server(s). Can be specified multiple times."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Run pytest with verbose output"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip dependency installation (assumes dependencies are already installed)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available MCP servers"
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_available_servers()
        return 0

    # Determine which servers to test
    servers_to_test = args.server if args.server else MCP_SERVERS

    print(f"MCP Server Test Runner")
    print(f"Testing {len(servers_to_test)} server(s)")
    print()

    # Install dependencies unless --no-install is specified
    if not args.no_install:
        print("Installing dependencies for each server...")
        install_failures = []

        for server in servers_to_test:
            success, error = install_dependencies(server)
            if not success:
                install_failures.append((server, error))
                print(f"  ✗ Failed to install dependencies for {server}: {error}")
            else:
                print(f"  ✓ Dependencies installed for {server}")

        if install_failures:
            print("\n⚠️  Warning: Some dependencies failed to install.")
            print("Tests may fail. You can:")
            print("  1. Manually install dependencies: cd mcp-servers/<server> && uv pip install -e '.[test]'")
            print("  2. Skip this step with --no-install flag")
            print()

    # Track results
    results = {}

    # Run tests for each server
    for server in servers_to_test:
        success, error = run_pytest_for_server(server, args.verbose)
        results[server] = (success, error)

        if error:
            print(f"\n⚠️  Error: {error}")

    # Print summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")

    passed = sum(1 for success, _ in results.values() if success)
    failed = len(results) - passed

    for server, (success, error) in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status:12} {server}")
        if error:
            print(f"               → {error}")

    print()
    print(f"Total: {passed} passed, {failed} failed out of {len(results)} server(s)")

    # Return appropriate exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

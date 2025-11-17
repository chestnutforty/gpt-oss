"""
Centralized configuration for all MCP servers.

This is the single source of truth for MCP server definitions.
Add or remove servers here to update all scripts at once.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str  # Server name (e.g., "mcp-wikipedia")
    short_name: str  # Short name for CLI (e.g., "wikipedia")
    port: int  # Port number for the server
    entry_script: str  # Entry point script relative to server directory
    include_in_tests: bool = True  # Whether to include in test runs


# List of all MCP servers
# To add a new server: append to this list
# To remove a server: remove from this list
MCP_SERVERS = [
    MCPServerConfig(
        name="mcp-browser",
        short_name="browser",
        port=8001,
        entry_script="browser_server.py"
    ),
    MCPServerConfig(
        name="mcp-python",
        short_name="python",
        port=8002,
        entry_script="python_server.py"
    ),
    MCPServerConfig(
        name="mcp-wikipedia",
        short_name="wikipedia",
        port=8003,
        entry_script="wikipedia_server.py"
    ),
    MCPServerConfig(
        name="mcp-google-trends",
        short_name="google-trends",
        port=8004,
        entry_script="google_trends_server.py"
    ),
    MCPServerConfig(
        name="mcp-metaculus",
        short_name="metaculus",
        port=8005,
        entry_script="metaculus_server.py"
    ),
    MCPServerConfig(
        name="mcp-financial-datasets",
        short_name="financial-datasets",
        port=8006,
        entry_script="financial_datasets_server.py"
    ),
    MCPServerConfig(
        name="mcp-datacommons",
        short_name="datacommons",
        port=8007,
        entry_script="server.py"
    ),
    MCPServerConfig(
        name="mcp-eu",
        short_name="eu",
        port=8008,
        entry_script="eu_data_server.py"
    ),
]


def get_server_by_name(short_name: str) -> Optional[MCPServerConfig]:
    """Get server config by short name."""
    for server in MCP_SERVERS:
        if server.short_name == short_name:
            return server
    return None


def get_server_names() -> list[str]:
    """Get list of all server directory names (e.g., 'mcp-wikipedia')."""
    return [server.name for server in MCP_SERVERS]


def get_server_short_names() -> list[str]:
    """Get list of all server short names (e.g., 'wikipedia')."""
    return [server.short_name for server in MCP_SERVERS]


def get_testable_servers() -> list[str]:
    """Get list of server names that should be included in tests."""
    return [server.name for server in MCP_SERVERS if server.include_in_tests]

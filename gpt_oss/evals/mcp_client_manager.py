import asyncio
from typing import Any

from openai.types.responses.tool_choice_types import ToolChoiceTypes

from fastmcp import Client
from fastmcp.client.transports import SSETransport


class MCPClientManager:
    """Manages connections to multiple MCP servers and aggregates their tools."""

    def __init__(self, servers: list[tuple[str, int]]):
        """
        Initialize MCP client manager.

        Args:
            servers: List of (server_name, port) tuples
                    e.g., [("wikipedia", 8003), ("browser", 8001)]
        """
        self.servers = servers
        self.clients: dict[str, Client] = {}
        self.tool_to_server: dict[str, str] = {}
        self._connected = False

    async def connect(self):
        """Connect to all MCP servers and build tool registry."""
        if self._connected:
            return

        for name, port in self.servers:
            url = f"http://localhost:{port}/sse"
            transport = SSETransport(url=url)
            client = Client(transport)

            # Enter context manager
            await client.__aenter__()
            self.clients[name] = client

            # List tools and register them
            tools = await client.list_tools()
            for tool in tools:
                self.tool_to_server[tool.name] = name
        
            print(tools)

        self._connected = True

    async def get_tools_schema(self) -> list[dict]:
        """
        Get OpenAI-compatible tool schemas from all connected servers.

        Returns:
            List of tool schemas in OpenAI function calling format
        """
        if not self._connected:
            raise RuntimeError("Must call connect() before getting tools")

        all_tools = []
        for client in self.clients.values():
            tools = await client.list_tools()
            for tool in tools:
                tool_schema = {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                }
                all_tools.append(tool_schema)

        return all_tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool via the appropriate MCP server.

        Args:
            name: Tool name
            arguments: Tool arguments as dictionary

        Returns:
            Tool result as string

        Raises:
            ValueError: If tool name is not found
        """
        if not self._connected:
            raise RuntimeError("Must call connect() before calling tools")

        server_name = self.tool_to_server.get(name)
        if not server_name:
            raise ValueError(f"Unknown tool: {name}")

        client = self.clients[server_name]
        result = await client.call_tool(name, arguments)

        # Return text result - handle both .data and .content
        if result.data is not None:
            # FastMCP's structured output
            return str(result.data)
        elif result.content:
            # Fall back to content blocks
            return result.content[0].text
        else:
            return ""

    async def close(self):
        """Close all MCP server connections."""
        for client in self.clients.values():
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass  # Ignore errors during cleanup

        self.clients.clear()
        self.tool_to_server.clear()
        self._connected = False

    def __del__(self):
        """Cleanup on garbage collection."""
        if self._connected and self.clients:
            # Try to close connections if event loop exists
            try:
                # Try to get the running loop first (preferred method)
                try:
                    loop = asyncio.get_running_loop()
                    # Loop is running, can't call async in __del__
                    # Just let the connections close naturally
                except RuntimeError:
                    # No running loop, try to get or create one
                    try:
                        loop = asyncio.get_event_loop_policy().get_event_loop()
                        if not loop.is_closed():
                            loop.run_until_complete(self.close())
                    except Exception:
                        pass  # Best effort cleanup
            except Exception:
                pass  # Best effort cleanup

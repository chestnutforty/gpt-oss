#!/usr/bin/env python3
"""Test script for MCP servers running at ports 8001 and 8002."""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client


async def test_server(port: int, server_name: str):
    """Test an MCP server at the given port."""
    print(f"\n{'='*60}")
    print(f"Testing {server_name} server at port {port}")
    print(f"{'='*60}\n")

    url = f"http://localhost:{port}/sse"

    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List available tools
                tools_response = await session.list_tools()
                print(f"✓ Connected to {server_name} server")
                print(f"\nAvailable tools ({len(tools_response.tools)}):")
                for tool in tools_response.tools:
                    print(f"  - {tool.name}: {tool.description[:80]}...")

                # Test tools based on server
                if server_name == "browser":
                    await test_browser_tools(session)
                elif server_name == "python":
                    await test_python_tools(session)

    except Exception as e:
        print(f"✗ Error testing {server_name} server: {e}")
        import traceback
        traceback.print_exc()


async def test_browser_tools(session: ClientSession):
    """Test browser server tools."""
    print("\n" + "-"*60)
    print("Testing browser tools")
    print("-"*60)

    # Test search tool
    print("\n1. Testing 'search' tool with query 'MCP protocol'...")
    try:
        result = await session.call_tool("search", arguments={
            "query": "Outcome of major election in NY",
            "topn": 1,
            "end_crawl_date": "2025-10-05T23:59:59Z"
        })
        print(f"✓ Search completed")
        print(f"Result preview: {str(result.content[0].text)[:1000]}...")
    except Exception as e:
        print(f"✗ Search failed: {e}")

    # Test open tool with a URL
    print("\n2. Testing 'open' tool with example.com...")
    try:
        result = await session.call_tool("open", arguments={
            "id": "https://example.com",
            "num_lines": 20
        })
        print(f"✓ Open completed")
        print(f"Result preview: {str(result.content[0].text)[:200]}...")
    except Exception as e:
        print(f"✗ Open failed: {e}")

    # Test find tool
    print("\n3. Testing 'find' tool...")
    try:
        result = await session.call_tool("find", arguments={
            "pattern": "example"
        })
        print(f"✓ Find completed")
        print(f"Result preview: {str(result.content[0].text)[:200]}...")
    except Exception as e:
        print(f"✗ Find failed: {e}")


async def test_python_tools(session: ClientSession):
    """Test python server tools."""
    print("\n" + "-"*60)
    print("Testing python tools")
    print("-"*60)

    # Test python execution with simple calculation
    print("\n1. Testing 'python' tool with simple calculation...")
    try:
        code = """
print("Testing Python execution:")
result = 2 + 2
print(f"2 + 2 = {result}")
import sys
print(f"Python version: {sys.version.split()[0]}")
"""
        result = await session.call_tool("python", arguments={
            "code": code
        })
        print(f"✓ Python execution completed")
        print(f"Output:\n{result.content[0].text}")
    except Exception as e:
        print(f"✗ Python execution failed: {e}")

    # Test python with imports
    print("\n2. Testing 'python' tool with NumPy...")
    try:
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {arr.mean()}")
print(f"Sum: {arr.sum()}")
"""
        result = await session.call_tool("python", arguments={
            "code": code
        })
        print(f"✓ NumPy test completed")
        print(f"Output:\n{result.content[0].text}")
    except Exception as e:
        print(f"✗ NumPy test failed: {e}")

    # Test python with error handling
    print("\n3. Testing 'python' tool with deliberate error...")
    try:
        code = """
try:
    x = 1 / 0
except ZeroDivisionError as e:
    print(f"Caught error: {e}")
    print("Error handling works correctly!")
"""
        result = await session.call_tool("python", arguments={
            "code": code
        })
        print(f"✓ Error handling test completed")
        print(f"Output:\n{result.content[0].text}")
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")


async def main():
    """Main test function."""
    print("="*60)
    print("MCP Servers Test Suite")
    print("="*60)

    # Test browser server (port 8001)
    await test_server(8001, "browser")

    # Test python server (port 8002)
    await test_server(8002, "python")

    print("\n" + "="*60)
    print("Test suite completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

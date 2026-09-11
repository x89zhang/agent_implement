"""Stateless MCP stdio facade over the runner-owned benchmark session."""

from __future__ import annotations

import asyncio
import json
import os
from urllib.request import Request, urlopen


def request(path: str, value: dict):
    data = json.dumps(value).encode()
    req = Request(
        os.environ["BENCHMARK_BRIDGE_URL"] + path,
        data=data,
        headers={
            "Authorization": "Bearer " + os.environ["BENCHMARK_BRIDGE_TOKEN"],
            "Content-Type": "application/json",
        },
    )
    # No automatic retries: tools may have mutated the environment before a transport error.
    with urlopen(
        req, timeout=float(os.environ.get("BENCHMARK_BRIDGE_TIMEOUT", "300"))
    ) as response:
        return json.load(response)


async def serve():
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

    async def list_tools():
        return [Tool(**tool) for tool in await asyncio.to_thread(request, "/tools", {})]

    async def call_tool(name: str, arguments: dict):
        result = await asyncio.to_thread(
            request, "/call", {"name": name, "arguments": arguments}
        )
        return CallToolResult(
            content=[TextContent(type="text", text=result["text"])],
            isError=result["is_error"],
        )

    if hasattr(Server, "list_tools"):
        # MCP Python SDK 1.x uses decorators.
        server = Server("agent-scaffold-benchmark")
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
    else:
        # SDK 2.x registers typed request handlers in the constructor.
        async def on_list_tools(context, params):
            return ListToolsResult(tools=await list_tools())

        async def on_call_tool(context, params):
            return await call_tool(params.name, params.arguments or {})

        server = Server(
            "agent-scaffold-benchmark",
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(serve())

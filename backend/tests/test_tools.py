"""MCP discovery + call over stdio against tests/mcp_echo_server.py, and an HTTP tool."""

import sys
from pathlib import Path

import httpx
import pytest

from agentforge.tools import build_tools, discover_mcp, make_http_tool, make_mcp_tool

SERVER = f"{sys.executable} {Path(__file__).parent / 'mcp_echo_server.py'}"


@pytest.mark.anyio
async def test_mcp_discover_and_call():
    specs = await discover_mcp(SERVER)
    assert {s["name"] for s in specs} == {"echo", "fail"}
    echo = make_mcp_tool(SERVER, next(s for s in specs if s["name"] == "echo"))
    assert echo.args_schema["properties"]["text"]["type"] == "string"
    out = await echo.ainvoke({"text": "hi"})
    assert "echo: hi" in out and out.startswith("<untrusted-data>")
    fail = make_mcp_tool(SERVER, next(s for s in specs if s["name"] == "fail"))
    with pytest.raises(RuntimeError, match="fail"):  # becomes a ToolMessage inside the graph
        await fail.ainvoke({"reason": "nope"})


@pytest.mark.anyio
async def test_http_tool(monkeypatch):
    seen = {}

    async def fake_request(self, method, url, **kw):
        seen.update(method=method, url=url, params=kw.get("params"))
        return httpx.Response(200, text='{"status":"shipped"}', request=httpx.Request(method, url))

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)
    t = make_http_tool({"name": "order_status", "description": "d", "http_url": "https://x/orders/{order_id}",
                        "parameters_schema": {"type": "object", "properties": {"order_id": {"type": "string"},
                                                                               "verbose": {"type": "boolean"}}}})
    out = await t.ainvoke({"order_id": "ORD-1", "verbose": True})
    assert seen == {"method": "GET", "url": "https://x/orders/ORD-1", "params": {"verbose": True}}
    assert "shipped" in out


def test_build_tools_names():
    names = [t.name for t in build_tools(None, [{"tool_type": "mcp", "target": SERVER, "name": "echo",
                                                   "description": "", "input_schema": {"type": "object"}}])]
    assert names == ["get_current_time", "calculate", "echo"]


@pytest.fixture
def anyio_backend():
    return "asyncio"

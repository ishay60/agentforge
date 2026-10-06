"""Minimal stdio MCP server for tests. Run: python tests/mcp_echo_server.py"""

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("echo")


@mcp.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return f"echo: {text}"


@mcp.tool()
def fail(reason: str) -> str:
    """Always fails. Used to test error propagation."""
    raise ValueError(reason)


if __name__ == "__main__":
    mcp.run()

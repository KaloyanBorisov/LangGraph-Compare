"""
Minimal MCP math server — exposes add, subtract, multiply, divide as tools.
Run standalone (stdio transport) or import from math_agent.py.

    python math_server.py          # start the server directly
    python math_agent.py           # agent launches this automatically via MultiServerMCPClient
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Division by zero")
    return a / b


if __name__ == "__main__":
    mcp.run(transport="stdio")

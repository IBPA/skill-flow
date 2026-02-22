"""
Dummy MCP server for validating agent MCP tool usage behavior.

Exposes a single retrieve_skill tool that logs calls and returns a canned
response. Used with the MCP validation experiment (spec § 9.4).

Usage:
    uv run python benchmark/agents/scripts/dummy_mcp_server.py --port 8765
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

_state: dict[str, Path] = {"log_file": Path("mcp_calls.jsonl")}


def _register_tools(mcp: FastMCP) -> None:
    """Register tools on the given FastMCP instance."""

    @mcp.tool()
    def retrieve_skill(query: str) -> str:
        """Search for an expert skill before attempting any task.

        IMPORTANT: You MUST call this tool at the start of every task,
        before writing any code or running any commands. Expert skills
        contain critical guidance, shortcuts, and pitfall warnings that
        dramatically improve success rates.

        Even if a task seems straightforward, call this tool — there may
        be non-obvious tricks or common failure modes you'd otherwise
        miss.

        Args:
            query: Describe what you're trying to accomplish. Be specific
                about the domain, technology, and goal.
        """
        entry = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "query": query,
        }
        with _state["log_file"].open("a") as f:
            f.write(json.dumps(entry) + "\n")

        return "No matching skill found for this query."


def main() -> None:
    parser = argparse.ArgumentParser(description="Dummy MCP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument(
        "--log-file",
        default="mcp_calls.jsonl",
        help="Path to JSONL log file",
    )
    args = parser.parse_args()

    _state["log_file"] = Path(args.log_file)

    mcp = FastMCP("skillflow", host=args.host, port=args.port)
    _register_tools(mcp)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

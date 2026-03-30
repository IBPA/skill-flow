"""
SkillFlow MCP agent for tool-description validation experiments.

Uses Codex with a single MCP server to test whether tool description alone
makes the agent proactively call retrieve_skill.
"""

from typing import Any

from benchmark.agents.base import BaseCodexAgent, McpServer


class SkillFlowMCPAgent(BaseCodexAgent):
    """
    Codex agent configured with an MCP server for validation testing.

    No instruction template injection — the experiment tests whether
    the MCP tool description alone is sufficient for agent discovery.
    """

    DEFAULT_MCP_URL = "http://host.docker.internal:8765/mcp"

    def __init__(
        self,
        *args: Any,
        mcp_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        mcp_servers = [
            McpServer(
                name="skillflow",
                url=mcp_url or self.DEFAULT_MCP_URL,
            ),
        ]
        super().__init__(*args, mcp_servers=mcp_servers, **kwargs)

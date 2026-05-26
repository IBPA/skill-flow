"""Tests for the cached SkillFlow MCP agent."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.codex_injection_agent import McpServer
from benchmark.agents.skillflow_mcp_cached_agent import (
    SkillFlowMCPCachedAgent,
    _derive_base_url,
)


class TestDeriveBaseUrl:
    """Tests for _derive_base_url helper."""

    def test_strips_mcp_suffix(self) -> None:
        assert _derive_base_url("http://host:8765/mcp") == "http://host:8765"

    def test_no_suffix(self) -> None:
        assert _derive_base_url("http://host:8765") == "http://host:8765"

    def test_ngrok_url(self) -> None:
        url = "https://abc.ngrok-free.dev/mcp"
        assert _derive_base_url(url) == "https://abc.ngrok-free.dev"


class TestSkillFlowMCPCachedAgentInit:
    """Tests for SkillFlowMCPCachedAgent initialization."""

    def test_default_server_base_url(self) -> None:
        agent = create_mock_cached_agent()
        assert agent._server_base_url == "http://host.docker.internal:8765"

    def test_custom_mcp_url(self) -> None:
        agent = create_mock_cached_agent(mcp_url="https://abc.ngrok-free.dev/mcp")
        assert agent._server_base_url == "https://abc.ngrok-free.dev"
        assert agent._mcp_servers[0].url == "https://abc.ngrok-free.dev/mcp"


class TestSkillFlowMCPCachedAgentSetup:
    """Tests for SkillFlowMCPCachedAgent.setup."""

    def test_setup_calls_notify_server_with_keys(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        cache = {"my-task": ["skillsmp/foo", "skillsmp/bar"]}
        agent = create_mock_cached_agent(logs_dir=logs_dir, cache=cache)

        environment = AsyncMock()

        with (
            patch.object(
                SkillFlowMCPCachedAgent.__bases__[0], "setup", new_callable=AsyncMock
            ),
            patch.object(agent, "_notify_server") as mock_notify,
        ):
            asyncio.run(agent.setup(environment))
            mock_notify.assert_called_once_with(
                "my-task", ["skillsmp/foo", "skillsmp/bar"]
            )

    def test_setup_sends_empty_keys_for_unknown_task(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "unknown-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = create_mock_cached_agent(logs_dir=logs_dir, cache={})

        environment = AsyncMock()

        with (
            patch.object(
                SkillFlowMCPCachedAgent.__bases__[0], "setup", new_callable=AsyncMock
            ),
            patch.object(agent, "_notify_server") as mock_notify,
        ):
            asyncio.run(agent.setup(environment))
            mock_notify.assert_called_once_with("unknown-task", [])

    def test_setup_skips_when_no_task_name(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "no-separator" / "agent"
        logs_dir.mkdir(parents=True)
        agent = create_mock_cached_agent(logs_dir=logs_dir)

        environment = AsyncMock()

        with (
            patch.object(
                SkillFlowMCPCachedAgent.__bases__[0], "setup", new_callable=AsyncMock
            ),
            patch.object(agent, "_notify_server") as mock_notify,
        ):
            asyncio.run(agent.setup(environment))
            mock_notify.assert_not_called()


def create_mock_cached_agent(
    logs_dir: Path | None = None,
    mcp_url: str | None = None,
    cache: dict[str, list[str]] | None = None,
) -> SkillFlowMCPCachedAgent:
    """Create a mock SkillFlowMCPCachedAgent for testing."""
    with patch.object(
        SkillFlowMCPCachedAgent.__bases__[0].__bases__[0],
        "__init__",
        lambda self, *args, **kwargs: None,
    ):
        agent = SkillFlowMCPCachedAgent.__new__(SkillFlowMCPCachedAgent)

    effective_url = mcp_url or SkillFlowMCPCachedAgent.DEFAULT_MCP_URL
    agent.logger = MagicMock()
    agent.logs_dir = logs_dir or Path("/tmp/test-logs")
    agent.model_name = "openai/gpt-5-mini"
    agent._OUTPUT_FILENAME = "output.json"
    agent.reasoning_effort = None
    agent._mcp_servers = [McpServer(name="skillflow", url=effective_url)]
    agent._server_base_url = _derive_base_url(effective_url)
    agent._cache = cache or {}
    return agent

"""Tests for evaluation agents base module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmark.agents.base import (
    TASK_NAME_SEPARATOR,
    BaseCodexAgent,
    McpServer,
    get_project_root,
)


class TestConstants:
    """Tests for module constants."""

    def test_task_name_separator(self) -> None:
        """Test that TASK_NAME_SEPARATOR is defined."""
        assert TASK_NAME_SEPARATOR == "__"


class TestGetProjectRoot:
    """Tests for get_project_root function."""

    def test_returns_path(self) -> None:
        """Test that function returns a Path."""
        result = get_project_root()
        assert isinstance(result, Path)

    def test_contains_benchmark_dir(self) -> None:
        """Test that project root contains benchmark directory."""
        root = get_project_root()
        assert (root / "benchmark").exists()


class TestBaseCodexAgentCreateRunAgentCommands:
    """Tests for BaseCodexAgent.create_run_agent_commands."""

    def test_create_commands_with_model(self) -> None:
        """Test creating run commands with model."""
        agent = create_mock_base_agent(model_name="openai/gpt-5-mini")
        commands = agent.create_run_agent_commands("test instruction")

        assert len(commands) == 2
        assert "gpt-5-mini" in commands[1].command

    def test_create_commands_escapes_instruction(self) -> None:
        """Test that instruction is properly escaped."""
        agent = create_mock_base_agent(model_name="openai/gpt-5-mini")
        commands = agent.create_run_agent_commands("test 'with' quotes")

        assert len(commands) == 2

    def test_create_commands_no_model_raises(self) -> None:
        """Test that missing model raises ValueError."""
        agent = create_mock_base_agent(model_name=None)

        with pytest.raises(ValueError, match="Model name is required"):
            agent.create_run_agent_commands("test")

    def test_create_commands_includes_env_vars(self) -> None:
        """Test that commands include environment variables."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            agent = create_mock_base_agent(model_name="openai/gpt-5-mini")
            commands = agent.create_run_agent_commands("test")

            assert commands[0].env["OPENAI_API_KEY"] == "test-key"

    def test_create_commands_contains_codex_exec(self) -> None:
        """Test that second command runs codex exec."""
        agent = create_mock_base_agent(model_name="openai/gpt-5-mini")
        commands = agent.create_run_agent_commands("test")

        assert "codex exec" in commands[1].command
        assert "--dangerously-bypass-approvals-and-sandbox" in commands[1].command

    def test_create_commands_with_reasoning_effort(self) -> None:
        """Test that reasoning_effort is included in command."""
        agent = create_mock_base_agent(
            model_name="openai/gpt-5-mini",
            reasoning_effort="high",
        )
        commands = agent.create_run_agent_commands("test")

        assert '--config model_reasoning_effort="high"' in commands[1].command

    def test_create_commands_without_reasoning_effort(self) -> None:
        """Test that reasoning_effort is omitted when not set."""
        agent = create_mock_base_agent(model_name="openai/gpt-5-mini")
        commands = agent.create_run_agent_commands("test")

        assert "model_reasoning_effort" not in commands[1].command


class TestBaseCodexAgentMcpRegistration:
    """Tests for MCP server registration in create_run_agent_commands."""

    def test_mcp_registration_included_when_servers_set(self) -> None:
        """Test that MCP registration commands are included in setup."""
        agent = create_mock_base_agent(
            model_name="openai/gpt-5-mini",
            mcp_servers=[McpServer(name="skillflow", url="http://host:8765/sse")],
        )
        commands = agent.create_run_agent_commands("test")

        expected = "codex mcp add skillflow --url http://host:8765/sse"
        assert expected in commands[0].command

    def test_no_mcp_registration_when_no_servers(self) -> None:
        """Test that no MCP commands appear when no servers configured."""
        agent = create_mock_base_agent(model_name="openai/gpt-5-mini")
        commands = agent.create_run_agent_commands("test")

        assert "codex mcp add" not in commands[0].command


def create_mock_base_agent(
    logs_dir: Path | None = None,
    model_name: str | None = "openai/gpt-5-mini",
    reasoning_effort: str | None = None,
    mcp_servers: list[McpServer] | None = None,
) -> BaseCodexAgent:
    """Create a mock BaseCodexAgent for testing."""
    with patch.object(BaseCodexAgent, "__init__", lambda self, *args, **kwargs: None):
        agent = BaseCodexAgent.__new__(BaseCodexAgent)

    agent.logger = MagicMock()
    agent.logs_dir = logs_dir or Path("/tmp/test-logs")
    agent.model_name = model_name
    agent.reasoning_effort = reasoning_effort
    agent._mcp_servers = mcp_servers or []
    agent._OUTPUT_FILENAME = "output.json"
    return agent

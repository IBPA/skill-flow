"""Tests for evaluation agents."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.base import McpServer, get_project_root
from benchmark.agents.codex_with_skillflow import CodexWithSkillFlow
from benchmark.agents.mcp_test_agent import McpTestAgent
from benchmark.agents.skill_agent import SkillAgent
from benchmark.agents.skills import SkillManager, TarGzSkillInjector


class TestCodexWithSkillFlowInit:
    """Tests for CodexWithSkillFlow initialization."""

    def test_default_peer_url(self) -> None:
        """Test default peer URL is set."""
        agent = create_mock_skillflow_agent()
        assert agent._skillflow_peer_url == "http://172.17.0.1:8765"

    def test_custom_peer_url(self) -> None:
        """Test custom peer URL is set."""
        agent = create_mock_skillflow_agent(skillflow_peer_url="http://localhost:9000")
        assert agent._skillflow_peer_url == "http://localhost:9000"


class TestCodexWithSkillFlowSetup:
    """Tests for CodexWithSkillFlow.setup."""

    def test_setup_calls_parent_setup(self, tmp_path: Path) -> None:
        """Test that setup calls parent setup."""
        agent = create_mock_skillflow_agent(logs_dir=tmp_path)

        environment = AsyncMock()
        environment.exec = AsyncMock()

        with patch.object(
            CodexWithSkillFlow.__bases__[0], "setup", new_callable=AsyncMock
        ) as mock_super_setup:
            asyncio.run(agent.setup(environment))
            mock_super_setup.assert_called_once_with(environment)

    def test_setup_uploads_client_script(self, tmp_path: Path) -> None:
        """Test that setup uploads skillflow-client script."""
        agent = create_mock_skillflow_agent(logs_dir=tmp_path)

        environment = AsyncMock()
        environment.exec = AsyncMock()

        with patch.object(
            CodexWithSkillFlow.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(environment))

            calls = environment.exec.call_args_list
            assert any("skillflow-client" in str(call) for call in calls)


class TestSkillAgentInit:
    """Tests for SkillAgent initialization."""

    def test_default_source_dir(self) -> None:
        """Test default source directory."""
        agent = create_mock_skill_agent()
        assert "outputs/skills/downloaded" in str(agent._skill_manager._source_dir)

    def test_custom_source_dir(self, tmp_path: Path) -> None:
        """Test custom source directory."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        agent = create_mock_skill_agent(skills_source_dir=str(skills_dir))
        assert agent._skill_manager._source_dir == skills_dir

    def test_skills_list_file(self, tmp_path: Path) -> None:
        """Test skills list file is set."""
        list_file = tmp_path / "skills.txt"
        list_file.write_text("skill1\n")
        agent = create_mock_skill_agent(skills_list_file=str(list_file))
        assert agent._skill_manager._skills_list_file == list_file

    def test_match_skill_to_task(self) -> None:
        """Test match_skill_to_task is set."""
        agent = create_mock_skill_agent(match_skill_to_task=True)
        assert agent._skill_manager._match_skill_to_task is True


class TestSkillAgentSetup:
    """Tests for SkillAgent.setup."""

    def test_setup_calls_parent_setup(self, tmp_path: Path) -> None:
        """Test that setup calls parent setup."""
        agent = create_mock_skill_agent(logs_dir=tmp_path)

        environment = AsyncMock()
        environment.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        environment.upload_file = AsyncMock()

        with patch.object(
            SkillAgent.__bases__[0], "setup", new_callable=AsyncMock
        ) as mock_super_setup:
            asyncio.run(agent.setup(environment))
            mock_super_setup.assert_called_once_with(environment)


class TestSkillAgentExtractTaskName:
    """Tests for SkillAgent._extract_task_name."""

    def test_extract_from_trial_dir(self, tmp_path: Path) -> None:
        """Test extracting task name from trial directory."""
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)

        agent = create_mock_skill_agent(logs_dir=logs_dir)
        task_name = agent._extract_task_name()

        assert task_name == "my-task"

    def test_extract_no_separator(self, tmp_path: Path) -> None:
        """Test extracting when no separator in path."""
        logs_dir = tmp_path / "no-separator" / "agent"
        logs_dir.mkdir(parents=True)

        agent = create_mock_skill_agent(logs_dir=logs_dir)
        task_name = agent._extract_task_name()

        assert task_name is None


class TestMcpTestAgentInit:
    """Tests for McpTestAgent initialization."""

    def test_default_mcp_url(self) -> None:
        """Test default MCP URL is set."""
        agent = create_mock_mcp_agent()
        assert len(agent._mcp_servers) == 1
        assert agent._mcp_servers[0].name == "skillflow"
        assert agent._mcp_servers[0].url == "http://host.docker.internal:8765/mcp"

    def test_custom_mcp_url(self) -> None:
        """Test custom MCP URL is set."""
        agent = create_mock_mcp_agent(mcp_url="http://localhost:9000/sse")
        assert len(agent._mcp_servers) == 1
        assert agent._mcp_servers[0].url == "http://localhost:9000/sse"


def create_mock_mcp_agent(
    logs_dir: Path | None = None,
    mcp_url: str | None = None,
) -> McpTestAgent:
    """Create a mock McpTestAgent for testing."""
    with patch.object(
        McpTestAgent.__bases__[0], "__init__", lambda self, *args, **kwargs: None
    ):
        agent = McpTestAgent.__new__(McpTestAgent)

    agent.logger = MagicMock()
    agent.logs_dir = logs_dir or Path("/tmp/test-logs")
    agent.model_name = "openai/gpt-5-mini"
    agent._OUTPUT_FILENAME = "output.json"
    agent.reasoning_effort = None
    agent._mcp_servers = [
        McpServer(
            name="skillflow",
            url=mcp_url or McpTestAgent.DEFAULT_MCP_URL,
        ),
    ]
    return agent


def create_mock_skillflow_agent(
    logs_dir: Path | None = None,
    skillflow_peer_url: str | None = None,
) -> CodexWithSkillFlow:
    """Create a mock CodexWithSkillFlow for testing."""
    with patch.object(
        CodexWithSkillFlow.__bases__[0], "__init__", lambda self, *args, **kwargs: None
    ):
        agent = CodexWithSkillFlow.__new__(CodexWithSkillFlow)

    agent.logger = MagicMock()
    agent.logs_dir = logs_dir or Path("/tmp/test-logs")
    agent.model_name = "openai/gpt-5-mini"
    agent._OUTPUT_FILENAME = "output.json"
    agent._skillflow_peer_url = (
        skillflow_peer_url or CodexWithSkillFlow.DEFAULT_PEER_URL
    )
    return agent


def create_mock_skill_agent(
    logs_dir: Path | None = None,
    skills_source_dir: str | None = None,
    skills_list_file: str | None = None,
    match_skill_to_task: bool = False,
) -> SkillAgent:
    """Create a mock SkillAgent for testing."""
    with patch.object(
        SkillAgent.__bases__[0], "__init__", lambda self, *args, **kwargs: None
    ):
        agent = SkillAgent.__new__(SkillAgent)

    agent.logger = MagicMock()
    agent.logs_dir = logs_dir or Path("/tmp/test-logs/trial__hash/agent")
    agent.model_name = "openai/gpt-5-mini"
    agent._OUTPUT_FILENAME = "output.json"

    if skills_source_dir:
        source_dir = Path(skills_source_dir)
    else:
        source_dir = get_project_root() / "outputs" / "skills" / "downloaded"

    list_file = Path(skills_list_file) if skills_list_file else None

    agent._skill_manager = SkillManager(
        source_dir=source_dir,
        skills_list_file=list_file,
        match_skill_to_task=match_skill_to_task,
        logger=agent.logger,
    )
    agent._skill_injector = TarGzSkillInjector(logger=agent.logger)
    return agent

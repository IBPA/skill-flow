"""Tests for evaluation agents."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.base import McpServer
from benchmark.agents.skillflow_injection_agent import SkillFlowInjectionAgent
from benchmark.agents.skillflow_mcp_agent import SkillFlowMCPAgent
from benchmark.agents.skillflow_mcp_cached_agent import (
    SkillFlowMCPCachedAgent,
    _derive_base_url,
)
from benchmark.agents.skills import SkillManager, TarGzSkillInjector
from mcp_servers.utils.skill_loader import ResolvedSkill


class TestSkillFlowInjectionAgentInit:
    """Tests for SkillFlowInjectionAgent initialization."""

    def test_with_eval_results(self) -> None:
        agent = create_mock_injection_agent(eval_results="/path/eval.json")
        assert agent._eval_results == Path("/path/eval.json")
        assert agent._selector_cache is None

    def test_with_selector_cache(self) -> None:
        agent = create_mock_injection_agent(selector_cache="/path/cache.json")
        assert agent._selector_cache == Path("/path/cache.json")
        assert agent._eval_results is None


class TestSkillFlowInjectionAgentSetup:
    """Tests for SkillFlowInjectionAgent.setup."""

    def test_setup_resolves_from_eval_results(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        skill_dir = tmp_path / "s1"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("x")
        agent = create_mock_injection_agent(
            logs_dir=logs_dir, eval_results="/eval.json", tasks_dir="/tasks"
        )
        env = AsyncMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        env.upload_file = AsyncMock()
        mock_resolved = [ResolvedSkill("s1", "", skill_dir)]
        with (
            patch.object(
                SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
            ),
            patch(
                "benchmark.agents.skillflow_injection_agent.resolve_eval_skill_folders",
                return_value=mock_resolved,
            ),
        ):
            asyncio.run(agent.setup(env))
        assert env.exec.call_count > 0

    def test_setup_resolves_from_selector_cache(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps({"my-task": ["skillsbench/t/s1"]}))
        tasks_dir = tmp_path / "tasks"
        skill_path = tasks_dir / "t" / "environment" / "skills" / "s1"
        skill_path.mkdir(parents=True)
        (skill_path / "SKILL.md").write_text("x")
        agent = create_mock_injection_agent(
            logs_dir=logs_dir,
            selector_cache=str(cache_file),
            tasks_dir=str(tasks_dir),
        )
        env = AsyncMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        env.upload_file = AsyncMock()
        with patch.object(
            SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(env))
        assert env.exec.call_count > 0

    def test_setup_skips_when_no_task_name(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "no-separator" / "agent"
        logs_dir.mkdir(parents=True)
        agent = create_mock_injection_agent(
            logs_dir=logs_dir, eval_results="/eval.json"
        )
        env = AsyncMock()
        with patch.object(
            SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(env))
        agent.logger.warning.assert_called()


class TestSkillFlowInjectionAgentSkillsDir:
    """Tests for SkillFlowInjectionAgent with skills_dir source."""

    def test_skills_dir_sets_manager(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        agent = create_mock_injection_agent(skills_dir=str(skills_dir))
        assert agent._skill_manager is not None
        assert agent._skill_manager._source_dir == skills_dir

    def test_skills_list_file(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        list_file = tmp_path / "skills.txt"
        list_file.write_text("skill1\n")
        agent = create_mock_injection_agent(
            skills_dir=str(skills_dir), skills_list_file=str(list_file)
        )
        assert agent._skill_manager is not None
        assert agent._skill_manager._skills_list_file == list_file

    def test_match_skill_to_task(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        agent = create_mock_injection_agent(
            skills_dir=str(skills_dir), match_skill_to_task=True
        )
        assert agent._skill_manager is not None
        assert agent._skill_manager._match_skill_to_task is True

    def test_no_source_means_no_manager(self) -> None:
        agent = create_mock_injection_agent()
        assert agent._skill_manager is None

    def test_setup_with_skills_dir(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills" / "my-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("x")
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = create_mock_injection_agent(
            logs_dir=logs_dir, skills_dir=str(tmp_path / "skills")
        )
        env = AsyncMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        env.upload_file = AsyncMock()
        with patch.object(
            SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(env))
        assert env.exec.call_count > 0


class TestSkillFlowMCPAgentInit:
    """Tests for SkillFlowMCPAgent initialization."""

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
) -> SkillFlowMCPAgent:
    """Create a mock SkillFlowMCPAgent for testing."""
    with patch.object(
        SkillFlowMCPAgent.__bases__[0], "__init__", lambda self, *args, **kwargs: None
    ):
        agent = SkillFlowMCPAgent.__new__(SkillFlowMCPAgent)

    agent.logger = MagicMock()
    agent.logs_dir = logs_dir or Path("/tmp/test-logs")
    agent.model_name = "openai/gpt-5-mini"
    agent._OUTPUT_FILENAME = "output.json"
    agent.reasoning_effort = None
    agent._mcp_servers = [
        McpServer(
            name="skillflow",
            url=mcp_url or SkillFlowMCPAgent.DEFAULT_MCP_URL,
        ),
    ]
    return agent


def create_mock_injection_agent(
    logs_dir: Path | None = None,
    skills_dir: str | None = None,
    skills_list_file: str | None = None,
    match_skill_to_task: bool = False,
    eval_results: str | None = None,
    selector_cache: str | None = None,
    tasks_dir: str | None = None,
    corpus_dir: str | None = None,
) -> SkillFlowInjectionAgent:
    """Create a mock SkillFlowInjectionAgent for testing."""
    with patch.object(
        SkillFlowInjectionAgent.__bases__[0],
        "__init__",
        lambda self, *args, **kwargs: None,
    ):
        agent = SkillFlowInjectionAgent.__new__(SkillFlowInjectionAgent)

    agent.logger = MagicMock()
    agent.logs_dir = logs_dir or Path("/tmp/test-logs/trial__hash/agent")
    agent.model_name = "openai/gpt-5-mini"
    agent._OUTPUT_FILENAME = "output.json"
    agent.reasoning_effort = None
    agent._mcp_servers = []
    agent._eval_results = Path(eval_results) if eval_results else None
    agent._selector_cache = Path(selector_cache) if selector_cache else None
    agent._tasks_dir = Path(tasks_dir) if tasks_dir else None
    agent._corpus_dir = Path(corpus_dir) if corpus_dir else None
    agent._injector = TarGzSkillInjector(logger=agent.logger)

    if skills_dir:
        list_file = Path(skills_list_file) if skills_list_file else None
        agent._skill_manager = SkillManager(
            source_dir=Path(skills_dir),
            skills_list_file=list_file,
            match_skill_to_task=match_skill_to_task,
            logger=agent.logger,
        )
    else:
        agent._skill_manager = None
    return agent


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

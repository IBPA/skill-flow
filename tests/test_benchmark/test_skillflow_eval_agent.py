"""Tests for SkillFlowEvalAgent."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.base import McpServer
from benchmark.agents.skillflow_eval_agent import SkillFlowEvalAgent
from benchmark.agents.skills.injector import TarGzSkillInjector


def _create_mock_agent(
    logs_dir: Path | None = None,
    eval_results: str = "/tmp/results.json",
    tasks_dir: str = "/tmp/tasks",
    mcp_url: str | None = None,
    corpus_dir: str | None = None,
) -> SkillFlowEvalAgent:
    """Create a mock SkillFlowEvalAgent for testing."""
    with patch.object(
        SkillFlowEvalAgent.__bases__[0],
        "__init__",
        lambda self, *args, **kwargs: None,
    ):
        agent = SkillFlowEvalAgent.__new__(SkillFlowEvalAgent)

    agent.logger = MagicMock()
    agent.logs_dir = logs_dir or Path("/tmp/test-logs/trial__hash/agent")
    agent.model_name = "openai/gpt-5-mini"
    agent._OUTPUT_FILENAME = "output.json"
    agent.reasoning_effort = None
    agent._mcp_servers = [
        McpServer(
            name="skillflow",
            url=mcp_url or SkillFlowEvalAgent.DEFAULT_MCP_URL,
        ),
    ]
    agent._eval_results = Path(eval_results)
    agent._tasks_dir = Path(tasks_dir)
    agent._corpus_dir = Path(corpus_dir) if corpus_dir else None
    agent._injector = TarGzSkillInjector(logger=agent.logger)
    return agent


def _write_eval_results(
    path: Path,
    task_results: list[dict[str, object]],
) -> None:
    """Write a minimal eval results JSON file."""
    data = {"summary": {}, "task_results": task_results}
    path.write_text(json.dumps(data))


class TestSkillFlowEvalAgentInit:
    """Tests for SkillFlowEvalAgent initialization."""

    def test_default_mcp_url(self) -> None:
        agent = _create_mock_agent()
        assert len(agent._mcp_servers) == 1
        assert agent._mcp_servers[0].name == "skillflow"
        assert agent._mcp_servers[0].url == SkillFlowEvalAgent.DEFAULT_MCP_URL

    def test_custom_mcp_url(self) -> None:
        agent = _create_mock_agent(mcp_url="http://localhost:9000/mcp")
        assert agent._mcp_servers[0].url == "http://localhost:9000/mcp"

    def test_eval_results_stored(self) -> None:
        agent = _create_mock_agent(eval_results="/data/results.json")
        assert agent._eval_results == Path("/data/results.json")

    def test_tasks_dir_stored(self) -> None:
        agent = _create_mock_agent(tasks_dir="/data/tasks")
        assert agent._tasks_dir == Path("/data/tasks")


class TestSkillFlowEvalAgentExtractTaskName:
    """Tests for task name extraction."""

    def test_extract_from_trial_dir(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(logs_dir=logs_dir)
        assert agent._extract_task_name() == "my-task"

    def test_extract_no_separator(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "no-separator" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(logs_dir=logs_dir)
        assert agent._extract_task_name() is None


class TestSkillFlowEvalAgentSetup:
    """Tests for SkillFlowEvalAgent.setup."""

    def test_setup_calls_parent_setup(self, tmp_path: Path) -> None:
        agent = _create_mock_agent(logs_dir=tmp_path)
        environment = AsyncMock()

        with patch.object(
            SkillFlowEvalAgent.__bases__[0], "setup", new_callable=AsyncMock
        ) as mock_super:
            asyncio.run(agent.setup(environment))
            mock_super.assert_called_once_with(environment)

    def test_setup_injects_resolved_skills(self, tmp_path: Path) -> None:
        # Create task directory with skill folder
        tasks_dir = tmp_path / "tasks"
        skill_dir = tasks_dir / "my-task" / "environment" / "skills" / "mesh-analysis"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: mesh-analysis\n---\n# Content")

        # Write eval results
        eval_path = tmp_path / "results.json"
        _write_eval_results(
            eval_path,
            [
                {
                    "task_id": "my-task",
                    "retrieved_skills": [
                        {
                            "key": "skillsbench/my-task/mesh-analysis",
                            "description": "Analyzes meshes",
                        }
                    ],
                }
            ],
        )

        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(
            logs_dir=logs_dir,
            eval_results=str(eval_path),
            tasks_dir=str(tasks_dir),
        )

        environment = AsyncMock()
        environment.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        environment.upload_file = AsyncMock()

        with patch.object(
            SkillFlowEvalAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(environment))

        # Verify injector was called (mkdir + tar extract + ls verify)
        assert environment.exec.call_count >= 1

    def test_setup_skips_when_no_task_name(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "no-separator" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(logs_dir=logs_dir)

        environment = AsyncMock()

        with patch.object(
            SkillFlowEvalAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(environment))

        agent.logger.warning.assert_any_call(
            "Could not extract task name, skipping skill injection"
        )

    def test_setup_skips_when_no_skills_resolved(self, tmp_path: Path) -> None:
        eval_path = tmp_path / "results.json"
        _write_eval_results(
            eval_path,
            [{"task_id": "other-task", "retrieved_skills": []}],
        )

        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(
            logs_dir=logs_dir,
            eval_results=str(eval_path),
            tasks_dir=str(tmp_path / "tasks"),
        )

        environment = AsyncMock()

        with patch.object(
            SkillFlowEvalAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(environment))

        agent.logger.warning.assert_any_call(
            "No skills resolved for task '%s'", "my-task"
        )

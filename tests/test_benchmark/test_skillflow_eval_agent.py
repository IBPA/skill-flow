"""Tests for SkillFlowInjectionAgent."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.skillflow_injection_agent import SkillFlowInjectionAgent
from benchmark.agents.skills.injector import TarGzSkillInjector


def _create_mock_agent(
    logs_dir: Path | None = None,
    eval_results: str | None = None,
    selector_cache: str | None = None,
    tasks_dir: str = "/tmp/tasks",
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
    agent._tasks_dir = Path(tasks_dir)
    agent._corpus_dir = Path(corpus_dir) if corpus_dir else None
    agent._skill_manager = None
    agent._injector = TarGzSkillInjector(logger=agent.logger)
    return agent


def _write_eval_results(
    path: Path,
    task_results: list[dict[str, object]],
) -> None:
    """Write a minimal eval results JSON file."""
    data = {"summary": {}, "task_results": task_results}
    path.write_text(json.dumps(data))


class TestSkillFlowInjectionAgentInit:
    """Tests for SkillFlowInjectionAgent initialization."""

    def test_eval_results_stored(self) -> None:
        agent = _create_mock_agent(eval_results="/data/results.json")
        assert agent._eval_results == Path("/data/results.json")
        assert agent._selector_cache is None

    def test_selector_cache_stored(self) -> None:
        agent = _create_mock_agent(selector_cache="/data/cache.json")
        assert agent._selector_cache == Path("/data/cache.json")
        assert agent._eval_results is None

    def test_tasks_dir_stored(self) -> None:
        agent = _create_mock_agent(
            eval_results="/data/results.json", tasks_dir="/data/tasks"
        )
        assert agent._tasks_dir == Path("/data/tasks")


class TestSkillFlowInjectionAgentExtractTaskName:
    """Tests for task name extraction."""

    def test_extract_from_trial_dir(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(logs_dir=logs_dir, eval_results="/r.json")
        assert agent._extract_task_name() == "my-task"

    def test_extract_no_separator(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "no-separator" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(logs_dir=logs_dir, eval_results="/r.json")
        assert agent._extract_task_name() is None


class TestSkillFlowInjectionAgentSetup:
    """Tests for SkillFlowInjectionAgent.setup."""

    def test_setup_calls_parent_setup(self, tmp_path: Path) -> None:
        agent = _create_mock_agent(logs_dir=tmp_path, eval_results="/r.json")
        environment = AsyncMock()

        with patch.object(
            SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
        ) as mock_super:
            asyncio.run(agent.setup(environment))
            mock_super.assert_called_once_with(environment)

    def test_setup_injects_from_eval_results(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        skill_dir = tasks_dir / "my-task" / "environment" / "skills" / "mesh-analysis"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: mesh-analysis\n---\n# Content")

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
            SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(environment))

        assert environment.exec.call_count >= 1

    def test_setup_injects_from_selector_cache(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        skill_dir = tasks_dir / "src-task" / "environment" / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\n# Content")

        cache_path = tmp_path / "cache.json"
        cache_path.write_text(
            json.dumps({"my-task": ["skillsbench/src-task/my-skill"]})
        )

        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(
            logs_dir=logs_dir,
            selector_cache=str(cache_path),
            tasks_dir=str(tasks_dir),
        )

        environment = AsyncMock()
        environment.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        environment.upload_file = AsyncMock()

        with patch.object(
            SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(environment))

        assert environment.exec.call_count >= 1

    def test_setup_skips_when_no_task_name(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "no-separator" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _create_mock_agent(logs_dir=logs_dir, eval_results="/r.json")

        environment = AsyncMock()

        with patch.object(
            SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(environment))

        agent.logger.warning.assert_any_call(
            "Could not extract task name, skipping injection"
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
            SkillFlowInjectionAgent.__bases__[0], "setup", new_callable=AsyncMock
        ):
            asyncio.run(agent.setup(environment))

        agent.logger.warning.assert_any_call(
            "No skills resolved for task '%s'", "my-task"
        )

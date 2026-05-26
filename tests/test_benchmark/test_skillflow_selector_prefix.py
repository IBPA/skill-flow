"""Tests for selector-cache task ID matching in skill injection."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from benchmark.agents.codex_injection_agent import SkillFlowCodexAgent
from benchmark.agents.skills import TarGzSkillInjector


def _make_agent(
    selector_cache: Path, tasks_dir: Path, logs_dir: Path
) -> SkillFlowCodexAgent:
    """Build a lightweight SkillFlowCodexAgent for resolver testing."""
    agent = SkillFlowCodexAgent.__new__(SkillFlowCodexAgent)
    agent.logger = MagicMock()
    agent.logs_dir = logs_dir
    agent._eval_results = None
    agent._selector_cache = selector_cache
    agent._tasks_dir = tasks_dir
    agent._corpus_dir = None
    agent._injector = TarGzSkillInjector(logger=agent.logger)
    agent._skill_manager = None
    return agent


def test_selector_cache_resolves_unique_truncated_task_prefix(tmp_path: Path) -> None:
    """Selector-cache injection handles Harbor-shortened trial names."""
    tasks_dir = tmp_path / "tasks"
    skill_dir = tasks_dir / "source-task" / "environment" / "skills" / "skill-one"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Skill")
    selector_cache = tmp_path / "cache.json"
    selector_cache.write_text(
        json.dumps(
            {"task-alpha-with-long-suffix": ["skillsbench/source-task/skill-one"]}
        )
    )
    logs_dir = tmp_path / "task-alpha-with-long__abc123" / "agent"
    logs_dir.mkdir(parents=True)
    agent = _make_agent(selector_cache, tasks_dir, logs_dir)

    folders = agent._resolve_from_selector_cache("task-alpha-with-long")

    assert folders == [skill_dir]


def test_selector_cache_rejects_ambiguous_truncated_task_prefix(
    tmp_path: Path,
) -> None:
    """Selector-cache injection does not guess between multiple full task IDs."""
    tasks_dir = tmp_path / "tasks"
    selector_cache = tmp_path / "cache.json"
    selector_cache.write_text(
        json.dumps(
            {
                "task-alpha-long-one": ["skillsbench/source-task/skill-one"],
                "task-alpha-long-two": ["skillsbench/source-task/skill-two"],
            }
        )
    )
    logs_dir = tmp_path / "task-alpha-long__abc123" / "agent"
    logs_dir.mkdir(parents=True)
    agent = _make_agent(selector_cache, tasks_dir, logs_dir)

    folders = agent._resolve_from_selector_cache("task-alpha-long")

    assert folders == []

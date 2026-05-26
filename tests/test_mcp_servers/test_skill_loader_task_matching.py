"""Tests for task ID matching in skill resolution."""

import json
from pathlib import Path

from mcp_servers.utils.skill_loader import (
    find_matching_task_id,
    resolve_eval_skill_folders,
)


def _write_eval_results(path: Path, task_results: list[dict[str, object]]) -> None:
    """Write a minimal eval results JSON file."""
    path.write_text(json.dumps({"summary": {}, "task_results": task_results}))


def test_find_matching_task_id_prefers_exact_match() -> None:
    """Exact task IDs are used before any prefix fallback."""
    task_ids = ["task-alpha", "task-alpha-long"]

    result = find_matching_task_id(task_ids, "task-alpha")

    assert result == "task-alpha"


def test_find_matching_task_id_resolves_unique_prefix() -> None:
    """Shortened trial-dir task names resolve to a unique full task ID."""
    task_ids = ["task-alpha-with-long-suffix", "task-beta"]

    result = find_matching_task_id(task_ids, "task-alpha-with-long")

    assert result == "task-alpha-with-long-suffix"


def test_find_matching_task_id_rejects_ambiguous_prefix() -> None:
    """Ambiguous shortened task names do not choose arbitrarily."""
    task_ids = ["task-alpha-long-one", "task-alpha-long-two"]

    result = find_matching_task_id(task_ids, "task-alpha-long")

    assert result is None


def test_eval_resolution_uses_unique_truncated_task_prefix(tmp_path: Path) -> None:
    """Eval-results skill resolution handles Harbor-shortened trial names."""
    tasks_dir = tmp_path / "tasks"
    skill_dir = tasks_dir / "task-alpha" / "environment" / "skills" / "mesh-analysis"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: mesh-analysis\n---\n# Content")
    eval_path = tmp_path / "results.json"
    _write_eval_results(
        eval_path,
        [
            {
                "task_id": "task-alpha-with-long-suffix",
                "retrieved_skills": [
                    {
                        "key": "skillsbench/task-alpha/mesh-analysis",
                        "description": "Analyzes 3D meshes",
                    }
                ],
            }
        ],
    )

    resolved = resolve_eval_skill_folders(eval_path, tasks_dir, "task-alpha-with-long")

    assert len(resolved) == 1
    assert resolved[0].name == "mesh-analysis"

"""Tests for src.eval.ground_truth."""

import json
from pathlib import Path

from skill_flow.eval.utils.ground_truth import (
    _parse_frontmatter,
    load_corpus_keys,
    load_ground_truth,
)


def _make_skill_dir(task_dir: Path, skill_name: str, desc: str) -> None:
    """Create a skill directory with a SKILL.md containing frontmatter."""
    skill_dir = task_dir / "environment" / "skills" / skill_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f'---\nname: {skill_name}\ndescription: "{desc}"\n---\n# {skill_name}\n'
    (skill_dir / "SKILL.md").write_text(content)


def _make_task(
    tasks_dir: Path,
    task_id: str,
    instruction: str,
    skill_names: list[str] | None = None,
) -> Path:
    """Create a task directory with instruction and optional skills."""
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(instruction)
    if skill_names:
        for name in skill_names:
            _make_skill_dir(task_dir, name, f"Description for {name}")
    return task_dir


class TestLoadCorpusKeys:
    def test_loads_keys(self, tmp_path: Path):
        keys = ["skillsmp/foo", "skillsmp/bar", "skillsmp/baz"]
        (tmp_path / "skill_ids.json").write_text(json.dumps(keys))
        result = load_corpus_keys(tmp_path)
        assert result == {"skillsmp/foo", "skillsmp/bar", "skillsmp/baz"}


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        content = '---\nname: test-skill\ndescription: "A test"\n---\n'
        name, desc = _parse_frontmatter(content)
        assert name == "test-skill"
        assert desc == "A test"

    def test_no_frontmatter(self):
        name, desc = _parse_frontmatter("# Just a heading")
        assert name == ""
        assert desc == ""

    def test_missing_end_delimiter(self):
        name, desc = _parse_frontmatter("---\nname: x\n")
        assert name == ""
        assert desc == ""

    def test_invalid_yaml(self):
        name, desc = _parse_frontmatter("---\n: :\n---\n")
        assert name == ""
        assert desc == ""

    def test_non_dict_yaml(self):
        content = "---\n- item1\n- item2\n---\n"
        name, desc = _parse_frontmatter(content)
        assert name == ""
        assert desc == ""


class TestLoadGroundTruth:
    def test_task_scoped_keys(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        _make_task(tasks_dir, "task-1", "Do something", ["my-skill"])

        evaluable, injected, _skipped = load_ground_truth(tasks_dir)
        assert len(evaluable) == 1
        assert evaluable[0].ground_truth_keys == ["skillsbench/task-1/my-skill"]
        assert evaluable[0].injected_skills == ["skillsbench/task-1/my-skill"]
        assert len(injected) == 1
        assert injected[0].key == "skillsbench/task-1/my-skill"
        assert injected[0].name == "my-skill"
        assert injected[0].content.startswith("---")

    def test_multiple_skills(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        _make_task(tasks_dir, "task-1", "Do it", ["alpha", "beta"])

        evaluable, injected, _skipped = load_ground_truth(tasks_dir)
        assert len(evaluable) == 1
        gt = evaluable[0]
        assert "skillsbench/task-1/alpha" in gt.ground_truth_keys
        assert "skillsbench/task-1/beta" in gt.ground_truth_keys
        assert len(injected) == 2

    def test_same_name_different_tasks_separate_keys(self, tmp_path: Path):
        """Same skill name in two tasks gets separate task-scoped keys."""
        tasks_dir = tmp_path / "tasks"
        _make_task(tasks_dir, "task-1", "Query 1", ["pdf"])
        _make_task(tasks_dir, "task-2", "Query 2", ["pdf"])

        evaluable, injected, _skipped = load_ground_truth(tasks_dir)
        assert len(evaluable) == 2
        assert len(injected) == 2
        keys = {s.key for s in injected}
        assert "skillsbench/task-1/pdf" in keys
        assert "skillsbench/task-2/pdf" in keys

    def test_no_skills_dir_skipped(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "task-no-skills"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Do something")

        evaluable, _injected, skipped = load_ground_truth(tasks_dir)
        assert len(evaluable) == 0
        assert len(skipped) == 1
        assert skipped[0] == ("task-no-skills", "no skills directory")

    def test_no_instruction_skipped(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        (tasks_dir / "task-no-inst").mkdir(parents=True)

        _evaluable, _injected, skipped = load_ground_truth(tasks_dir)
        assert len(skipped) == 1
        assert skipped[0] == ("task-no-inst", "no instruction.md")

    def test_dir_without_skill_md_skipped(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "task-1"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Do it")
        licenses_dir = task_dir / "environment" / "skills" / "licenses"
        licenses_dir.mkdir(parents=True)
        (licenses_dir / "some.LICENSE").write_text("MIT")

        evaluable, _injected, skipped = load_ground_truth(tasks_dir)
        assert len(evaluable) == 0
        assert len(skipped) == 1
        assert "no skills with SKILL.md" in skipped[0][1]

    def test_standalone_md_files_ignored(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "task-1"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Do it")
        skills_dir = task_dir / "environment" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "reference.md").write_text("Some reference")

        evaluable, _injected, _skipped = load_ground_truth(tasks_dir)
        assert len(evaluable) == 0

    def test_max_query_chars(self, tmp_path: Path):
        tasks_dir = tmp_path / "tasks"
        _make_task(tasks_dir, "task-1", "A" * 1000, ["skill-a"])

        evaluable, _, _ = load_ground_truth(tasks_dir, max_query_chars=50)
        assert len(evaluable[0].query) == 50

    def test_description_fallback_to_name(self, tmp_path: Path):
        """Skills without frontmatter description use folder name."""
        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "task-1"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Do it")
        skill_dir = task_dir / "environment" / "skills" / "no-desc"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# No frontmatter")

        _evaluable, injected, _skipped = load_ground_truth(tasks_dir)
        assert len(injected) == 1
        assert injected[0].description == "no-desc"

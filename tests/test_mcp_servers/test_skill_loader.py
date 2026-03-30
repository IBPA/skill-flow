"""Tests for skill_loader module."""

import json
from pathlib import Path

import pytest
from mcp_servers.utils.skill_loader import (
    LoadedSkill,
    ResolvedSkill,
    _load_skill_from_dir,
    _load_skill_from_file,
    _parse_frontmatter,
    load_task_skills,
    resolve_eval_skill_folders,
)


class TestParseFrontmatter:
    """Tests for _parse_frontmatter."""

    def test_valid_yaml(self) -> None:
        content = '---\nname: my-skill\ndescription: "A test skill"\n---\n# Body'
        name, desc = _parse_frontmatter(content)
        assert name == "my-skill"
        assert desc == "A test skill"

    def test_missing_fields(self) -> None:
        content = "---\ntitle: something\n---\n# Body"
        name, desc = _parse_frontmatter(content)
        assert name == ""
        assert desc == ""

    def test_no_frontmatter(self) -> None:
        content = "# Just a markdown file\nSome text."
        name, desc = _parse_frontmatter(content)
        assert name == ""
        assert desc == ""

    def test_unclosed_frontmatter(self) -> None:
        content = "---\nname: broken\n# No closing delimiter"
        name, desc = _parse_frontmatter(content)
        assert name == ""
        assert desc == ""

    def test_name_only(self) -> None:
        content = "---\nname: just-name\n---\n# Body"
        name, desc = _parse_frontmatter(content)
        assert name == "just-name"
        assert desc == ""


class TestLoadSkillFromDir:
    """Tests for _load_skill_from_dir."""

    def test_normal_skill_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            '---\nname: my-skill\ndescription: "Test"\n---\n# Content'
        )

        skill = _load_skill_from_dir(skill_dir)
        assert skill is not None
        assert skill.name == "my-skill"
        assert skill.description == "Test"
        assert "# Content" in skill.content

    def test_missing_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "empty-dir"
        skill_dir.mkdir()

        skill = _load_skill_from_dir(skill_dir)
        assert skill is None

    def test_fallback_name_from_dirname(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "dirname-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# No frontmatter")

        skill = _load_skill_from_dir(skill_dir)
        assert skill is not None
        assert skill.name == "dirname-skill"


class TestLoadSkillFromFile:
    """Tests for _load_skill_from_file."""

    def test_standalone_md(self, tmp_path: Path) -> None:
        md_file = tmp_path / "reference.md"
        md_file.write_text("# Reference content\nSome details.")

        skill = _load_skill_from_file(md_file)
        assert skill is not None
        assert skill.name == "reference"
        assert skill.description == ""
        assert "Reference content" in skill.content

    def test_standalone_with_frontmatter(self, tmp_path: Path) -> None:
        md_file = tmp_path / "guide.md"
        md_file.write_text('---\nname: my-guide\ndescription: "A guide"\n---\n# Guide')

        skill = _load_skill_from_file(md_file)
        assert skill is not None
        assert skill.name == "my-guide"
        assert skill.description == "A guide"

    def test_missing_file(self, tmp_path: Path) -> None:
        skill = _load_skill_from_file(tmp_path / "nonexistent.md")
        assert skill is None


class TestLoadTaskSkills:
    """Integration tests for load_task_skills."""

    @pytest.fixture()
    def tasks_dir(self, tmp_path: Path) -> Path:
        """Create a mock tasks directory structure."""
        tasks = tmp_path / "tasks"
        tasks.mkdir()

        # Task with skill subdirectory
        task_a = tasks / "task-alpha"
        skills_a = task_a / "environment" / "skills" / "skill-one"
        skills_a.mkdir(parents=True)
        (skills_a / "SKILL.md").write_text(
            '---\nname: skill-one\ndescription: "First"\n---\n# Skill One'
        )

        # Task with standalone file + subdirectory
        task_b = tasks / "task-beta"
        skills_b = task_b / "environment" / "skills"
        skills_b.mkdir(parents=True)
        (skills_b / "reference.md").write_text("# Reference doc")
        sub = skills_b / "skill-two"
        sub.mkdir()
        (sub / "SKILL.md").write_text(
            '---\nname: skill-two\ndescription: "Second"\n---\n# Skill Two'
        )

        # Task with no skills directory
        task_c = tasks / "task-gamma"
        (task_c / "environment").mkdir(parents=True)

        return tasks

    def test_loads_all_tasks(self, tasks_dir: Path) -> None:
        result = load_task_skills(tasks_dir)
        assert "task-alpha" in result
        assert "task-beta" in result
        assert "task-gamma" not in result

    def test_skill_counts(self, tasks_dir: Path) -> None:
        result = load_task_skills(tasks_dir)
        assert len(result["task-alpha"]) == 1
        assert len(result["task-beta"]) == 2

    def test_filter_by_task_names(self, tasks_dir: Path) -> None:
        result = load_task_skills(tasks_dir, task_names=["task-alpha"])
        assert "task-alpha" in result
        assert "task-beta" not in result

    def test_loaded_skill_content(self, tasks_dir: Path) -> None:
        result = load_task_skills(tasks_dir)
        skill = result["task-alpha"][0]
        assert isinstance(skill, LoadedSkill)
        assert skill.name == "skill-one"
        assert skill.description == "First"
        assert "# Skill One" in skill.content


class TestResolveEvalSkillFolders:
    """Tests for resolve_eval_skill_folders."""

    @pytest.fixture()
    def tasks_dir(self, tmp_path: Path) -> Path:
        """Create a mock tasks directory with skill folders."""
        tasks = tmp_path / "tasks"

        # Task with one skill
        skill_a = tasks / "task-alpha" / "environment" / "skills" / "mesh-analysis"
        skill_a.mkdir(parents=True)
        (skill_a / "SKILL.md").write_text("---\nname: mesh-analysis\n---\n# Content")

        # Task with multiple skills
        skill_b1 = tasks / "task-beta" / "environment" / "skills" / "pid-controller"
        skill_b1.mkdir(parents=True)
        (skill_b1 / "SKILL.md").write_text("---\nname: pid-controller\n---\n# PID")

        skill_b2 = tasks / "task-beta" / "environment" / "skills" / "vehicle-dynamics"
        skill_b2.mkdir(parents=True)
        (skill_b2 / "SKILL.md").write_text("---\nname: vehicle-dynamics\n---\n# Veh")

        return tasks

    def _write_eval_results(
        self, path: Path, task_results: list[dict[str, object]]
    ) -> None:
        """Write a minimal eval results JSON file."""
        data = {"summary": {}, "task_results": task_results}
        path.write_text(json.dumps(data))

    def test_resolves_single_skill(self, tmp_path: Path, tasks_dir: Path) -> None:
        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [
                {
                    "task_id": "task-alpha",
                    "retrieved_skills": [
                        {
                            "key": "skillsbench/task-alpha/mesh-analysis",
                            "description": "Analyzes 3D meshes",
                        }
                    ],
                }
            ],
        )

        resolved = resolve_eval_skill_folders(eval_path, tasks_dir, "task-alpha")
        assert len(resolved) == 1
        assert isinstance(resolved[0], ResolvedSkill)
        assert resolved[0].name == "mesh-analysis"
        assert resolved[0].description == "Analyzes 3D meshes"
        assert resolved[0].folder_path == (
            tasks_dir / "task-alpha" / "environment" / "skills" / "mesh-analysis"
        )

    def test_resolves_multiple_skills(self, tmp_path: Path, tasks_dir: Path) -> None:
        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [
                {
                    "task_id": "task-beta",
                    "retrieved_skills": [
                        {
                            "key": "skillsbench/task-beta/pid-controller",
                            "description": "PID control",
                        },
                        {
                            "key": "skillsbench/task-beta/vehicle-dynamics",
                            "description": "Vehicle sim",
                        },
                    ],
                }
            ],
        )

        resolved = resolve_eval_skill_folders(eval_path, tasks_dir, "task-beta")
        assert len(resolved) == 2
        names = {s.name for s in resolved}
        assert names == {"pid-controller", "vehicle-dynamics"}

    def test_cross_task_skill(self, tmp_path: Path, tasks_dir: Path) -> None:
        """Skill key references a different source task than the evaluated task."""
        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [
                {
                    "task_id": "task-gamma",
                    "retrieved_skills": [
                        {
                            "key": "skillsbench/task-alpha/mesh-analysis",
                            "description": "Cross-task skill",
                        }
                    ],
                }
            ],
        )

        resolved = resolve_eval_skill_folders(eval_path, tasks_dir, "task-gamma")
        assert len(resolved) == 1
        assert resolved[0].name == "mesh-analysis"
        assert "task-alpha" in str(resolved[0].folder_path)

    def test_missing_folder_skipped(self, tmp_path: Path, tasks_dir: Path) -> None:
        """Missing skill folder on disk is skipped with warning."""
        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [
                {
                    "task_id": "task-alpha",
                    "retrieved_skills": [
                        {
                            "key": "skillsbench/task-alpha/nonexistent-skill",
                            "description": "Does not exist",
                        }
                    ],
                }
            ],
        )

        resolved = resolve_eval_skill_folders(eval_path, tasks_dir, "task-alpha")
        assert len(resolved) == 0

    def test_task_not_in_results(self, tmp_path: Path, tasks_dir: Path) -> None:
        """Task not found in eval results returns empty list."""
        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [{"task_id": "other-task", "retrieved_skills": []}],
        )

        resolved = resolve_eval_skill_folders(eval_path, tasks_dir, "task-alpha")
        assert resolved == []

    def test_empty_retrieved_skills(self, tmp_path: Path, tasks_dir: Path) -> None:
        """Task with no retrieved skills returns empty list."""
        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [{"task_id": "task-alpha", "retrieved_skills": []}],
        )

        resolved = resolve_eval_skill_folders(eval_path, tasks_dir, "task-alpha")
        assert resolved == []

    def test_resolves_skillsmp_key(self, tmp_path: Path, tasks_dir: Path) -> None:
        """skillsmp/{skill} keys resolve via corpus_dir."""
        corpus = tmp_path / "corpus"
        skill_dir = corpus / "skillsmp" / "dep-audit"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Dep audit")

        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [
                {
                    "task_id": "task-alpha",
                    "retrieved_skills": [
                        {
                            "key": "skillsmp/dep-audit",
                            "description": "Audit deps",
                        }
                    ],
                }
            ],
        )

        resolved = resolve_eval_skill_folders(
            eval_path, tasks_dir, "task-alpha", corpus_dir=corpus
        )
        assert len(resolved) == 1
        assert resolved[0].name == "dep-audit"
        assert resolved[0].folder_path == skill_dir

    def test_skillsmp_without_corpus_dir_skipped(
        self, tmp_path: Path, tasks_dir: Path
    ) -> None:
        """skillsmp keys are skipped when corpus_dir is not provided."""
        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [
                {
                    "task_id": "task-alpha",
                    "retrieved_skills": [
                        {"key": "skillsmp/some-skill", "description": "X"}
                    ],
                }
            ],
        )

        resolved = resolve_eval_skill_folders(eval_path, tasks_dir, "task-alpha")
        assert resolved == []

    def test_mixed_skillsbench_and_skillsmp(
        self, tmp_path: Path, tasks_dir: Path
    ) -> None:
        """Both skillsbench and skillsmp keys resolve in the same call."""
        corpus = tmp_path / "corpus"
        skill_dir = corpus / "skillsmp" / "ext-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# External")

        eval_path = tmp_path / "results.json"
        self._write_eval_results(
            eval_path,
            [
                {
                    "task_id": "task-alpha",
                    "retrieved_skills": [
                        {
                            "key": "skillsbench/task-alpha/mesh-analysis",
                            "description": "Bench skill",
                        },
                        {
                            "key": "skillsmp/ext-skill",
                            "description": "Corpus skill",
                        },
                    ],
                }
            ],
        )

        resolved = resolve_eval_skill_folders(
            eval_path, tasks_dir, "task-alpha", corpus_dir=corpus
        )
        assert len(resolved) == 2
        names = {s.name for s in resolved}
        assert names == {"mesh-analysis", "ext-skill"}

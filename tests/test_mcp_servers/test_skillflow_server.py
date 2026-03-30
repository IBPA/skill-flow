"""Tests for skillflow_server module."""

import io
import json
import tarfile
from pathlib import Path

import pytest
from mcp_servers.utils.server_helpers import (
    CONTAINER_SKILLS_DIR,
    create_tar_gz,
    format_results,
    log_query,
    resolve_skill_folder,
    skill_name,
)
from skill_flow.retriever.retriever import SearchResult


class TestSkillName:
    """Tests for skill_name."""

    def test_two_segments(self) -> None:
        assert skill_name("skillsmp/my-skill") == "my-skill"

    def test_single_segment(self) -> None:
        assert skill_name("my-skill") == "my-skill"

    def test_three_segments(self) -> None:
        assert skill_name("a/b/c") == "c"


class TestCreateTarGz:
    """Tests for create_tar_gz."""

    def test_creates_valid_tar(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill")

        data = create_tar_gz(skill_dir)

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = tar.getnames()
        assert "SKILL.md" in names

    def test_includes_subdirectories(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill")
        refs = skill_dir / "references"
        refs.mkdir()
        (refs / "helper.py").write_text("# helper")

        data = create_tar_gz(skill_dir)

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = tar.getnames()
        assert "SKILL.md" in names
        assert "references/helper.py" in names


class TestResolveSkillFolder:
    """Tests for resolve_skill_folder."""

    def test_skillsmp_key(self, tmp_path: Path) -> None:
        d = tmp_path / "skillsmp" / "my-skill"
        d.mkdir(parents=True)
        result = resolve_skill_folder("skillsmp/my-skill", tmp_path, None)
        assert result == d

    def test_skillsmp_missing(self, tmp_path: Path) -> None:
        result = resolve_skill_folder("skillsmp/missing", tmp_path, None)
        assert result is None

    def test_skillsbench_key(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        skill_dir = tasks_dir / "my-task" / "environment" / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        result = resolve_skill_folder(
            "skillsbench/my-task/my-skill",
            tmp_path,
            tasks_dir,
        )
        assert result == skill_dir

    def test_skillsbench_no_tasks_dir(self, tmp_path: Path) -> None:
        result = resolve_skill_folder(
            "skillsbench/my-task/my-skill",
            tmp_path,
            None,
        )
        assert result is None

    def test_skillsbench_missing_folder(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        result = resolve_skill_folder(
            "skillsbench/my-task/my-skill",
            tmp_path,
            tasks_dir,
        )
        assert result is None


class TestFormatResults:
    """Tests for format_results."""

    @pytest.fixture()
    def corpus_dir(self, tmp_path: Path) -> Path:
        """Create a corpus with one skill folder."""
        skill_dir = tmp_path / "skillsmp" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# My Skill")
        return tmp_path

    def test_single_result(self, corpus_dir: Path) -> None:
        results = [SearchResult(key="skillsmp/my-skill", score=0.95)]
        response = format_results(results, "https://example.com", corpus_dir)
        assert "Found 1 skills" in response
        assert "curl -sL" in response
        assert "/download/skillsmp/my-skill" in response
        assert f"{CONTAINER_SKILLS_DIR}/my-skill" in response

    def test_empty_results(self, corpus_dir: Path) -> None:
        response = format_results([], "https://example.com", corpus_dir)
        assert "No matching skills found" in response

    def test_missing_folder_skipped(self, tmp_path: Path) -> None:
        results = [SearchResult(key="skillsmp/nonexistent", score=0.9)]
        response = format_results(results, "https://example.com", tmp_path)
        assert "No matching skills found" in response

    def test_multiple_results(self, tmp_path: Path) -> None:
        for name in ["skill-a", "skill-b"]:
            d = tmp_path / "skillsmp" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"# {name}")

        results = [
            SearchResult(key="skillsmp/skill-a", score=0.9),
            SearchResult(key="skillsmp/skill-b", score=0.8),
        ]
        response = format_results(results, "https://example.com", tmp_path)
        assert "Found 2 skills" in response
        assert "skill-a" in response
        assert "skill-b" in response

    def test_partial_missing_folders(self, tmp_path: Path) -> None:
        """Valid folders included, missing folders skipped."""
        d = tmp_path / "skillsmp" / "exists"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("# Exists")

        results = [
            SearchResult(key="skillsmp/exists", score=0.9),
            SearchResult(key="skillsmp/gone", score=0.8),
        ]
        response = format_results(results, "https://example.com", tmp_path)
        assert "Found 1 skills" in response
        assert "exists" in response
        assert "gone" not in response

    def test_skillsbench_key_with_tasks_dir(self, tmp_path: Path) -> None:
        """skillsbench/ keys resolve via tasks_dir."""
        tasks_dir = tmp_path / "tasks"
        skill_dir = tasks_dir / "my-task" / "environment" / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Bench Skill")

        results = [SearchResult(key="skillsbench/my-task/my-skill", score=1.0)]
        response = format_results(
            results,
            "https://example.com",
            tmp_path,
            tasks_dir,
        )
        assert "Found 1 skills" in response
        assert "/download/skillsbench/my-task/my-skill" in response


class TestLogQuery:
    """Tests for log_query."""

    def test_writes_jsonl_entry(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.jsonl"
        results = [SearchResult(key="skillsmp/foo", score=0.9512)]
        log_query("test query", results, 123.4, log_file)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["query"] == "test query"
        assert entry["n_results"] == 1
        assert entry["latency_ms"] == 123.4
        assert entry["retrieved_skills"][0]["key"] == "skillsmp/foo"
        assert entry["retrieved_skills"][0]["score"] == 0.9512

    def test_appends_to_existing(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.jsonl"
        log_query("query 1", [], 10.0, log_file)
        log_query("query 2", [], 20.0, log_file)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["query"] == "query 1"
        assert json.loads(lines[1])["query"] == "query 2"

    def test_includes_timestamp(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.jsonl"
        log_query("q", [], 0.0, log_file)

        entry = json.loads(log_file.read_text().strip())
        assert "timestamp" in entry

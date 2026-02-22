"""Tests for skillflow_retriever_server module."""

import io
import json
import tarfile
from pathlib import Path

import pytest
from mcp_servers.skillflow_retriever_server import (
    CONTAINER_SKILLS_DIR,
    _create_tar_gz,
    _format_results,
    _log_query,
    _skill_name,
)
from skill_flow.retriever.retriever import SearchResult


class TestSkillName:
    """Tests for _skill_name."""

    def test_two_segments(self) -> None:
        assert _skill_name("skillsmp/my-skill") == "my-skill"

    def test_single_segment(self) -> None:
        assert _skill_name("my-skill") == "my-skill"

    def test_three_segments(self) -> None:
        assert _skill_name("a/b/c") == "c"


class TestCreateTarGz:
    """Tests for _create_tar_gz."""

    def test_creates_valid_tar(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# My Skill")

        data = _create_tar_gz(skill_dir)

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

        data = _create_tar_gz(skill_dir)

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = tar.getnames()
        assert "SKILL.md" in names
        assert "references/helper.py" in names


class TestFormatResults:
    """Tests for _format_results."""

    @pytest.fixture()
    def corpus_dir(self, tmp_path: Path) -> Path:
        """Create a corpus with one skill folder."""
        skill_dir = tmp_path / "skillsmp" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# My Skill")
        return tmp_path

    def test_single_result(self, corpus_dir: Path) -> None:
        results = [SearchResult(key="skillsmp/my-skill", score=0.95)]
        response = _format_results(results, "https://example.com", corpus_dir)
        assert "Found 1 skills" in response
        assert "curl -sL" in response
        assert "/download/skillsmp/my-skill" in response
        assert f"{CONTAINER_SKILLS_DIR}/my-skill" in response

    def test_empty_results(self, corpus_dir: Path) -> None:
        response = _format_results([], "https://example.com", corpus_dir)
        assert "No matching skills found" in response

    def test_missing_folder_skipped(self, tmp_path: Path) -> None:
        results = [SearchResult(key="skillsmp/nonexistent", score=0.9)]
        response = _format_results(results, "https://example.com", tmp_path)
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
        response = _format_results(results, "https://example.com", tmp_path)
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
        response = _format_results(results, "https://example.com", tmp_path)
        assert "Found 1 skills" in response
        assert "exists" in response
        assert "gone" not in response


class TestLogQuery:
    """Tests for _log_query."""

    def test_writes_jsonl_entry(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.jsonl"
        results = [SearchResult(key="skillsmp/foo", score=0.9512)]
        _log_query("test query", results, 123.4, log_file)

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
        _log_query("query 1", [], 10.0, log_file)
        _log_query("query 2", [], 20.0, log_file)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["query"] == "query 1"
        assert json.loads(lines[1])["query"] == "query 2"

    def test_includes_timestamp(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.jsonl"
        _log_query("q", [], 0.0, log_file)

        entry = json.loads(log_file.read_text().strip())
        assert "timestamp" in entry

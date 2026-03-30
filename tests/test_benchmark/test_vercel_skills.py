"""Tests for the Vercel skills.sh search and download pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from benchmark.third_party.vercel.cli import _load_queries
from benchmark.third_party.vercel.download import download_skill
from benchmark.third_party.vercel.search import search_skills_sh

if TYPE_CHECKING:
    from collections.abc import Iterator


def _make_skill(
    skill_id: str = "python-testing",
    source: str = "vercel-labs/agent-skills",
    *,
    is_duplicate: bool = False,
) -> dict[str, Any]:
    return {
        "id": f"id-{skill_id}",
        "skillId": skill_id,
        "name": skill_id.replace("-", " ").title(),
        "source": source,
        "installs": 42,
        "isDuplicate": is_duplicate,
    }


# ── search_skills_sh ────────────────────────────────────────────────


class TestSearchSkillsSh:
    def test_returns_top_k_results(self) -> None:
        skills = [_make_skill(f"skill-{i}") for i in range(5)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"skills": skills}
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmark.third_party.vercel.search.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ctx = mock_client.return_value.__enter__.return_value
            ctx.get.return_value = mock_resp

            result = search_skills_sh("testing", top_k=3)

        assert len(result) == 3
        assert result[0]["skillId"] == "skill-0"

    def test_filters_duplicates(self) -> None:
        skills = [
            _make_skill("original"),
            _make_skill("dup", is_duplicate=True),
            _make_skill("another"),
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"skills": skills}
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmark.third_party.vercel.search.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ctx = mock_client.return_value.__enter__.return_value
            ctx.get.return_value = mock_resp

            result = search_skills_sh("test", top_k=5)

        ids = [s["skillId"] for s in result]
        assert "dup" not in ids
        assert len(result) == 2

    def test_empty_results(self) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"skills": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("benchmark.third_party.vercel.search.httpx.Client") as mock_client:
            mock_client.return_value.__enter__ = MagicMock(return_value=MagicMock())
            ctx = mock_client.return_value.__enter__.return_value
            ctx.get.return_value = mock_resp

            result = search_skills_sh("nonexistent")

        assert result == []


# ── Query loading ────────────────────────────────────────────────────


class TestLoadQueries:
    def test_cache_source(self, tmp_path: Path) -> None:
        cache = {"task-a": "query for a", "task-b": "query for b"}
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps(cache))

        args = MagicMock()
        args.query_source = "cache"
        args.query_cache = str(cache_path)

        result = _load_queries(args)
        assert result == cache

    def test_cache_source_with_list_values(self, tmp_path: Path) -> None:
        cache = {"task-a": ["first query", "second query"], "task-b": "single"}
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps(cache))

        args = MagicMock()
        args.query_source = "cache"
        args.query_cache = str(cache_path)

        result = _load_queries(args)
        assert result["task-a"] == "first query"
        assert result["task-b"] == "single"

    def test_name_source(self, tmp_path: Path) -> None:
        cache = {"build-cython-ext": "ignored", "run-pytest": "ignored"}
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps(cache))

        args = MagicMock()
        args.query_source = "name"
        args.query_cache = str(cache_path)

        result = _load_queries(args)
        assert result["build-cython-ext"] == "build cython ext"
        assert result["run-pytest"] == "run pytest"


# ── Download caching ────────────────────────────────────────────────


class TestDownloadSkipIfExists:
    def test_skips_existing_skill(self, tmp_path: Path) -> None:
        dest = tmp_path / "my-task"
        dest.mkdir()
        (dest / "SKILL.md").write_text("# Existing skill")

        result = download_skill("some/repo", "skill-id", "my-task", tmp_path)
        assert result is True


# ── Folder rename ────────────────────────────────────────────────────


class TestDownloadFolderRename:
    @pytest.fixture()
    def _mock_npx(self) -> Iterator[None]:
        """Patch subprocess.run to simulate npx creating a skill folder."""

        def fake_run(cmd: list[str], **kwargs: Any) -> MagicMock:
            cwd = Path(kwargs["cwd"])
            skill_id = cmd[cmd.index("-s") + 1]
            skill_dir = cwd / ".claude" / "skills" / skill_id
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("# Test skill\n")
            scripts_dir = skill_dir / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "run.sh").write_text("#!/bin/bash\necho hi\n")
            return MagicMock(returncode=0)

        with patch(
            "benchmark.third_party.vercel.download.subprocess.run", side_effect=fake_run
        ):
            yield

    @pytest.mark.usefixtures("_mock_npx")
    def test_renames_skill_folder_to_task_name(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = download_skill(
            "vercel-labs/skills", "python-testing", "my-task", output_dir
        )

        assert result is True
        dest = output_dir / "my-task"
        assert dest.is_dir()
        assert (dest / "SKILL.md").exists()
        assert (dest / "scripts" / "run.sh").exists()

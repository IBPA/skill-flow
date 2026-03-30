"""Tests for case study table generation (T13--T15)."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_mod = importlib.import_module("analysis.results.t13_14_15_generate_case_studies")
TOP_K = _mod.TOP_K
load_descriptions = _mod.load_descriptions
load_pipeline_stages = _mod.load_pipeline_stages
_build_table_lines = _mod._build_table_lines


def _make_skill(
    key: str,
    score: float | dict[str, int],
    rank: int,
) -> dict[str, Any]:
    return {"key": key, "score": score, "rank": rank}


def _make_stage_file(
    stage: str,
    task_id: str,
    skills: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "num_tasks": 1,
        "task_results": [{"task_id": task_id, "query": "test query", "skills": skills}],
    }


@pytest.fixture
def pipeline_dir(tmp_path: Path) -> Path:
    """Create a minimal pipeline directory with all 4 stages."""
    pdir = tmp_path / "pipeline"
    pdir.mkdir()

    task_id = "test-task"
    skills_s1 = [
        _make_skill("skillsmp/tool-a", 0.8, 1),
        _make_skill("skillsmp/tool-b", 0.7, 2),
    ]
    skills_s2 = [
        _make_skill(f"skillsbench/{task_id}/oracle-skill", 5.0, 1),
        _make_skill("skillsmp/tool-a", 2.0, 2),
    ]
    skills_s3 = [
        _make_skill(f"skillsbench/{task_id}/oracle-skill", 0.99, 1),
        _make_skill("skillsmp/tool-a", 0.5, 2),
    ]
    skills_s4 = [
        _make_skill(
            f"skillsbench/{task_id}/oracle-skill", {"relevancy": 1, "specificity": 1}, 1
        ),
        _make_skill("skillsmp/tool-a", {"relevancy": 0, "specificity": 0}, 2),
    ]

    files = {
        "pipeline-stage1-retriever.json": _make_stage_file(
            "retriever", task_id, skills_s1
        ),
        "pipeline-stage2-reranker.json": _make_stage_file(
            "reranker", task_id, skills_s2
        ),
        "pipeline-stage3-deep_reranker.json": _make_stage_file(
            "deep_reranker", task_id, skills_s3
        ),
        "pipeline-stage4-selector.json": _make_stage_file(
            "selector", task_id, skills_s4
        ),
    }

    for name, data in files.items():
        (pdir / name).write_text(json.dumps(data), encoding="utf-8")

    return pdir


@pytest.fixture
def descriptions_path(tmp_path: Path) -> Path:
    """Create a minimal descriptions file."""
    descs = {"skillsmp/tool-a": "A useful tool", "skillsmp/tool-b": "Another tool"}
    p = tmp_path / "descriptions.json"
    p.write_text(json.dumps(descs), encoding="utf-8")
    return p


class TestLoadPipelineStages:
    def test_loads_all_stages(self, pipeline_dir: Path) -> None:
        stages = load_pipeline_stages(pipeline_dir)
        assert set(stages.keys()) == {
            "retriever",
            "reranker",
            "deep_reranker",
            "selector",
        }

    def test_respects_top_k(self, pipeline_dir: Path) -> None:
        stages = load_pipeline_stages(pipeline_dir)
        for task_skills in stages["retriever"].values():
            assert len(task_skills) <= TOP_K

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        stages = load_pipeline_stages(tmp_path / "nonexistent")
        assert stages == {}


class TestLoadDescriptions:
    def test_loads(self, descriptions_path: Path) -> None:
        descs = load_descriptions(descriptions_path)
        assert "skillsmp/tool-a" in descs

    def test_missing_returns_empty(self, tmp_path: Path) -> None:
        assert load_descriptions(tmp_path / "missing.json") == {}


class TestBuildTableLines:
    def test_contains_table_structure(self, pipeline_dir: Path) -> None:
        stages = load_pipeline_stages(pipeline_dir)
        lines = _build_table_lines("test-task", stages, 1)
        text = "\n".join(lines)
        assert "\\begin{tabular}" in text
        assert "\\end{tabular}" in text
        assert "\\bottomrule" in text

    def test_gt_skill_gets_checkmark(self, pipeline_dir: Path) -> None:
        stages = load_pipeline_stages(pipeline_dir)
        lines = _build_table_lines("test-task", stages, 1)
        text = "\n".join(lines)
        assert "\\checkmark" in text

    def test_balanced_braces(self, pipeline_dir: Path) -> None:
        stages = load_pipeline_stages(pipeline_dir)
        lines = _build_table_lines("test-task", stages, 1)
        text = "\n".join(lines)
        opens = text.count("{")
        closes = text.count("}")
        assert opens == closes, f"Unbalanced braces: {opens} open vs {closes} close"

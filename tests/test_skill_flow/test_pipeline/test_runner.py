"""Tests for skill_flow.pipeline.runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from skill_flow.config import Config, PipelineConfig
from skill_flow.pipeline.models import (
    PipelineStageResult,
    SkillResult,
    TaskInstruction,
    _to_stage_results,
    _write_stage_results,
    load_task_instructions,
    write_result_cache,
)
from skill_flow.pipeline.runner import run_pipeline
from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from pathlib import Path


def _no_cache(*_args: object, **_kwargs: object) -> tuple[None, int]:
    return None, 0


class TestLoadTaskInstructions:
    def test_loads_from_directory(self, tmp_path: Path) -> None:
        for name in ["task-a", "task-b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "instruction.md").write_text(f"Do {name}")

        result = load_task_instructions(tmp_path)

        assert len(result) == 2
        assert result[0].task_id == "task-a"
        assert result[0].query == "Do task-a"
        assert result[1].task_id == "task-b"

    def test_skips_missing_instruction(self, tmp_path: Path) -> None:
        (tmp_path / "has-instruction").mkdir()
        (tmp_path / "has-instruction" / "instruction.md").write_text("query")
        (tmp_path / "no-instruction").mkdir()

        result = load_task_instructions(tmp_path)

        assert len(result) == 1
        assert result[0].task_id == "has-instruction"

    def test_truncates_query(self, tmp_path: Path) -> None:
        d = tmp_path / "task"
        d.mkdir()
        (d / "instruction.md").write_text("a" * 200)

        result = load_task_instructions(tmp_path, max_query_chars=50)

        assert len(result[0].query) == 50

    def test_max_tasks(self, tmp_path: Path) -> None:
        for i in range(5):
            d = tmp_path / f"task-{i:02d}"
            d.mkdir()
            (d / "instruction.md").write_text(f"query {i}")

        result = load_task_instructions(tmp_path, max_tasks=2)

        assert len(result) == 2

    def test_include_tasks(self, tmp_path: Path) -> None:
        for name in ["task-a", "task-b", "task-c"]:
            d = tmp_path / name
            d.mkdir()
            (d / "instruction.md").write_text(f"Do {name}")

        result = load_task_instructions(tmp_path, include_tasks=["task-a", "task-c"])

        assert len(result) == 2
        ids = [r.task_id for r in result]
        assert "task-a" in ids
        assert "task-c" in ids


class TestToStageResults:
    def test_converts_search_results(self) -> None:
        task = TaskInstruction(task_id="t1", query="q1")
        results = [
            SearchResult(key="k1", score=0.9),
            SearchResult(key="k2", score=0.7),
        ]

        stage = _to_stage_results(task, results)

        assert stage.task_id == "t1"
        assert stage.query == "q1"
        assert len(stage.skills) == 2
        assert stage.skills[0].key == "k1"
        assert stage.skills[0].rank == 1
        assert stage.skills[1].rank == 2


class TestWriteStageResults:
    def test_writes_json(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "stage.json"
        results = [
            PipelineStageResult(
                task_id="t1",
                query="q1",
                skills=[SkillResult(key="k1", score=0.9, rank=1)],
            ),
        ]

        _write_stage_results(path, "retriever", results)

        data = json.loads(path.read_text())
        assert data["stage"] == "retriever"
        assert data["num_tasks"] == 1
        assert data["task_results"][0]["task_id"] == "t1"


class TestWriteSelectorCache:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        cache = {"task-a": ["skill-1", "skill-2"], "task-b": ["skill-3"]}

        write_result_cache(cache, path)

        data = json.loads(path.read_text())
        assert data == cache

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "cache.json"

        write_result_cache({"t": ["s"]}, path)

        assert path.exists()


class TestRunPipeline:
    @patch("skill_flow.pipeline.runner.find_latest_cache", side_effect=_no_cache)
    @patch("skill_flow.pipeline.runner.load_content_maps", return_value=({}, {}))
    @patch("skill_flow.pipeline.stages.Selector")
    @patch("skill_flow.pipeline.stages.Reranker")
    @patch("skill_flow.pipeline.stage_helpers.IndexSearcher")
    @patch("skill_flow.pipeline.stage_helpers.Encoder")
    def test_runs_pipeline_and_produces_cache(
        self,
        mock_encoder_cls: MagicMock,
        mock_searcher_cls: MagicMock,
        mock_reranker_cls: MagicMock,
        mock_selector_cls: MagicMock,
        _mock_content: MagicMock,
        _mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Set up task dir
        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "fix-bug"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Fix the bug in auth module")

        output_dir = tmp_path / "output"

        # Mock searcher
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            SearchResult(key="skillsmp/auth-fix", score=0.9, content="# Auth"),
            SearchResult(key="skillsmp/testing", score=0.7, content="# Test"),
        ]
        mock_searcher_cls.return_value = mock_searcher

        # Mock reranker
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            SearchResult(key="skillsmp/auth-fix", score=5.0, content="# Auth"),
            SearchResult(key="skillsmp/testing", score=2.0, content="# Test"),
        ]
        mock_reranker_cls.return_value = mock_reranker

        # Mock selector
        mock_selector = MagicMock()
        mock_selector.select.return_value = [
            SearchResult(key="skillsmp/auth-fix", score=5.0, content="# Auth"),
        ]
        mock_selector.get_labels.return_value = (
            {"skillsmp/auth-fix"},
            {"skillsmp/auth-fix"},
        )
        mock_selector_cls.return_value = mock_selector

        # Config with reranker + selector enabled, no deep_reranker
        config = Config.model_validate(
            {
                "index": {"output_index_path": str(tmp_path / "index")},
                "models": {
                    "retriever": {"top_k": 10},
                    "reranker": {"enabled": True},
                    "deep_reranker": {"enabled": False},
                    "selector": {"enabled": True},
                },
            }
        )
        pipeline_config = PipelineConfig(
            tasks_dir=str(tasks_dir),
            output_dir=str(output_dir),
        )

        cache = run_pipeline(config, pipeline_config)

        # Verify cache
        assert "fix-bug" in cache
        assert cache["fix-bug"] == ["skillsmp/auth-fix"]

        # Verify selector received task_id
        mock_selector.select.assert_called_once()
        call_kwargs = mock_selector.select.call_args
        assert call_kwargs[1]["task_id"] == "fix-bug"

        # Verify stage files written
        assert (output_dir / "pipeline-stage1-retriever.json").exists()
        assert (output_dir / "pipeline-stage2-reranker.json").exists()
        assert not (output_dir / "pipeline-stage3-deep-reranker.json").exists()
        assert (output_dir / "pipeline-stage4-selector.json").exists()
        assert (output_dir / "result_cache.json").exists()

    @patch("skill_flow.pipeline.runner.find_latest_cache", side_effect=_no_cache)
    @patch("skill_flow.pipeline.runner.load_content_maps", return_value=({}, {}))
    @patch("skill_flow.pipeline.stage_helpers.IndexSearcher")
    @patch("skill_flow.pipeline.stage_helpers.Encoder")
    def test_retriever_only(
        self,
        mock_encoder_cls: MagicMock,
        mock_searcher_cls: MagicMock,
        _mock_content: MagicMock,
        _mock_cache: MagicMock,
        tmp_path: Path,
    ) -> None:
        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "task-1"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Do something")

        output_dir = tmp_path / "output"

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            SearchResult(key="skillsmp/a", score=0.5),
        ]
        mock_searcher_cls.return_value = mock_searcher

        config = Config.model_validate(
            {
                "index": {"output_index_path": str(tmp_path / "index")},
                "models": {
                    "retriever": {"top_k": 10},
                    "reranker": {"enabled": False},
                    "deep_reranker": {"enabled": False},
                    "selector": {"enabled": False},
                },
            }
        )
        pipeline_config = PipelineConfig(
            tasks_dir=str(tasks_dir),
            output_dir=str(output_dir),
        )

        cache = run_pipeline(config, pipeline_config)

        assert cache == {"task-1": ["skillsmp/a"]}
        assert (output_dir / "pipeline-stage1-retriever.json").exists()
        assert not (output_dir / "pipeline-stage2-reranker.json").exists()

    def test_empty_tasks_dir(self, tmp_path: Path) -> None:
        tasks_dir = tmp_path / "empty"
        tasks_dir.mkdir()

        config = Config()
        pipeline_config = PipelineConfig(
            tasks_dir=str(tasks_dir),
            output_dir=str(tmp_path / "output"),
        )

        cache = run_pipeline(config, pipeline_config)

        assert cache == {}

    @patch("skill_flow.pipeline.stages.Selector")
    @patch("skill_flow.pipeline.runner.load_content_maps", return_value=({}, {}))
    def test_resumes_from_stage2_cache(
        self,
        _mock_content: MagicMock,
        mock_selector_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When stage 2 output exists, stages 1 and 2 should be skipped."""
        tasks_dir = tmp_path / "tasks"
        task_dir = tasks_dir / "fix-bug"
        task_dir.mkdir(parents=True)
        (task_dir / "instruction.md").write_text("Fix the bug")

        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True)

        # Write a fake stage 2 cache
        stage2_data = {
            "stage": "reranker",
            "num_tasks": 1,
            "task_results": [
                {
                    "task_id": "fix-bug",
                    "query": "Fix the bug",
                    "skills": [
                        {"key": "skillsmp/auth-fix", "score": 5.0, "rank": 1},
                    ],
                }
            ],
        }
        (output_dir / "pipeline-stage2-reranker.json").write_text(
            json.dumps(stage2_data),
        )

        mock_selector = MagicMock()
        mock_selector.select.return_value = [
            SearchResult(key="skillsmp/auth-fix", score=5.0),
        ]
        mock_selector.get_labels.return_value = (
            {"skillsmp/auth-fix"},
            {"skillsmp/auth-fix"},
        )
        mock_selector_cls.return_value = mock_selector

        config = Config.model_validate(
            {
                "index": {"output_index_path": str(tmp_path / "index")},
                "models": {
                    "retriever": {"top_k": 10},
                    "reranker": {"enabled": True},
                    "deep_reranker": {"enabled": False},
                    "selector": {"enabled": True},
                },
            }
        )
        pipeline_config = PipelineConfig(
            tasks_dir=str(tasks_dir),
            output_dir=str(output_dir),
        )

        cache = run_pipeline(config, pipeline_config)

        assert cache == {"fix-bug": ["skillsmp/auth-fix"]}
        mock_selector.select.assert_called_once()
        # Stage 4 should be written
        assert (output_dir / "pipeline-stage4-selector.json").exists()

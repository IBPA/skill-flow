"""Tests for elapsed_ms field and latency summary generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from skill_flow.pipeline.models import (
    PipelineStageResult,
    SkillResult,
    TaskInstruction,
    _compute_latency_stats,
    _to_selector_results,
    _to_stage_results,
    _write_stage_results,
    write_latency_summary,
)
from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from pathlib import Path


class TestElapsedMs:
    def test_default_zero(self) -> None:
        r = PipelineStageResult(task_id="t1", query="q", skills=[])
        assert r.elapsed_ms == 0.0

    def test_custom_value(self) -> None:
        r = PipelineStageResult(task_id="t1", query="q", skills=[], elapsed_ms=42.5)
        assert r.elapsed_ms == 42.5

    def test_to_stage_results_passes_elapsed(self) -> None:
        task = TaskInstruction(task_id="t1", query="q")
        results = [SearchResult(key="k1", score=0.9)]
        stage = _to_stage_results(task, results, elapsed_ms=100.0)
        assert stage.elapsed_ms == 100.0

    def test_to_selector_results_passes_elapsed(self) -> None:
        task = TaskInstruction(task_id="t1", query="q")
        candidates = [SearchResult(key="k1", score=0.9)]
        result = _to_selector_results(task, candidates, {"k1"}, set(), elapsed_ms=55.0)
        assert result.elapsed_ms == 55.0

    def test_serialized_in_stage_output(self, tmp_path: Path) -> None:
        path = tmp_path / "stage.json"
        results = [
            PipelineStageResult(
                task_id="t1",
                query="q",
                skills=[SkillResult(key="k1", score=0.9, rank=1)],
                elapsed_ms=123.4,
            ),
        ]
        _write_stage_results(path, "test", results)
        data = json.loads(path.read_text())
        assert data["task_results"][0]["elapsed_ms"] == 123.4


class TestComputeLatencyStats:
    def test_empty_list(self) -> None:
        stats = _compute_latency_stats([])
        assert stats == {"median_ms": 0.0, "mean_ms": 0.0, "p95_ms": 0.0}

    def test_single_value(self) -> None:
        stats = _compute_latency_stats([42.0])
        assert stats["median_ms"] == 42.0
        assert stats["mean_ms"] == 42.0
        assert stats["p95_ms"] == 42.0

    def test_multiple_values(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = _compute_latency_stats(values)
        assert stats["median_ms"] == 30.0
        assert stats["mean_ms"] == 30.0
        assert stats["p95_ms"] == 50.0


class TestWriteLatencySummary:
    def test_writes_summary(self, tmp_path: Path) -> None:
        stage_data = {
            "stage": "retriever",
            "num_tasks": 2,
            "task_results": [
                {"task_id": "t1", "query": "q1", "skills": [], "elapsed_ms": 10.0},
                {"task_id": "t2", "query": "q2", "skills": [], "elapsed_ms": 20.0},
            ],
        }
        (tmp_path / "pipeline-stage1-retriever.json").write_text(json.dumps(stage_data))

        write_latency_summary(tmp_path)

        summary_path = tmp_path / "latency_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "retriever" in summary["stages"]
        agg = summary["stages"]["retriever"]["aggregate"]
        assert agg["median_ms"] == 15.0
        assert agg["mean_ms"] == 15.0

    def test_no_op_when_no_stages(self, tmp_path: Path) -> None:
        write_latency_summary(tmp_path)
        assert not (tmp_path / "latency_summary.json").exists()

    def test_includes_per_task_timings(self, tmp_path: Path) -> None:
        stage_data = {
            "stage": "reranker",
            "num_tasks": 1,
            "task_results": [
                {"task_id": "t1", "query": "q", "skills": [], "elapsed_ms": 42.0},
            ],
        }
        (tmp_path / "pipeline-stage2-reranker.json").write_text(json.dumps(stage_data))

        write_latency_summary(tmp_path)

        summary = json.loads((tmp_path / "latency_summary.json").read_text())
        assert summary["stages"]["reranker"]["per_task_ms"]["t1"] == 42.0

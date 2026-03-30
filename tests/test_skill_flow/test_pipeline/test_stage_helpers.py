"""Tests for skill_flow.pipeline.stage_helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from skill_flow.pipeline.models import PipelineStageResult, TaskInstruction
from skill_flow.pipeline.stage_helpers import (
    build_selector_stage_results,
    measure_call,
    process_retrieval_task,
    process_selector_task,
    write_eval_if_needed,
)
from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.eval.models import TaskResult


class TestMeasureCall:
    def test_returns_result_and_timing(self) -> None:
        result, elapsed_ms = measure_call(lambda x: x * 2, 5)

        assert result == 10
        assert elapsed_ms >= 0.0

    def test_timing_is_positive(self) -> None:
        def slow_fn() -> str:
            total = 0
            for i in range(10000):
                total += i
            return f"done-{total}"

        result, elapsed_ms = measure_call(slow_fn)

        assert isinstance(result, str)
        assert elapsed_ms > 0.0


class TestProcessRetrievalTask:
    def test_appends_result_and_updates_candidates(self) -> None:
        task = TaskInstruction(task_id="t1", query="q")
        results = [SearchResult(key="k1", score=0.9)]
        candidates: dict[str, list[SearchResult]] = {}
        stage: list[PipelineStageResult] = []
        evals: list[TaskResult] = []

        process_retrieval_task(
            task,
            results,
            candidates,
            None,
            10,
            stage,
            evals,
            elapsed_ms=42.0,
        )

        assert len(stage) == 1
        assert stage[0].task_id == "t1"
        assert stage[0].elapsed_ms == 42.0
        assert candidates["t1"] == results
        assert len(evals) == 0  # no gt_context


class TestProcessSelectorTask:
    def test_updates_candidates(self) -> None:
        task = TaskInstruction(task_id="t1", query="q")
        selected = [SearchResult(key="k1", score=1.0)]
        inputs: dict[str, list[SearchResult]] = {
            "t1": [SearchResult(key="k1", score=0.5)],
        }
        candidates: dict[str, list[SearchResult]] = {}
        evals: list[TaskResult] = []

        process_selector_task(
            task,
            selected,
            inputs,
            candidates,
            None,
            10,
            evals,
        )

        assert candidates["t1"] == selected


class TestBuildSelectorStageResults:
    def test_creates_results_with_timings(self) -> None:
        tasks = [TaskInstruction(task_id="t1", query="q")]
        inputs = {"t1": [SearchResult(key="k1", score=0.5)]}
        mock_selector = MagicMock()
        mock_selector.get_labels.return_value = ({"k1"}, {"k1"})
        timings = {"t1": 123.4}

        results = build_selector_stage_results(tasks, inputs, mock_selector, timings)

        assert len(results) == 1
        assert results[0].elapsed_ms == 123.4
        assert results[0].task_id == "t1"


class TestWriteEvalIfNeeded:
    def test_no_op_when_no_gt(self, tmp_path: Path) -> None:
        # Should not raise or write anything
        write_eval_if_needed(
            "test",
            [],
            None,
            10,
            tmp_path,
            tmp_path,
            tmp_path,
        )
        assert not list(tmp_path.glob("eval-*.json"))

    def test_no_op_when_empty_results(self, tmp_path: Path) -> None:
        gt = MagicMock()
        write_eval_if_needed(
            "test",
            [],
            gt,
            10,
            tmp_path,
            tmp_path,
            tmp_path,
        )
        assert not list(tmp_path.glob("eval-*.json"))

"""Unit tests for skill_flow.eval.runner."""

from unittest.mock import MagicMock

import numpy as np
from skill_flow.eval.models import (
    InjectedSkill,
    TaskGroundTruth,
    TaskResult,
)
from skill_flow.eval.reporting import build_summary
from skill_flow.eval.runner import (
    _augment_index,
    _build_task_result,
)
from skill_flow.retriever.retriever import SearchResult


def _make_task_gt(
    task_id: str = "task-1",
    query: str = "test query",
    gt_keys: list[str] | None = None,
) -> TaskGroundTruth:
    keys = gt_keys or ["skillsbench/task-1/a"]
    return TaskGroundTruth(
        task_id=task_id,
        query=query,
        ground_truth_keys=keys,
        all_skill_names=[k.split("/")[-1] for k in keys],
        injected_skills=keys,
    )


class TestAugmentIndex:
    def test_no_skills(self):
        searcher = MagicMock()
        encoder = MagicMock()
        _augment_index(searcher, encoder, [])
        encoder.encode_documents.assert_not_called()

    def test_adds_vectors_and_keys(self):
        searcher = MagicMock()
        searcher._index = MagicMock()
        searcher._skill_keys = ["skillsmp/existing"]
        encoder = MagicMock()
        vectors = np.zeros((2, 768), dtype=np.float32)
        encoder.encode_documents.return_value = vectors

        skills = [
            InjectedSkill(key="skillsbench/t1/a", name="a", description="d a"),
            InjectedSkill(key="skillsbench/t1/b", name="b", description="d b"),
        ]
        _augment_index(searcher, encoder, skills)

        encoder.encode_documents.assert_called_once_with(["d a", "d b"])
        searcher._index.add.assert_called_once()
        assert "skillsbench/t1/a" in searcher._skill_keys
        assert "skillsbench/t1/b" in searcher._skill_keys

    def test_injects_contents(self):
        searcher = MagicMock()
        searcher._index = MagicMock()
        searcher._skill_keys = []
        encoder = MagicMock()
        encoder.encode_documents.return_value = np.zeros((1, 768), dtype=np.float32)

        skills = [
            InjectedSkill(
                key="skillsbench/t1/a",
                name="a",
                description="d",
                content="# Full content",
            ),
        ]
        _augment_index(searcher, encoder, skills)

        searcher.add_contents.assert_called_once_with(
            {"skillsbench/t1/a": "# Full content"}
        )


class TestBuildTaskResult:
    def test_computes_metrics(self):
        results = [
            SearchResult(key="skillsmp/x", score=0.9, description="desc x"),
            SearchResult(key="skillsbench/task-1/a", score=0.8, description="desc a"),
            SearchResult(key="skillsmp/y", score=0.7),
        ]

        task = _make_task_gt(gt_keys=["skillsbench/task-1/a"])
        result = _build_task_result(task, results, ks=[1, 3])

        assert result.task_id == "task-1"
        assert result.query == "test query"
        assert result.reciprocal_rank == 0.5
        assert result.recall_at[1] == 0.0
        assert result.recall_at[3] == 1.0
        assert result.precision_at[1] == 0.0
        assert result.precision_at[3] == 1.0 / 3
        assert result.hit_at[1] == 0.0
        assert result.hit_at[3] == 1.0
        assert len(result.retrieved_skills) == 3
        assert result.retrieved_skills[0].key == "skillsmp/x"
        assert result.retrieved_skills[0].score == 0.9
        assert result.retrieved_skills[1].description == "desc a"

    def test_no_hits(self):
        results = [
            SearchResult(key="skillsmp/x", score=0.9),
        ]

        task = _make_task_gt(gt_keys=["skillsbench/task-1/a"])
        result = _build_task_result(task, results, ks=[1])

        assert result.query == "test query"
        assert result.reciprocal_rank == 0.0
        assert result.recall_at[1] == 0.0
        assert result.precision_at[1] == 0.0
        assert result.hit_at[1] == 0.0
        assert len(result.retrieved_skills) == 1


class TestBuildSummary:
    def test_aggregation(self):
        results = [
            TaskResult(
                task_id="t1",
                num_ground_truth=1,
                num_injected=1,
                recall_at={5: 1.0, 10: 1.0},
                precision_at={5: 0.2, 10: 0.1},
                hit_at={5: 1.0, 10: 1.0},
                reciprocal_rank=1.0,
            ),
            TaskResult(
                task_id="t2",
                num_ground_truth=1,
                num_injected=1,
                recall_at={5: 0.0, 10: 0.5},
                precision_at={5: 0.0, 10: 0.1},
                hit_at={5: 0.0, 10: 1.0},
                reciprocal_rank=0.0,
            ),
        ]

        summary = build_summary(
            results,
            num_tasks_total=4,
            num_tasks_no_skills=2,
            num_skills_injected=3,
            ks=[5, 10],
        )

        assert summary.num_tasks_total == 4
        assert summary.num_tasks_evaluated == 2
        assert summary.num_tasks_no_skills == 2
        assert summary.num_skills_injected == 3
        assert summary.mean_recall_at[5] == 0.5
        assert summary.mean_recall_at[10] == 0.75
        assert summary.mean_precision_at[5] == 0.1
        assert summary.mean_precision_at[10] == 0.1
        assert summary.mean_hit_at[5] == 0.5
        assert summary.mean_hit_at[10] == 1.0
        assert summary.mrr == 0.5

    def test_empty_results(self):
        summary = build_summary(
            [],
            num_tasks_total=0,
            num_tasks_no_skills=0,
            num_skills_injected=0,
            ks=[5],
        )
        assert summary.mrr == 0.0
        assert summary.mean_recall_at[5] == 0.0
        assert summary.mean_precision_at[5] == 0.0

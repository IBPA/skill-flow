"""Tests for src.eval.models."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from skill_flow.eval.models import (
    EVAL_KS,
    EvalReport,
    EvalSummary,
    InjectedSkill,
    RetrievedSkill,
    TaskGroundTruth,
    TaskResult,
    filter_ks,
)


def _assert_frozen(instance: BaseModel, field: str, value: Any) -> None:
    """Assert that setting a field on a frozen model raises ValidationError."""
    with pytest.raises(ValidationError):
        setattr(instance, field, value)


class TestFilterKs:
    def test_returns_ks_up_to_top_k(self):
        assert filter_ks(10) == [1, 2, 5, 10]

    def test_returns_all_when_top_k_large(self):
        assert filter_ks(1000) == EVAL_KS

    def test_returns_all_for_zero(self):
        assert filter_ks(0) == EVAL_KS

    def test_returns_subset(self):
        assert filter_ks(5) == [1, 2, 5]


class TestInjectedSkill:
    def test_creation(self):
        s = InjectedSkill(key="skillsbench/t/foo", name="foo", description="desc")
        assert s.key == "skillsbench/t/foo"
        assert s.name == "foo"
        assert s.description == "desc"
        assert s.content == ""

    def test_creation_with_content(self):
        s = InjectedSkill(
            key="skillsbench/t/foo",
            name="foo",
            description="desc",
            content="# Full SKILL.md",
        )
        assert s.content == "# Full SKILL.md"

    def test_frozen(self):
        s = InjectedSkill(key="skillsbench/t/foo", name="foo", description="desc")
        _assert_frozen(s, "key", "other")


class TestTaskGroundTruth:
    def test_creation(self):
        gt = TaskGroundTruth(
            task_id="task-1",
            query="do something",
            ground_truth_keys=["skillsmp/a", "skillsbench/b"],
            all_skill_names=["a", "b"],
            injected_skills=["skillsbench/b"],
        )
        assert gt.task_id == "task-1"
        assert len(gt.ground_truth_keys) == 2
        assert gt.injected_skills == ["skillsbench/b"]

    def test_frozen(self):
        gt = TaskGroundTruth(
            task_id="t",
            query="q",
            ground_truth_keys=[],
            all_skill_names=[],
            injected_skills=[],
        )
        _assert_frozen(gt, "task_id", "other")


class TestRetrievedSkill:
    def test_creation(self):
        rs = RetrievedSkill(key="k", score=0.9, description="d", content="c")
        assert rs.key == "k"
        assert rs.score == 0.9
        assert rs.description == "d"
        assert rs.content == "c"

    def test_defaults(self):
        rs = RetrievedSkill(key="k", score=0.5)
        assert rs.description == ""
        assert rs.content == ""

    def test_frozen(self):
        rs = RetrievedSkill(key="k", score=0.5)
        _assert_frozen(rs, "score", 1.0)


class TestTaskResult:
    def test_creation(self):
        tr = TaskResult(
            task_id="t",
            query="do something",
            num_ground_truth=2,
            num_injected=1,
            retrieved_skills=[
                RetrievedSkill(key="a", score=0.9, description="d a"),
            ],
            recall_at={5: 0.5, 10: 1.0},
            precision_at={5: 0.2, 10: 0.1},
            hit_at={5: 1.0, 10: 1.0},
            reciprocal_rank=0.5,
        )
        assert tr.num_ground_truth == 2
        assert tr.query == "do something"
        assert tr.recall_at[5] == 0.5
        assert tr.precision_at[5] == 0.2
        assert len(tr.retrieved_skills) == 1
        assert tr.retrieved_skills[0].score == 0.9

    def test_defaults(self):
        tr = TaskResult(
            task_id="t",
            num_ground_truth=1,
            num_injected=0,
            recall_at={},
            hit_at={},
            reciprocal_rank=0.0,
        )
        assert tr.query == ""
        assert tr.retrieval_query == ""
        assert tr.rerank_query == ""
        assert tr.retrieved_skills == []

    def test_retrieval_query(self):
        tr = TaskResult(
            task_id="t",
            query="original",
            retrieval_query="concise search query",
            num_ground_truth=1,
            num_injected=0,
            recall_at={},
            hit_at={},
            reciprocal_rank=0.0,
        )
        assert tr.retrieval_query == "concise search query"

    def test_frozen(self):
        tr = TaskResult(
            task_id="t",
            num_ground_truth=1,
            num_injected=0,
            recall_at={},
            precision_at={},
            hit_at={},
            reciprocal_rank=0.0,
        )
        _assert_frozen(tr, "task_id", "other")


class TestEvalSummary:
    def test_creation(self):
        es = EvalSummary(
            num_tasks_total=10,
            num_tasks_evaluated=8,
            num_tasks_no_skills=2,
            num_skills_injected=5,
            mean_recall_at={10: 0.6},
            mean_precision_at={10: 0.1},
            mean_hit_at={10: 0.8},
            mrr=0.55,
        )
        assert es.num_tasks_total == 10
        assert es.mrr == 0.55

    def test_frozen(self):
        es = EvalSummary(
            num_tasks_total=1,
            num_tasks_evaluated=1,
            num_tasks_no_skills=0,
            num_skills_injected=0,
            mean_recall_at={},
            mean_precision_at={},
            mean_hit_at={},
            mrr=0.0,
        )
        _assert_frozen(es, "mrr", 1.0)


class TestEvalReport:
    def test_creation(self):
        summary = EvalSummary(
            num_tasks_total=1,
            num_tasks_evaluated=1,
            num_tasks_no_skills=0,
            num_skills_injected=0,
            mean_recall_at={10: 1.0},
            mean_precision_at={10: 0.1},
            mean_hit_at={10: 1.0},
            mrr=1.0,
        )
        cfg: dict[str, object] = {"tasks_dir": "/t"}
        report = EvalReport(summary=summary, task_results=[], config=cfg)
        assert report.summary.mrr == 1.0
        assert report.task_results == []

    def test_creation_with_empty_config(self):
        summary = EvalSummary(
            num_tasks_total=1,
            num_tasks_evaluated=1,
            num_tasks_no_skills=0,
            num_skills_injected=0,
            mean_recall_at={},
            mean_precision_at={},
            mean_hit_at={},
            mrr=0.0,
        )
        report = EvalReport(summary=summary, task_results=[])
        assert report.config == {}

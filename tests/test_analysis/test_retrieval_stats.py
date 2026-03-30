"""Tests for retrieval stats integration layer."""

import pytest
from analysis.stats.retrieval_stats import (
    all_retrieval_cis,
    retrieval_ci,
    retrieval_paired_test,
)
from analysis.stats.types import ConfidenceInterval, PairedTestResult
from skill_flow.eval.models import EvalReport, EvalSummary, TaskResult


def _make_task_result(task_id: str, recall: float, mrr: float) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        num_ground_truth=2,
        num_injected=2,
        recall_at={1: recall, 5: min(recall + 0.2, 1.0), 10: min(recall + 0.3, 1.0)},
        precision_at={1: recall, 5: recall / 5, 10: recall / 10},
        hit_at={1: float(recall > 0), 5: 1.0, 10: 1.0},
        reciprocal_rank=mrr,
    )


def _make_report(task_results: list[TaskResult]) -> EvalReport:
    return EvalReport(
        summary=EvalSummary(
            num_tasks_total=len(task_results),
            num_tasks_evaluated=len(task_results),
            num_tasks_no_skills=0,
            num_skills_injected=len(task_results) * 2,
            mean_recall_at={},
            mean_hit_at={},
            mrr=0.0,
        ),
        task_results=task_results,
        config={"tasks_dir": "fake/"},
    )


@pytest.fixture
def report_high() -> EvalReport:
    tasks = [_make_task_result(f"t-{i}", 0.8, 0.9) for i in range(40)]
    tasks += [_make_task_result(f"t-{i}", 0.3, 0.5) for i in range(40, 50)]
    return _make_report(tasks)


@pytest.fixture
def report_low() -> EvalReport:
    tasks = [_make_task_result(f"t-{i}", 0.1, 0.2) for i in range(40)]
    tasks += [_make_task_result(f"t-{i}", 0.05, 0.1) for i in range(40, 50)]
    return _make_report(tasks)


class TestRetrievalCI:
    def test_recall_ci(self, report_high: EvalReport) -> None:
        ci = retrieval_ci(report_high, "recall", k=1)
        assert isinstance(ci, ConfidenceInterval)
        assert ci.ci_lo <= ci.mean <= ci.ci_hi

    def test_precision_ci(self, report_high: EvalReport) -> None:
        ci = retrieval_ci(report_high, "precision", k=5)
        assert ci.ci_lo <= ci.mean <= ci.ci_hi

    def test_mrr_ci(self, report_high: EvalReport) -> None:
        ci = retrieval_ci(report_high, "mrr")
        assert ci.ci_lo <= ci.mean <= ci.ci_hi

    def test_unknown_metric_raises(self, report_high: EvalReport) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            retrieval_ci(report_high, "invalid")


class TestRetrievalPairedTest:
    def test_same_report_not_significant(self, report_high: EvalReport) -> None:
        result = retrieval_paired_test(report_high, report_high, "mrr")
        assert isinstance(result, PairedTestResult)
        assert result.observed_diff == 0.0
        assert not result.significant

    def test_different_reports_significant(
        self, report_high: EvalReport, report_low: EvalReport
    ) -> None:
        result = retrieval_paired_test(report_high, report_low, "recall", k=1)
        assert result.observed_diff > 0
        assert result.significant

    def test_select_recall_metric(self, report_high: EvalReport) -> None:
        result = retrieval_paired_test(report_high, report_high, "select_recall")
        assert not result.significant


class TestAllRetrievalCIs:
    def test_returns_all_keys(self, report_high: EvalReport) -> None:
        cis = all_retrieval_cis(report_high, ks=[1, 5, 10])
        assert "R@1" in cis
        assert "R@5" in cis
        assert "R@10" in cis
        assert "P@1" in cis
        assert "P@5" in cis
        assert "MRR" in cis

    def test_all_cis_valid(self, report_high: EvalReport) -> None:
        cis = all_retrieval_cis(report_high, ks=[1, 5])
        for ci in cis.values():
            assert isinstance(ci, ConfidenceInterval)
            assert ci.ci_lo <= ci.mean <= ci.ci_hi

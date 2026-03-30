"""Tests for benchmark stats integration layer."""

from pathlib import Path

import pytest
from analysis.comparison.utils.loader import ConditionResults, TaskReward
from analysis.stats.benchmark_stats import (
    benchmark_ci,
    benchmark_mcnemar,
    benchmark_paired_test,
    significance_stars,
)
from analysis.stats.types import ConfidenceInterval, McNemarResult, PairedTestResult


def _make_task(
    name: str, rewards: list[float], *, errors: list[bool] | None = None
) -> TaskReward:
    n = len(rewards)
    return TaskReward(
        task=name,
        rewards=rewards,
        errors=errors or [False] * n,
        steps=[10] * n,
        costs=[0.01] * n,
    )


def _make_condition(
    label: str, tasks: list[TaskReward], n_runs: int = 3
) -> ConditionResults:
    return ConditionResults(
        label=label,
        runs=[Path(f"/fake/run-{i}") for i in range(n_runs)],
        task_rewards={t.task: t for t in tasks},
    )


@pytest.fixture
def high_cond() -> ConditionResults:
    """Condition where most tasks pass."""
    tasks = [_make_task(f"task-{i}", [1.0, 1.0, 1.0]) for i in range(30)]
    tasks += [_make_task(f"task-{i}", [0.0, 1.0, 0.0]) for i in range(30, 40)]
    return _make_condition("high", tasks)


@pytest.fixture
def low_cond() -> ConditionResults:
    """Condition where most tasks fail."""
    tasks = [_make_task(f"task-{i}", [0.0, 0.0, 0.0]) for i in range(30)]
    tasks += [_make_task(f"task-{i}", [1.0, 0.0, 0.0]) for i in range(30, 40)]
    return _make_condition("low", tasks)


class TestBenchmarkCI:
    def test_returns_confidence_interval(self, high_cond: ConditionResults) -> None:
        ci = benchmark_ci(high_cond, "pass_at", k=1)
        assert isinstance(ci, ConfidenceInterval)

    def test_ci_contains_mean(self, high_cond: ConditionResults) -> None:
        ci = benchmark_ci(high_cond, "pass_at", k=1)
        assert ci.ci_lo <= ci.mean <= ci.ci_hi

    def test_pass_at_vs_pass_pow(self, high_cond: ConditionResults) -> None:
        ci_at = benchmark_ci(high_cond, "pass_at", k=1)
        ci_pow = benchmark_ci(high_cond, "pass_pow", k=3)
        # Pass@1 >= Pass^3 in general.
        assert ci_at.mean >= ci_pow.mean

    def test_unknown_metric_raises(self, high_cond: ConditionResults) -> None:
        with pytest.raises(ValueError, match="Unknown metric"):
            benchmark_ci(high_cond, "invalid_metric")

    def test_mean_steps_ci(self, high_cond: ConditionResults) -> None:
        ci = benchmark_ci(high_cond, "mean_steps", k=1)
        assert ci.mean == pytest.approx(10.0)

    def test_mean_cost_ci(self, high_cond: ConditionResults) -> None:
        ci = benchmark_ci(high_cond, "mean_cost", k=1)
        assert ci.mean == pytest.approx(0.01)


class TestBenchmarkPairedTest:
    def test_same_condition_not_significant(self, high_cond: ConditionResults) -> None:
        result = benchmark_paired_test(high_cond, high_cond, "pass_at", k=1)
        assert isinstance(result, PairedTestResult)
        assert result.observed_diff == 0.0
        assert not result.significant

    def test_different_conditions_significant(
        self, high_cond: ConditionResults, low_cond: ConditionResults
    ) -> None:
        result = benchmark_paired_test(high_cond, low_cond, "pass_at", k=1)
        assert result.observed_diff > 0
        assert result.significant
        assert result.p_value < 0.05


class TestBenchmarkMcNemar:
    def test_returns_mcnemar_result(
        self, high_cond: ConditionResults, low_cond: ConditionResults
    ) -> None:
        result = benchmark_mcnemar(high_cond, low_cond)
        assert isinstance(result, McNemarResult)

    def test_same_condition_not_significant(self, high_cond: ConditionResults) -> None:
        result = benchmark_mcnemar(high_cond, high_cond)
        assert result.n_discordant == 0
        assert not result.significant

    def test_different_conditions_has_discordant(
        self, high_cond: ConditionResults, low_cond: ConditionResults
    ) -> None:
        result = benchmark_mcnemar(high_cond, low_cond)
        assert result.n_discordant > 0


class TestSignificanceStars:
    def test_not_significant(self) -> None:
        assert significance_stars(0.10) == ""

    def test_one_star(self) -> None:
        assert significance_stars(0.03) == "*"

    def test_two_stars(self) -> None:
        assert significance_stars(0.005) == "**"

    def test_three_stars(self) -> None:
        assert significance_stars(0.0001) == "***"

    def test_boundary_values(self) -> None:
        assert significance_stars(0.05) == ""
        assert significance_stars(0.049) == "*"
        assert significance_stars(0.01) == "*"
        assert significance_stars(0.009) == "**"

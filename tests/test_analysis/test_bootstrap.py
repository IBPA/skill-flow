"""Tests for bootstrap CI and paired bootstrap test."""

import pytest
from analysis.stats.bootstrap import bootstrap_ci, paired_bootstrap_test
from analysis.stats.types import ConfidenceInterval, PairedTestResult


class TestBootstrapCI:
    def test_empty_values(self) -> None:
        ci = bootstrap_ci([])
        assert ci.mean == 0.0
        assert ci.ci_lo == 0.0
        assert ci.ci_hi == 0.0

    def test_single_value(self) -> None:
        ci = bootstrap_ci([0.5])
        assert ci.mean == 0.5
        assert ci.ci_lo == 0.5
        assert ci.ci_hi == 0.5

    def test_all_zeros(self) -> None:
        ci = bootstrap_ci([0.0] * 50)
        assert ci.mean == 0.0
        assert ci.ci_lo == 0.0
        assert ci.ci_hi == 0.0

    def test_all_ones(self) -> None:
        ci = bootstrap_ci([1.0] * 50)
        assert ci.mean == 1.0
        assert ci.ci_lo == 1.0
        assert ci.ci_hi == 1.0

    def test_ci_contains_mean(self) -> None:
        values = [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 1.0, 0.5, 0.0]
        ci = bootstrap_ci(values)
        assert ci.ci_lo <= ci.mean <= ci.ci_hi

    def test_ci_bounds_order(self) -> None:
        values = [float(i % 3) / 2 for i in range(64)]
        ci = bootstrap_ci(values)
        assert ci.ci_lo <= ci.ci_hi

    def test_ci_width_decreases_with_n(self) -> None:
        small = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.5] * 2, seed=42)
        large = bootstrap_ci([0.0, 1.0, 0.0, 1.0, 0.5] * 20, seed=42)
        small_width = small.ci_hi - small.ci_lo
        large_width = large.ci_hi - large.ci_lo
        assert large_width < small_width

    def test_seed_reproducibility(self) -> None:
        values = [0.1, 0.4, 0.7, 0.2, 0.9]
        ci1 = bootstrap_ci(values, seed=123)
        ci2 = bootstrap_ci(values, seed=123)
        assert ci1 == ci2

    def test_different_seeds_differ(self) -> None:
        values = [float(i) / 100 for i in range(100)]
        ci1 = bootstrap_ci(values, seed=1)
        ci2 = bootstrap_ci(values, seed=2)
        assert ci1.ci_lo != ci2.ci_lo or ci1.ci_hi != ci2.ci_hi

    def test_returns_confidence_interval(self) -> None:
        ci = bootstrap_ci([0.5, 0.5])
        assert isinstance(ci, ConfidenceInterval)
        assert ci.alpha == 0.05

    def test_custom_alpha(self) -> None:
        values = [float(i % 3) / 2 for i in range(64)]
        ci_95 = bootstrap_ci(values, alpha=0.05)
        ci_99 = bootstrap_ci(values, alpha=0.01)
        # 99% CI should be wider than 95% CI.
        assert (ci_99.ci_hi - ci_99.ci_lo) >= (ci_95.ci_hi - ci_95.ci_lo)


class TestPairedBootstrapTest:
    def test_identical_values_not_significant(self) -> None:
        vals = [0.1, 0.4, 0.7, 0.2, 0.9]
        result = paired_bootstrap_test(vals, vals)
        assert result.observed_diff == 0.0
        assert not result.significant
        assert result.p_value == 1.0

    def test_clearly_different_values_significant(self) -> None:
        a = [1.0] * 50
        b = [0.0] * 50
        result = paired_bootstrap_test(a, b)
        assert result.observed_diff == 1.0
        assert result.significant
        assert result.p_value < 0.001

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            paired_bootstrap_test([0.1, 0.2], [0.1])

    def test_empty_inputs(self) -> None:
        result = paired_bootstrap_test([], [])
        assert result.observed_diff == 0.0
        assert result.p_value == 1.0
        assert not result.significant

    def test_returns_paired_test_result(self) -> None:
        result = paired_bootstrap_test([0.5], [0.3])
        assert isinstance(result, PairedTestResult)
        assert isinstance(result.ci, ConfidenceInterval)

    def test_diff_direction(self) -> None:
        a = [0.8, 0.9, 0.7, 0.85, 0.95]
        b = [0.2, 0.3, 0.1, 0.15, 0.25]
        result = paired_bootstrap_test(a, b)
        assert result.observed_diff > 0
        assert result.ci.ci_lo > 0

    def test_seed_reproducibility(self) -> None:
        a = [0.1, 0.4, 0.7]
        b = [0.2, 0.3, 0.6]
        r1 = paired_bootstrap_test(a, b, seed=42)
        r2 = paired_bootstrap_test(a, b, seed=42)
        assert r1 == r2

    def test_p_value_capped_at_one(self) -> None:
        result = paired_bootstrap_test([0.5, 0.5], [0.5, 0.5])
        assert result.p_value <= 1.0

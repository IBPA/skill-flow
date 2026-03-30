"""Tests for proportion-based statistical tests."""

import math

import pytest
from analysis.stats.proportions import (
    cohens_h,
    holm_bonferroni,
    mcnemar_test,
    wilson_score_interval,
)
from analysis.stats.types import EffectSize, McNemarResult, WilsonInterval


class TestWilsonScoreInterval:
    def test_zero_total(self) -> None:
        wi = wilson_score_interval(0, 0)
        assert wi.proportion == 0.0
        assert wi.n == 0

    def test_all_successes(self) -> None:
        wi = wilson_score_interval(100, 100)
        assert wi.proportion == 1.0
        assert wi.ci_hi <= 1.0
        assert wi.ci_lo > 0.9

    def test_no_successes(self) -> None:
        wi = wilson_score_interval(0, 100)
        assert wi.proportion == 0.0
        assert wi.ci_lo >= 0.0
        assert wi.ci_hi < 0.1

    def test_half_proportion(self) -> None:
        wi = wilson_score_interval(50, 100)
        assert wi.proportion == pytest.approx(0.5)
        assert wi.ci_lo < 0.5 < wi.ci_hi

    def test_interval_contains_proportion(self) -> None:
        wi = wilson_score_interval(30, 100)
        assert wi.ci_lo <= wi.proportion <= wi.ci_hi

    def test_returns_wilson_interval(self) -> None:
        wi = wilson_score_interval(10, 50)
        assert isinstance(wi, WilsonInterval)
        assert wi.n == 50

    def test_bounds_within_0_1(self) -> None:
        for k in (0, 1, 5, 10, 50):
            wi = wilson_score_interval(k, 50)
            assert 0.0 <= wi.ci_lo <= wi.ci_hi <= 1.0

    def test_wider_at_small_n(self) -> None:
        small = wilson_score_interval(5, 10)
        large = wilson_score_interval(50, 100)
        assert (small.ci_hi - small.ci_lo) > (large.ci_hi - large.ci_lo)


class TestMcNemarTest:
    def test_no_discordant_pairs(self) -> None:
        result = mcnemar_test(0, 0)
        assert result.p_value == 1.0
        assert not result.significant
        assert result.n_discordant == 0

    def test_equal_discordant_not_significant(self) -> None:
        result = mcnemar_test(10, 10)
        assert result.p_value > 0.05
        assert not result.significant

    def test_very_unequal_significant(self) -> None:
        result = mcnemar_test(30, 2)
        assert result.p_value < 0.05
        assert result.significant

    def test_returns_mcnemar_result(self) -> None:
        result = mcnemar_test(5, 3)
        assert isinstance(result, McNemarResult)
        assert result.n_discordant == 8

    def test_symmetry(self) -> None:
        r1 = mcnemar_test(10, 5)
        r2 = mcnemar_test(5, 10)
        assert r1.p_value == pytest.approx(r2.p_value, abs=1e-10)
        assert r1.statistic == pytest.approx(r2.statistic, abs=1e-10)


class TestCohensH:
    def test_identical_proportions(self) -> None:
        es = cohens_h(0.5, 0.5)
        assert es.cohens_h == pytest.approx(0.0)
        assert es.interpretation == "negligible"

    def test_small_effect(self) -> None:
        es = cohens_h(0.55, 0.50)
        assert abs(es.cohens_h) < 0.5
        assert es.interpretation in ("negligible", "small")

    def test_large_effect(self) -> None:
        es = cohens_h(0.9, 0.1)
        assert abs(es.cohens_h) > 0.8
        assert es.interpretation == "large"

    def test_returns_effect_size(self) -> None:
        es = cohens_h(0.3, 0.2)
        assert isinstance(es, EffectSize)

    def test_direction(self) -> None:
        es = cohens_h(0.8, 0.2)
        assert es.cohens_h > 0
        es_rev = cohens_h(0.2, 0.8)
        assert es_rev.cohens_h < 0

    def test_zero_and_one(self) -> None:
        es = cohens_h(1.0, 0.0)
        assert es.cohens_h == pytest.approx(math.pi)
        assert es.interpretation == "large"


class TestHolmBonferroni:
    def test_empty(self) -> None:
        assert holm_bonferroni([]) == []

    def test_single_value(self) -> None:
        assert holm_bonferroni([0.03]) == [0.03]

    def test_preserves_order(self) -> None:
        raw = [0.04, 0.01, 0.50]
        adj = holm_bonferroni(raw)
        assert len(adj) == 3
        # Most significant (0.01) should still be smallest after adjustment.
        assert adj[1] <= adj[0] <= adj[2]

    def test_known_values(self) -> None:
        # 3 comparisons: 0.01, 0.04, 0.50
        raw = [0.04, 0.01, 0.50]
        adj = holm_bonferroni(raw)
        # Sorted: 0.01*3=0.03, 0.04*2=0.08, 0.50*1=0.50
        assert adj[1] == pytest.approx(0.03)  # 0.01 * 3
        assert adj[0] == pytest.approx(0.08)  # 0.04 * 2
        assert adj[2] == pytest.approx(0.50)  # 0.50 * 1

    def test_monotonicity(self) -> None:
        raw = [0.001, 0.10, 0.02]
        adj = holm_bonferroni(raw)
        # Sorted by raw: 0.001, 0.02, 0.10
        # Adjusted: 0.003, 0.04, 0.10 (monotonically non-decreasing)
        sorted_adj = sorted(zip(raw, adj, strict=True), key=lambda x: x[0])
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i][1] >= sorted_adj[i - 1][1]

    def test_capped_at_one(self) -> None:
        raw = [0.80, 0.90]
        adj = holm_bonferroni(raw)
        assert all(a <= 1.0 for a in adj)

    def test_all_significant_stay_significant(self) -> None:
        raw = [0.001, 0.002, 0.003]
        adj = holm_bonferroni(raw)
        assert all(a < 0.05 for a in adj)

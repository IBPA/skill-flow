"""Tests for src.eval.metrics."""

import pytest
from skill_flow.eval.utils.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_perfect_recall(self):
        retrieved = ["a", "b", "c"]
        gt = ["a", "b"]
        assert recall_at_k(retrieved, gt, 3) == 1.0

    def test_partial_recall(self):
        retrieved = ["a", "x", "y"]
        gt = ["a", "b"]
        assert recall_at_k(retrieved, gt, 3) == 0.5

    def test_no_recall(self):
        retrieved = ["x", "y", "z"]
        gt = ["a", "b"]
        assert recall_at_k(retrieved, gt, 3) == 0.0

    def test_k_limits_search(self):
        retrieved = ["x", "a", "b"]
        gt = ["a", "b"]
        assert recall_at_k(retrieved, gt, 1) == 0.0
        assert recall_at_k(retrieved, gt, 2) == 0.5
        assert recall_at_k(retrieved, gt, 3) == 1.0

    def test_empty_gt(self):
        assert recall_at_k(["a", "b"], [], 5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["a"], 5) == 0.0


class TestPrecisionAtK:
    def test_perfect_precision(self):
        retrieved = ["a", "b", "c"]
        gt = ["a", "b", "c"]
        assert precision_at_k(retrieved, gt, 3) == 1.0

    def test_partial_precision(self):
        retrieved = ["a", "x", "y"]
        gt = ["a", "b"]
        assert precision_at_k(retrieved, gt, 3) == pytest.approx(1.0 / 3)

    def test_no_precision(self):
        retrieved = ["x", "y", "z"]
        gt = ["a", "b"]
        assert precision_at_k(retrieved, gt, 3) == 0.0

    def test_k_limits_search(self):
        retrieved = ["a", "b", "x"]
        gt = ["a", "b"]
        assert precision_at_k(retrieved, gt, 1) == 1.0
        assert precision_at_k(retrieved, gt, 2) == 1.0
        assert precision_at_k(retrieved, gt, 3) == pytest.approx(2.0 / 3)

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], 5) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], ["a"], 0) == 0.0


class TestHitAtK:
    def test_hit(self):
        retrieved = ["x", "a", "y"]
        gt = ["a"]
        assert hit_at_k(retrieved, gt, 3) == 1.0

    def test_miss(self):
        retrieved = ["x", "y", "z"]
        gt = ["a"]
        assert hit_at_k(retrieved, gt, 3) == 0.0

    def test_k_limits(self):
        retrieved = ["x", "a"]
        gt = ["a"]
        assert hit_at_k(retrieved, gt, 1) == 0.0
        assert hit_at_k(retrieved, gt, 2) == 1.0

    def test_empty_gt(self):
        assert hit_at_k(["a"], [], 5) == 0.0

    def test_empty_retrieved(self):
        assert hit_at_k([], ["a"], 5) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1.0 / 3)

    def test_not_found(self):
        assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_multiple_gt_first_wins(self):
        assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5

    def test_empty_gt(self):
        assert reciprocal_rank(["a"], []) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], ["a"]) == 0.0

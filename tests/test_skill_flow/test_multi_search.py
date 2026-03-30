"""Tests for skill_flow.retriever.multi_search."""

from unittest.mock import MagicMock

from skill_flow.retriever.multi_search import (
    collect_multi_scores,
    rank_from_scores,
    search_multi,
    union_from_collected,
)
from skill_flow.retriever.retriever import SearchResult


def _make_result(key: str, score: float) -> SearchResult:
    return SearchResult(key=key, score=score, description=f"desc-{key}")


def _make_searcher(results_per_query: list[list[SearchResult]]) -> MagicMock:
    """Build a mock searcher that returns different results per call."""
    searcher = MagicMock()
    searcher.search.side_effect = results_per_query
    return searcher


class TestSingleQuery:
    def test_delegates_to_searcher(self) -> None:
        expected = [_make_result("a", 0.9), _make_result("b", 0.5)]
        searcher = _make_searcher([expected])
        results = search_multi(searcher, ["test query"], top_k=10)
        assert results == expected
        searcher.search.assert_called_once_with("test query", top_k=10)


class TestMultiQuery:
    def test_merges_results_max(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.3)]
        q2 = [_make_result("b", 0.8), _make_result("c", 0.7)]
        searcher = _make_searcher([q1, q2])

        results = search_multi(searcher, ["q1", "q2"], top_k=10, aggregation="max")

        keys = [r.key for r in results]
        assert "a" in keys
        assert "b" in keys
        assert "c" in keys
        # "a" max score = 0.9, "b" max score = 0.8, "c" max score = 0.7
        assert results[0].key == "a"
        assert results[0].score == 0.9
        assert results[1].key == "b"
        assert results[1].score == 0.8

    def test_merges_results_mean(self) -> None:
        q1 = [_make_result("a", 1.0), _make_result("b", 0.4)]
        q2 = [_make_result("a", 0.0), _make_result("b", 0.6)]
        searcher = _make_searcher([q1, q2])

        results = search_multi(searcher, ["q1", "q2"], top_k=10, aggregation="mean")

        score_map = {r.key: r.score for r in results}
        assert score_map["a"] == 0.5
        assert score_map["b"] == 0.5

    def test_merges_results_rrf(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.3)]
        q2 = [_make_result("b", 0.8), _make_result("a", 0.2)]
        searcher = _make_searcher([q1, q2])

        results = search_multi(searcher, ["q1", "q2"], top_k=10, aggregation="rrf")

        # Both keys get RRF scores from both queries
        score_map = {r.key: r.score for r in results}
        assert score_map["a"] > 0
        assert score_map["b"] > 0

    def test_top_k_truncation(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.5)]
        q2 = [_make_result("c", 0.8), _make_result("d", 0.3)]
        searcher = _make_searcher([q1, q2])

        results = search_multi(searcher, ["q1", "q2"], top_k=2, aggregation="max")

        assert len(results) == 2

    def test_carries_query_scores(self) -> None:
        q1 = [_make_result("a", 0.9)]
        q2 = [_make_result("a", 0.5)]
        searcher = _make_searcher([q1, q2])

        results = search_multi(searcher, ["q1", "q2"], top_k=10, aggregation="max")

        assert results[0].query_scores == [0.9, 0.5]

    def test_missing_key_gets_zero_score(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.5)]
        q2 = [_make_result("a", 0.3)]  # "b" not in q2
        searcher = _make_searcher([q1, q2])

        results = search_multi(searcher, ["q1", "q2"], top_k=10, aggregation="mean")

        score_map = {r.key: r.score for r in results}
        # b: mean of [0.5, 0.0] = 0.25
        assert score_map["b"] == 0.25


class TestCollectMultiScores:
    def test_collects_per_key_scores(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.3)]
        q2 = [_make_result("b", 0.8), _make_result("c", 0.7)]
        searcher = _make_searcher([q1, q2])

        collected = collect_multi_scores(searcher, ["q1", "q2"])

        assert collected.per_key_scores["a"] == [0.9, 0.0]
        assert collected.per_key_scores["b"] == [0.3, 0.8]
        assert collected.per_key_scores["c"] == [0.0, 0.7]
        assert collected.queries == ["q1", "q2"]

    def test_stores_per_query_results(self) -> None:
        q1 = [_make_result("a", 0.9)]
        q2 = [_make_result("b", 0.8)]
        searcher = _make_searcher([q1, q2])

        collected = collect_multi_scores(searcher, ["q1", "q2"])

        assert len(collected.per_query_results) == 2
        assert collected.per_query_results[0][0].key == "a"
        assert collected.per_query_results[1][0].key == "b"

    def test_metadata_from_first_occurrence(self) -> None:
        q1 = [SearchResult(key="a", score=0.9, description="first")]
        q2 = [SearchResult(key="a", score=0.5, description="second")]
        searcher = _make_searcher([q1, q2])

        collected = collect_multi_scores(searcher, ["q1", "q2"])

        assert collected.metadata["a"].description == "first"


class TestUnionFromCollected:
    def test_dedupes_and_sorts_by_max_score(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.3)]
        q2 = [_make_result("b", 0.8), _make_result("c", 0.7)]
        searcher = _make_searcher([q1, q2])
        collected = collect_multi_scores(searcher, ["q1", "q2"])

        results = union_from_collected(collected, top_k_per_query=2)

        keys = [r.key for r in results]
        assert keys == ["a", "b", "c"]
        assert results[0].score == 0.9
        assert results[1].score == 0.8  # max(0.3, 0.8)

    def test_top_k_per_query_limits_per_query(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.3)]
        q2 = [_make_result("c", 0.8), _make_result("d", 0.7)]
        searcher = _make_searcher([q1, q2])
        collected = collect_multi_scores(searcher, ["q1", "q2"])

        results = union_from_collected(collected, top_k_per_query=1)

        # Only top-1 from each query: a (0.9) and c (0.8)
        keys = {r.key for r in results}
        assert keys == {"a", "c"}

    def test_preserves_description_from_metadata(self) -> None:
        q1 = [SearchResult(key="a", score=0.9, description="desc-a")]
        q2 = [SearchResult(key="a", score=0.5, description="other")]
        searcher = _make_searcher([q1, q2])
        collected = collect_multi_scores(searcher, ["q1", "q2"])

        results = union_from_collected(collected, top_k_per_query=10)

        assert results[0].description == "desc-a"


class TestRankFromScores:
    def test_aggregates_and_truncates(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.3)]
        q2 = [_make_result("b", 0.8), _make_result("c", 0.7)]
        searcher = _make_searcher([q1, q2])
        collected = collect_multi_scores(searcher, ["q1", "q2"])

        results = rank_from_scores(collected, "max", top_k=2)

        assert len(results) == 2
        assert results[0].key == "a"
        assert results[0].score == 0.9

    def test_carries_query_scores(self) -> None:
        q1 = [_make_result("a", 0.9)]
        q2 = [_make_result("a", 0.5)]
        searcher = _make_searcher([q1, q2])
        collected = collect_multi_scores(searcher, ["q1", "q2"])

        results = rank_from_scores(collected, "max", top_k=10)

        assert results[0].query_scores == [0.9, 0.5]

    def test_different_aggregations_differ(self) -> None:
        q1 = [_make_result("a", 0.9), _make_result("b", 0.1)]
        q2 = [_make_result("a", 0.1), _make_result("b", 0.9)]
        searcher = _make_searcher([q1, q2])
        collected = collect_multi_scores(searcher, ["q1", "q2"])

        max_results = rank_from_scores(collected, "max", top_k=10)
        mean_results = rank_from_scores(collected, "mean", top_k=10)

        max_scores = {r.key: r.score for r in max_results}
        mean_scores = {r.key: r.score for r in mean_results}
        # max: both get 0.9, mean: both get 0.5
        assert max_scores["a"] == 0.9
        assert mean_scores["a"] == 0.5

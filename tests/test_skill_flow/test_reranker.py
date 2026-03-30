"""Tests for src.rerank.reranker."""

from unittest.mock import MagicMock

import numpy as np
from skill_flow.config import DeepRerankerConfig, RerankerConfig
from skill_flow.reranker.reranker import Reranker, aggregate_scores
from skill_flow.retriever.retriever import SearchResult


def _make_candidates(n: int, *, with_content: bool = False) -> list[SearchResult]:
    return [
        SearchResult(
            key=f"skill-{i}",
            score=float(n - i),
            description=f"desc {i}",
            content=f"full content {i}" if with_content else "",
        )
        for i in range(n)
    ]


def _make_reranker(scores: list[float]) -> Reranker:
    """Create a Reranker with a mock model returning predetermined scores."""
    config = RerankerConfig(enabled=True)
    reranker = Reranker(config)
    reranker._model = MagicMock()
    reranker._model.predict.return_value = np.array(scores, dtype=np.float32)
    return reranker


class TestReranker:
    def test_rerank_reorders_by_score(self):
        candidates = _make_candidates(3, with_content=True)
        reranker = _make_reranker([0.1, 0.9, 0.5])

        results = reranker.rerank("query", candidates, top_k=3)

        assert [r.key for r in results] == ["skill-1", "skill-2", "skill-0"]

    def test_rerank_truncates_to_top_k(self):
        candidates = _make_candidates(5, with_content=True)
        reranker = _make_reranker([0.5, 0.4, 0.3, 0.2, 0.1])

        results = reranker.rerank("query", candidates, top_k=2)

        assert len(results) == 2

    def test_rerank_returns_all_without_top_k(self):
        reranker = _make_reranker([0.5, 0.4, 0.3])

        candidates = _make_candidates(3, with_content=True)
        results = reranker.rerank("query", candidates)

        assert len(results) == 3

    def test_rerank_empty_candidates(self):
        reranker = _make_reranker([])

        results = reranker.rerank("query", [])

        assert results == []

    def test_rerank_builds_correct_pairs(self):
        candidates = _make_candidates(2, with_content=True)
        reranker = _make_reranker([0.5, 0.3])

        reranker.rerank("my query", candidates, top_k=2)

        reranker._model.predict.assert_called_once()
        pairs = reranker._model.predict.call_args[0][0]
        assert pairs == [
            ["my query", "full content 0"],
            ["my query", "full content 1"],
        ]

    def test_rerank_uses_content_when_available(self):
        candidates = _make_candidates(2, with_content=True)
        reranker = _make_reranker([0.5, 0.3])

        reranker.rerank("my query", candidates, top_k=2)

        pairs = reranker._model.predict.call_args[0][0]
        assert pairs == [
            ["my query", "full content 0"],
            ["my query", "full content 1"],
        ]

    def test_rerank_assigns_below_min_score_to_empty_content(self):
        candidates = [
            SearchResult(key="a", score=1.0, description="desc a", content="content a"),
            SearchResult(key="b", score=0.5, description="desc b", content=""),
        ]
        reranker = _make_reranker([0.3])

        results = reranker.rerank("query", candidates, top_k=2)

        # Only the candidate with content is scored by the model
        pairs = reranker._model.predict.call_args[0][0]
        assert pairs == [["query", "content a"]]
        # Empty-content candidate scores below the lowest model score
        result_a = next(r for r in results if r.key == "a")
        result_b = next(r for r in results if r.key == "b")
        assert result_b.score < result_a.score
        assert results[0].key == "a"

    def test_rerank_preserves_descriptions(self):
        candidates = _make_candidates(2, with_content=True)
        reranker = _make_reranker([0.3, 0.9])

        results = reranker.rerank("query", candidates, top_k=2)

        assert results[0].description == "desc 1"
        assert results[1].description == "desc 0"

    def test_rerank_preserves_content(self):
        candidates = _make_candidates(2, with_content=True)
        reranker = _make_reranker([0.3, 0.9])

        results = reranker.rerank("query", candidates, top_k=2)

        assert results[0].content == "full content 1"
        assert results[1].content == "full content 0"

    def test_rerank_all_empty_content_skips_model(self):
        candidates = _make_candidates(2)
        reranker = _make_reranker([])

        results = reranker.rerank("query", candidates, top_k=2)

        reranker._model.predict.assert_not_called()
        assert all(r.score == -1.0 for r in results)

    def test_rerank_truncates_content(self):
        config = RerankerConfig(enabled=True, max_content_chars=10)
        reranker = Reranker(config)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([0.5], dtype=np.float32)

        candidates = [
            SearchResult(key="a", score=1.0, description="d", content="x" * 100),
        ]
        reranker.rerank("query", candidates, top_k=1)

        pairs = reranker._model.predict.call_args[0][0]
        assert pairs == [["query", "x" * 10]]

    def test_rerank_accepts_deep_reranker_config(self):
        config = DeepRerankerConfig(enabled=True)
        reranker = Reranker(config)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([0.8], dtype=np.float32)

        candidates = [
            SearchResult(key="a", score=1.0, description="d", content="x" * 100),
        ]
        results = reranker.rerank("query", candidates, top_k=1)

        pairs = reranker._model.predict.call_args[0][0]
        assert pairs == [["query", "x" * 100]]
        assert len(results) == 1

    def test_input_top_k_truncates_candidates(self):
        config = RerankerConfig(enabled=True, input_top_k=2)
        reranker = Reranker(config)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array([0.5, 0.3], dtype=np.float32)

        candidates = _make_candidates(5, with_content=True)
        results = reranker.rerank("query", candidates)

        # Only the first 2 candidates should be scored
        pairs = reranker._model.predict.call_args[0][0]
        assert len(pairs) == 2
        assert len(results) == 2

    def test_input_top_k_zero_uses_all(self):
        candidates = _make_candidates(3, with_content=True)
        reranker = _make_reranker([0.1, 0.9, 0.5])

        results = reranker.rerank("query", candidates, top_k=3)

        assert len(results) == 3


class TestRerankMulti:
    def test_single_query_delegates_to_rerank(self):
        candidates = _make_candidates(3, with_content=True)
        reranker = _make_reranker([0.1, 0.9, 0.5])

        results = reranker.rerank_multi(["query"], candidates, top_k=3)

        assert [r.key for r in results] == ["skill-1", "skill-2", "skill-0"]

    def test_empty_candidates(self):
        reranker = _make_reranker([])

        results = reranker.rerank_multi(["q1", "q2"], [], top_k=5)

        assert results == []

    def test_max_aggregation(self):
        """With max, each candidate keeps its best score across queries."""
        config = RerankerConfig(enabled=True)
        reranker = Reranker(config)
        reranker._model = MagicMock()

        # Query 1: skill-0=0.1, skill-1=0.9, skill-2=0.5
        # Query 2: skill-0=0.8, skill-1=0.2, skill-2=0.6
        # Max:     skill-0=0.8, skill-1=0.9, skill-2=0.6
        reranker._model.predict.side_effect = [
            np.array([0.1, 0.9, 0.5], dtype=np.float32),
            np.array([0.8, 0.2, 0.6], dtype=np.float32),
        ]

        candidates = _make_candidates(3, with_content=True)
        results = reranker.rerank_multi(
            ["q1", "q2"],
            candidates,
            top_k=3,
            aggregation="max",
        )

        assert [r.key for r in results] == ["skill-1", "skill-0", "skill-2"]
        assert abs(results[0].score - 0.9) < 1e-6
        assert abs(results[1].score - 0.8) < 1e-6

    def test_mean_aggregation(self):
        """With mean, each candidate gets the average score across queries."""
        config = RerankerConfig(enabled=True)
        reranker = Reranker(config)
        reranker._model = MagicMock()

        # Query 1: skill-0=0.2, skill-1=0.8
        # Query 2: skill-0=0.6, skill-1=0.4
        # Mean:    skill-0=0.4, skill-1=0.6
        reranker._model.predict.side_effect = [
            np.array([0.2, 0.8], dtype=np.float32),
            np.array([0.6, 0.4], dtype=np.float32),
        ]

        candidates = _make_candidates(2, with_content=True)
        results = reranker.rerank_multi(
            ["q1", "q2"],
            candidates,
            top_k=2,
            aggregation="mean",
        )

        assert results[0].key == "skill-1"
        assert abs(results[0].score - 0.6) < 1e-6
        assert results[1].key == "skill-0"
        assert abs(results[1].score - 0.4) < 1e-6

    def test_rrf_aggregation(self):
        """With RRF, scores are based on reciprocal rank fusion."""
        config = RerankerConfig(enabled=True)
        reranker = Reranker(config)
        reranker._model = MagicMock()

        # Query 1 ranks: skill-1 (rank 1), skill-0 (rank 2)
        # Query 2 ranks: skill-0 (rank 1), skill-1 (rank 2)
        # RRF: skill-0 = 1/62 + 1/61, skill-1 = 1/61 + 1/62 => equal
        reranker._model.predict.side_effect = [
            np.array([0.1, 0.9], dtype=np.float32),
            np.array([0.9, 0.1], dtype=np.float32),
        ]

        candidates = _make_candidates(2, with_content=True)
        results = reranker.rerank_multi(
            ["q1", "q2"],
            candidates,
            top_k=2,
            aggregation="rrf",
        )

        assert len(results) == 2
        # Both should have same RRF score (1/61 + 1/62)
        assert abs(results[0].score - results[1].score) < 1e-6

    def test_multi_respects_top_k(self):
        config = RerankerConfig(enabled=True)
        reranker = Reranker(config)
        reranker._model = MagicMock()

        reranker._model.predict.side_effect = [
            np.array([0.1, 0.9, 0.5], dtype=np.float32),
            np.array([0.8, 0.2, 0.6], dtype=np.float32),
        ]

        candidates = _make_candidates(3, with_content=True)
        results = reranker.rerank_multi(
            ["q1", "q2"],
            candidates,
            top_k=2,
            aggregation="max",
        )

        assert len(results) == 2

    def test_multi_input_top_k_truncates_candidates(self):
        config = RerankerConfig(enabled=True, input_top_k=2)
        reranker = Reranker(config)
        reranker._model = MagicMock()

        # Only 2 candidates after input_top_k truncation
        reranker._model.predict.side_effect = [
            np.array([0.1, 0.9], dtype=np.float32),
            np.array([0.8, 0.2], dtype=np.float32),
        ]

        candidates = _make_candidates(5, with_content=True)
        results = reranker.rerank_multi(
            ["q1", "q2"],
            candidates,
            aggregation="max",
        )

        assert len(results) == 2

    def test_multi_stores_query_scores(self):
        """Multi-query reranking stores per-query scores on each result."""
        config = RerankerConfig(enabled=True)
        reranker = Reranker(config)
        reranker._model = MagicMock()

        # Query 1: skill-0=0.1, skill-1=0.9
        # Query 2: skill-0=0.8, skill-1=0.2
        reranker._model.predict.side_effect = [
            np.array([0.1, 0.9], dtype=np.float32),
            np.array([0.8, 0.2], dtype=np.float32),
        ]

        candidates = _make_candidates(2, with_content=True)
        results = reranker.rerank_multi(
            ["q1", "q2"],
            candidates,
            aggregation="max",
        )

        r0 = next(r for r in results if r.key == "skill-0")
        r1 = next(r for r in results if r.key == "skill-1")
        assert r0.query_scores == [float(np.float32(0.1)), float(np.float32(0.8))]
        assert r1.query_scores == [float(np.float32(0.9)), float(np.float32(0.2))]

    def test_single_query_has_empty_query_scores(self):
        """Single-query delegation does not populate query_scores."""
        candidates = _make_candidates(2, with_content=True)
        reranker = _make_reranker([0.5, 0.3])

        results = reranker.rerank_multi(["q1"], candidates)

        assert all(r.query_scores == [] for r in results)


class TestAggregateScores:
    def test_max(self):
        per_key = {"a": [0.1, 0.8], "b": [0.9, 0.2]}
        agg = aggregate_scores(per_key, "max")
        assert abs(agg["a"] - 0.8) < 1e-6
        assert abs(agg["b"] - 0.9) < 1e-6

    def test_mean(self):
        per_key = {"a": [0.2, 0.6], "b": [0.8, 0.4]}
        agg = aggregate_scores(per_key, "mean")
        assert abs(agg["a"] - 0.4) < 1e-6
        assert abs(agg["b"] - 0.6) < 1e-6

    def test_rrf(self):
        # a ranks: [2nd, 1st], b ranks: [1st, 2nd]
        per_key = {"a": [0.1, 0.9], "b": [0.9, 0.1]}
        agg = aggregate_scores(per_key, "rrf")
        # Both get 1/61 + 1/62 => equal
        assert abs(agg["a"] - agg["b"]) < 1e-6

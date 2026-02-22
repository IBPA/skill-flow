"""Tests for src.rerank.reranker."""

from unittest.mock import MagicMock

import numpy as np
from skill_flow.config import Reranker2Config, RerankerConfig
from skill_flow.reranker.reranker import Reranker
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

    def test_rerank_uses_config_top_k(self):
        config = RerankerConfig(enabled=True, top_k=2)
        reranker = Reranker(config)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = np.array(
            [0.5, 0.4, 0.3], dtype=np.float32
        )

        candidates = _make_candidates(3, with_content=True)
        results = reranker.rerank("query", candidates)

        assert len(results) == 2

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

    def test_rerank_assigns_zero_score_to_empty_content(self):
        candidates = [
            SearchResult(key="a", score=1.0, description="desc a", content="content a"),
            SearchResult(key="b", score=0.5, description="desc b", content=""),
        ]
        reranker = _make_reranker([0.3])

        results = reranker.rerank("query", candidates, top_k=2)

        # Only the candidate with content is scored by the model
        pairs = reranker._model.predict.call_args[0][0]
        assert pairs == [["query", "content a"]]
        # Empty-content candidate gets score 0
        result_b = next(r for r in results if r.key == "b")
        assert result_b.score == 0.0

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
        assert all(r.score == 0.0 for r in results)

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

    def test_rerank_accepts_reranker2_config(self):
        config = Reranker2Config(enabled=True)
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

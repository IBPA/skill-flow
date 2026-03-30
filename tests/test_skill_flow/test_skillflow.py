"""Tests for src.skillflow."""

from unittest.mock import MagicMock

from skill_flow.models.core import SkillFlow
from skill_flow.retriever.retriever import SearchResult


def _make_results(n: int) -> list[SearchResult]:
    return [
        SearchResult(key=f"skill-{i}", score=float(n - i), description=f"desc {i}")
        for i in range(n)
    ]


class TestSkillFlow:
    def test_without_reranker_delegates_to_searcher(self):
        searcher = MagicMock()
        expected = _make_results(3)
        searcher.search.return_value = expected

        retriever = SkillFlow(searcher)
        results = retriever.search("query", top_k=3)

        assert results == expected
        searcher.search.assert_called_once_with("query", top_k=3)

    def test_with_reranker_calls_both_stages(self):
        searcher = MagicMock()
        candidates = _make_results(5)
        searcher.search.return_value = candidates

        reranker = MagicMock()
        reranked = _make_results(2)
        reranker.rerank.return_value = reranked

        retriever = SkillFlow(searcher, reranker)
        results = retriever.search("query", top_k=2)

        assert results == reranked
        searcher.search.assert_called_once_with("query")
        reranker.rerank.assert_called_once_with("query", candidates, top_k=2)

    def test_with_reranker_no_top_k(self):
        searcher = MagicMock()
        searcher.search.return_value = _make_results(3)

        reranker = MagicMock()
        reranker.rerank.return_value = _make_results(2)

        retriever = SkillFlow(searcher, reranker)
        retriever.search("query")

        reranker.rerank.assert_called_once_with(
            "query", searcher.search.return_value, top_k=None
        )

    def test_with_both_rerankers_calls_three_stages(self):
        searcher = MagicMock()
        candidates = _make_results(10)
        searcher.search.return_value = candidates

        reranker = MagicMock()
        stage2_results = _make_results(5)
        reranker.rerank.return_value = stage2_results

        deep_reranker = MagicMock()
        final_results = _make_results(2)
        deep_reranker.rerank.return_value = final_results

        retriever = SkillFlow(searcher, reranker, deep_reranker)
        results = retriever.search("query", top_k=2)

        assert results == final_results
        searcher.search.assert_called_once_with("query")
        reranker.rerank.assert_called_once_with("query", candidates)
        deep_reranker.rerank.assert_called_once_with("query", stage2_results, top_k=2)

    def test_with_deep_reranker_but_no_reranker_skips_both(self):
        searcher = MagicMock()
        expected = _make_results(3)
        searcher.search.return_value = expected

        deep_reranker = MagicMock()

        retriever = SkillFlow(searcher, reranker=None, deep_reranker=deep_reranker)
        results = retriever.search("query", top_k=3)

        assert results == expected
        searcher.search.assert_called_once_with("query", top_k=3)
        deep_reranker.rerank.assert_not_called()

    def test_with_full_pipeline_calls_four_stages(self):
        searcher = MagicMock()
        candidates = _make_results(10)
        searcher.search.return_value = candidates

        reranker = MagicMock()
        stage2_results = _make_results(5)
        reranker.rerank.return_value = stage2_results

        deep_reranker = MagicMock()
        stage3_results = _make_results(3)
        deep_reranker.rerank.return_value = stage3_results

        selector = MagicMock()
        final_results = _make_results(1)
        selector.select.return_value = final_results

        retriever = SkillFlow(searcher, reranker, deep_reranker, selector)
        results = retriever.search("query", top_k=3)

        assert results == final_results
        searcher.search.assert_called_once_with("query")
        reranker.rerank.assert_called_once_with("query", candidates)
        deep_reranker.rerank.assert_called_once_with("query", stage2_results, top_k=3)
        selector.select.assert_called_once_with("query", stage3_results)

    def test_selector_without_reranker_is_ignored(self):
        searcher = MagicMock()
        expected = _make_results(3)
        searcher.search.return_value = expected

        selector = MagicMock()

        retriever = SkillFlow(searcher, selector=selector)
        results = retriever.search("query", top_k=3)

        assert results == expected
        selector.select.assert_not_called()

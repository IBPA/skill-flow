"""SkillFlow facade — composes FAISS search with optional reranking and selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skill_flow.reranker.reranker import Reranker
    from skill_flow.retriever.retriever import IndexSearcher, SearchResult
    from skill_flow.selector.selector import Selector


class SkillFlow:
    """Multi-stage retriever: FAISS search + optional rerankers + LLM selector."""

    def __init__(
        self,
        searcher: IndexSearcher,
        reranker: Reranker | None = None,
        reranker2: Reranker | None = None,
        selector: Selector | None = None,
    ) -> None:
        self._searcher = searcher
        self._reranker = reranker
        self._reranker2 = reranker2
        self._selector = selector

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Retrieve skills for *query*.

        Without rerankers, delegates directly to the FAISS searcher.
        With reranker only, fetches the searcher's configured ``top_k``
        candidates and re-ranks them down to *top_k*.
        With both rerankers, chains: searcher -> reranker -> reranker2.
        With selector, applies LLM filtering after the final reranker stage.
        """
        if self._reranker is None:
            return self._searcher.search(query, top_k=top_k)

        candidates = self._searcher.search(query)
        if self._reranker2 is None:
            results = self._reranker.rerank(query, candidates, top_k=top_k)
        else:
            reranked = self._reranker.rerank(query, candidates)
            results = self._reranker2.rerank(query, reranked, top_k=top_k)

        if self._selector is not None:
            results = self._selector.select(query, results)

        return results

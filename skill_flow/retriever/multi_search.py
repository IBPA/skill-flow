"""Multi-query search orchestration for Stage 1 retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from skill_flow.reranker.reranker import aggregate_scores
from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from skill_flow.retriever.protocol import Searcher


class CollectedScores(NamedTuple):
    """Raw scores from multi-query search, before aggregation."""

    per_key_scores: dict[str, list[float]]
    metadata: dict[str, SearchResult]
    per_query_results: list[list[SearchResult]]
    queries: list[str]


def collect_multi_scores(
    searcher: Searcher,
    queries: list[str],
) -> CollectedScores:
    """Search the full index for each query and collect per-key scores.

    Returns raw data suitable for repeated aggregation via
    :func:`rank_from_scores`.
    """
    _ALL = 10**9  # effectively unlimited; searchers clamp to index size
    per_query_results = [searcher.search(q, top_k=_ALL) for q in queries]

    per_key_scores: dict[str, list[float]] = {}
    metadata: dict[str, SearchResult] = {}
    for qi, ranked in enumerate(per_query_results):
        seen: set[str] = set()
        for r in ranked:
            if r.key not in metadata:
                metadata[r.key] = r
            if r.key not in per_key_scores:
                per_key_scores[r.key] = [0.0] * qi
            per_key_scores[r.key].append(r.score)
            seen.add(r.key)
        for key in list(per_key_scores):
            if key not in seen:
                per_key_scores[key].append(0.0)

    return CollectedScores(
        per_key_scores=per_key_scores,
        metadata=metadata,
        per_query_results=per_query_results,
        queries=queries,
    )


def rank_from_scores(
    collected: CollectedScores,
    aggregation: str,
    top_k: int,
) -> list[SearchResult]:
    """Aggregate collected per-key scores and return the top-k results."""
    aggregated = aggregate_scores(collected.per_key_scores, aggregation)

    merged = [
        SearchResult(
            key=key,
            score=score,
            description=collected.metadata[key].description,
            content=collected.metadata[key].content,
            query_scores=collected.per_key_scores[key],
        )
        for key, score in aggregated.items()
    ]
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged[:top_k]


def union_from_collected(
    collected: CollectedScores,
    top_k_per_query: int,
) -> list[SearchResult]:
    """Take top-k from each query, dedupe, sort by max retrieval score."""
    seen: dict[str, float] = {}
    for qr in collected.per_query_results:
        for r in qr[:top_k_per_query]:
            if r.key not in seen or r.score > seen[r.key]:
                seen[r.key] = r.score
    merged = [
        SearchResult(
            key=key,
            score=score,
            description=collected.metadata[key].description,
            content=collected.metadata[key].content,
        )
        for key, score in seen.items()
    ]
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged


def search_multi(
    searcher: Searcher,
    queries: list[str],
    top_k: int,
    aggregation: str = "max",
) -> list[SearchResult]:
    """Search with one or more queries, merging results via aggregation.

    For a single query, delegates directly to ``searcher.search()``.
    For multiple queries, calls :func:`collect_multi_scores` then
    :func:`rank_from_scores`.
    """
    if len(queries) == 1:
        return searcher.search(queries[0], top_k=top_k)

    collected = collect_multi_scores(searcher, queries)
    return rank_from_scores(collected, aggregation, top_k)

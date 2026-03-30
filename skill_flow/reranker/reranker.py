"""Cross-encoder reranker for Stage 2 of the retrieval pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from skill_flow.index.encoder import pick_device
from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from skill_flow.config import DeepRerankerConfig, RerankerConfig

logger = logging.getLogger(__name__)


def aggregate_scores(
    per_key_scores: dict[str, list[float]],
    aggregation: str,
) -> dict[str, float]:
    """Aggregate per-query scores into a single score per candidate.

    Supports ``"max"``, ``"mean"``, and ``"rrf"`` strategies.  Can be
    called post-hoc on stored ``query_scores`` to compute alternative
    aggregations without re-running cross-encoder inference.
    """
    if aggregation == "rrf":
        n_queries = len(next(iter(per_key_scores.values())))
        agg: dict[str, float] = dict.fromkeys(per_key_scores, 0.0)
        for qi in range(n_queries):

            def _sort_key(k: str, _qi: int = qi) -> float:
                return per_key_scores[k][_qi]

            sorted_keys = sorted(per_key_scores, key=_sort_key, reverse=True)
            for rank, key in enumerate(sorted_keys, start=1):
                agg[key] += 1.0 / (60 + rank)
        return agg
    if aggregation == "mean":
        return {k: sum(v) / len(v) for k, v in per_key_scores.items()}
    return {k: max(v) for k, v in per_key_scores.items()}


class Reranker:
    """Re-scores search results using a cross-encoder model."""

    def __init__(self, config: RerankerConfig | DeepRerankerConfig) -> None:
        self._config = config
        device = pick_device()
        logger.info("Loading cross-encoder model: %s on %s", config.model_name, device)
        self._model = CrossEncoder(config.model_name, device=device)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars]
        return text

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Re-rank candidates by cross-encoder score.

        Builds ``(query, description)`` pairs, scores them with the
        cross-encoder, and returns the top results sorted by the new score.
        """
        if not candidates:
            return []

        input_k = self._config.input_top_k
        if input_k > 0:
            candidates = candidates[:input_k]

        max_chars = self._config.max_content_chars

        scorable = [c for c in candidates if c.content]
        empty = [c for c in candidates if not c.content]

        if scorable:
            pairs = [
                [query, self._truncate(c.content or "", max_chars)] for c in scorable
            ]
            scores: list[float] = self._model.predict(
                pairs, batch_size=self._config.batch_size
            ).tolist()
        else:
            scores = []

        reranked = [
            SearchResult(
                key=c.key,
                score=float(s),
                description=c.description,
                content=c.content,
            )
            for c, s in zip(scorable, scores, strict=True)
        ]
        min_score = min(scores) if scores else 0.0
        empty_score = min(min_score - 1.0, -1.0)
        reranked.extend(
            SearchResult(
                key=c.key,
                score=empty_score,
                description=c.description,
                content=c.content,
            )
            for c in empty
        )
        reranked.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            return reranked[:top_k]
        return reranked

    def rerank_multi(
        self,
        queries: list[str],
        candidates: list[SearchResult],
        top_k: int | None = None,
        aggregation: str = "max",
    ) -> list[SearchResult]:
        """Re-rank candidates using multiple queries with score aggregation.

        For each query, all candidates are scored by the cross-encoder.
        Per-candidate scores are stored in ``query_scores`` on each result
        and aggregated using the chosen strategy (``max``, ``mean``, or
        ``rrf``).  When only one query is provided, delegates directly to
        :meth:`rerank`.
        """
        if len(queries) == 1:
            return self.rerank(queries[0], candidates, top_k=top_k)
        if not candidates:
            return []

        input_k = self._config.input_top_k
        if input_k > 0:
            candidates = candidates[:input_k]

        # Score all candidates against each query
        per_query_results: list[list[SearchResult]] = [
            self.rerank(q, list(candidates)) for q in queries
        ]

        # Build per-key score lists: key -> [score_q0, score_q1, ...]
        per_key_scores: dict[str, list[float]] = {}
        for ranked in per_query_results:
            score_map = {r.key: r.score for r in ranked}
            for c in candidates:
                per_key_scores.setdefault(c.key, []).append(score_map.get(c.key, 0.0))

        aggregated = aggregate_scores(per_key_scores, aggregation)

        # Build result list with per-query scores
        candidate_map = {c.key: c for c in candidates}
        merged = [
            SearchResult(
                key=key,
                score=score,
                description=candidate_map[key].description,
                content=candidate_map[key].content,
                query_scores=per_key_scores[key],
            )
            for key, score in aggregated.items()
        ]
        merged.sort(key=lambda r: r.score, reverse=True)
        if top_k is not None:
            return merged[:top_k]
        return merged

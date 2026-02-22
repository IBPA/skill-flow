"""Cross-encoder reranker for Stage 2 of the retrieval pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sentence_transformers import CrossEncoder

from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from skill_flow.config import Reranker2Config, RerankerConfig

logger = logging.getLogger(__name__)


class Reranker:
    """Re-scores search results using a cross-encoder model."""

    def __init__(self, config: RerankerConfig | Reranker2Config) -> None:
        self._config = config
        logger.info("Loading cross-encoder model: %s", config.model_name)
        self._model = CrossEncoder(config.model_name)

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

        effective_k = top_k if top_k is not None else self._config.top_k
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
        reranked.extend(
            SearchResult(
                key=c.key,
                score=0.0,
                description=c.description,
                content=c.content,
            )
            for c in empty
        )
        reranked.sort(key=lambda r: r.score, reverse=True)

        return reranked[:effective_k]

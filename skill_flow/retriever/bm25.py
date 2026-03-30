"""BM25 sparse retriever using rank-bm25."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from skill_flow.config import RetrieverConfig
from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for English text."""
    return text.lower().split()


class BM25Searcher:
    """BM25-based sparse retriever over skill descriptions."""

    def __init__(
        self,
        skill_keys: list[str],
        corpus_tokens: list[list[str]],
        descriptions: dict[str, str],
        contents: dict[str, str],
        config: RetrieverConfig | None = None,
    ) -> None:
        self._config = config or RetrieverConfig()
        self._skill_keys = list(skill_keys)
        self._corpus_tokens = list(corpus_tokens)
        self._descriptions = dict(descriptions)
        self._contents = dict(contents)
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        logger.info("BM25 index built: %d documents", len(self._skill_keys))

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        config: RetrieverConfig | None = None,
    ) -> BM25Searcher:
        """Load skill artifacts from an existing index directory."""
        ids_path = index_dir / "skill_ids.json"
        skill_keys: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))

        desc_path = index_dir / "skill_descriptions.json"
        descriptions: dict[str, str] = (
            json.loads(desc_path.read_text(encoding="utf-8"))
            if desc_path.exists()
            else {}
        )

        content_path = index_dir / "skill_contents.json"
        contents: dict[str, str] = (
            json.loads(content_path.read_text(encoding="utf-8"))
            if content_path.exists()
            else {}
        )

        corpus_tokens = [_tokenize(descriptions.get(k, "")) for k in skill_keys]

        return cls(skill_keys, corpus_tokens, descriptions, contents, config)

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search the BM25 index and return the top-k results."""
        if self._bm25 is None or not self._skill_keys:
            return []
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        effective_k = top_k if top_k is not None else self._config.top_k
        k = min(effective_k, len(self._skill_keys))
        top_indices = scores.argsort()[::-1][:k]

        results: list[SearchResult] = []
        for idx in top_indices:
            key = self._skill_keys[int(idx)]
            score = float(scores[int(idx)])
            if score <= 0:
                break
            results.append(
                SearchResult(
                    key=key,
                    score=score,
                    description=self._descriptions.get(key, ""),
                    content=self._contents.get(key, ""),
                )
            )
        return results

    def augment(self, keys: list[str], descriptions: list[str]) -> None:
        """Inject additional skills and rebuild the BM25 index."""
        new_tokens = [_tokenize(d) for d in descriptions]
        self._skill_keys.extend(keys)
        self._corpus_tokens.extend(new_tokens)
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)

    def add_descriptions(self, descriptions: dict[str, str]) -> None:
        """Register descriptions for dynamically injected skills."""
        self._descriptions.update(descriptions)

    def add_contents(self, contents: dict[str, str]) -> None:
        """Register full SKILL.md content for dynamically injected skills."""
        self._contents.update(contents)

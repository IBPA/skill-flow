"""FAISS index searcher — loads a persisted index and runs queries."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import faiss
from pydantic import BaseModel

from skill_flow.config import RetrieverConfig

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.index.encoder import Encoder

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """A single search result with its cosine similarity score."""

    key: str  # skill key
    score: float  # cosine similarity
    description: str = ""
    content: str = ""
    query_scores: list[float] = []


class IndexSearcher:
    """Loads a persisted FAISS index and searches over it."""

    def __init__(
        self,
        index_dir: Path,
        encoder: Encoder,
        config: RetrieverConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._config = config or RetrieverConfig()

        logger.info("Loading FAISS index from %s …", index_dir)
        self._index: faiss.Index = faiss.read_index(str(index_dir / "faiss.index"))

        ids_path = index_dir / "skill_ids.json"
        self._skill_keys: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))

        desc_path = index_dir / "skill_descriptions.json"
        self._descriptions: dict[str, str] = (
            json.loads(desc_path.read_text(encoding="utf-8"))
            if desc_path.exists()
            else {}
        )

        content_path = index_dir / "skill_contents.json"
        self._contents: dict[str, str] = (
            json.loads(content_path.read_text(encoding="utf-8"))
            if content_path.exists()
            else {}
        )

        logger.info("Index loaded: %d vectors", self._index.ntotal)

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search the index and return the top-k results by score."""
        effective_k = top_k if top_k is not None else self._config.top_k
        query_vec = self._encoder.encode_query(query)

        # Clamp to the number of indexed vectors
        k = min(effective_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            key = self._skill_keys[int(idx)]
            results.append(
                SearchResult(
                    key=key,
                    score=float(score),
                    description=self._descriptions.get(key, ""),
                    content=self._contents.get(key, ""),
                )
            )

        return results

    def augment(self, keys: list[str], descriptions: list[str]) -> None:
        """Inject additional skill vectors into the FAISS index."""
        vectors = self._encoder.encode_documents(descriptions)
        self._index.add(vectors)
        self._skill_keys.extend(keys)

    def add_descriptions(self, descriptions: dict[str, str]) -> None:
        """Register descriptions for dynamically injected skills."""
        self._descriptions.update(descriptions)

    def add_contents(self, contents: dict[str, str]) -> None:
        """Register full SKILL.md content for dynamically injected skills."""
        self._contents.update(contents)

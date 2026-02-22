"""BGE encoder wrapper for generating skill embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer

from skill_flow.config import RetrieverConfig


class Encoder:
    """Wraps a sentence-transformers model for document and query encoding.

    Embeddings are L2-normalized so that inner product equals cosine
    similarity.
    """

    def __init__(self, config: RetrieverConfig | None = None) -> None:
        self._config = config or RetrieverConfig()
        self._model: SentenceTransformer = SentenceTransformer(self._config.model_name)

    def encode_documents(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """Encode documents without a query prefix.

        Returns an ``(N, dim)`` float32 matrix, L2-normalized.
        """
        bs = batch_size if batch_size is not None else self._config.batch_size
        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=bs,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query with the BGE query prefix.

        Returns a ``(1, dim)`` float32 matrix, L2-normalized.
        """
        prefixed = self._config.query_prompt + query
        embedding: np.ndarray = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

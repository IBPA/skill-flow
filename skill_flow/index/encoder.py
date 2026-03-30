"""BGE encoder wrapper for generating skill embeddings.

Includes ``pick_device()`` for automatic GPU selection (most free VRAM).
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from skill_flow.config import RetrieverConfig

logger = logging.getLogger(__name__)

_MIN_GPU_BYTES = 2 * 1024**3  # 2 GiB headroom for model + encoding


def pick_device() -> str:
    """Return the CUDA device with the most free memory, or ``'cpu'``."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"

    best_idx, best_free = 0, 0
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_idx = i

    if best_free < _MIN_GPU_BYTES:
        logger.warning(
            "No GPU has >= 2 GiB free (best: %.1f MiB on cuda:%d), falling back to CPU",
            best_free / 1024**2,
            best_idx,
        )
        return "cpu"

    logger.info(
        "Auto-selected cuda:%d (%.1f GiB free)",
        best_idx,
        best_free / 1024**3,
    )
    return f"cuda:{best_idx}"


class Encoder:
    """Wraps a sentence-transformers model for document and query encoding.

    Embeddings are L2-normalized so that inner product equals cosine
    similarity.
    """

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        device: str | None = None,
    ) -> None:
        self._config = config or RetrieverConfig()
        resolved_device = device if device is not None else pick_device()
        self._model: SentenceTransformer = SentenceTransformer(
            self._config.model_name, device=resolved_device
        )

    @property
    def max_seq_length(self) -> int:
        """Maximum token length accepted by the underlying model."""
        return self._model.max_seq_length

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate *text* to at most *max_tokens* using the model tokenizer."""
        tokenizer = self._model.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        result: str = tokenizer.decode(ids[:max_tokens])
        return result

    def encode_documents(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """Encode documents, optionally prepending ``doc_prompt``.

        Returns an ``(N, dim)`` float32 matrix, L2-normalized.
        """
        bs = batch_size if batch_size is not None else self._config.batch_size
        doc_prompt = self._config.doc_prompt
        prefixed = [doc_prompt + t for t in texts] if doc_prompt else texts
        embeddings: np.ndarray = self._model.encode(
            prefixed,
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

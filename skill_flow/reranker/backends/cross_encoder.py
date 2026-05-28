"""sentence-transformers CrossEncoder backend.

Used for cross-encoder reranking models with a sequence-classification head,
including ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (shallow) and
``BAAI/bge-reranker-v2-m3`` (deep). The class is re-exported as is because
:class:`sentence_transformers.CrossEncoder` already provides the
``predict(pairs, batch_size) -> np.ndarray`` interface the orchestrating
:class:`skill_flow.reranker.reranker.Reranker` calls.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder


def load(model_name: str, device: str) -> CrossEncoder:
    """Construct a sentence-transformers cross-encoder on ``device``."""
    model: CrossEncoder = CrossEncoder(model_name, device=device)
    return model


__all__ = ["CrossEncoder", "load"]

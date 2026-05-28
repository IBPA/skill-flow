"""Backend factory for the reranker stage.

Each backend module under this package provides a class exposing the
``predict(pairs, batch_size) -> np.ndarray`` interface that sentence-
transformers' :class:`CrossEncoder` defines. :func:`build_backend` selects
the right backend by model name so the orchestrating
:class:`skill_flow.reranker.reranker.Reranker` does not branch on model.

Routing:
    - ``Qwen3-Reranker*`` -> :class:`qwen3.Qwen3Reranker`
    - everything else    -> :class:`cross_encoder.CrossEncoder`
      (ms-marco-MiniLM-L-6-v2, BAAI/bge-reranker-v2-m3, ...)
"""

from __future__ import annotations

from skill_flow.reranker.backends import cross_encoder, qwen3
from skill_flow.reranker.backends.cross_encoder import CrossEncoder
from skill_flow.reranker.backends.qwen3 import Qwen3Reranker

RerankerBackend = CrossEncoder | Qwen3Reranker


def build_backend(model_name: str, device: str) -> RerankerBackend:
    """Construct the reranker backend matching ``model_name``."""
    if "qwen3-reranker" in model_name.lower():
        return qwen3.Qwen3Reranker(model_name, device=device)
    return cross_encoder.load(model_name, device=device)


__all__ = [
    "CrossEncoder",
    "Qwen3Reranker",
    "RerankerBackend",
    "build_backend",
    "cross_encoder",
    "qwen3",
]

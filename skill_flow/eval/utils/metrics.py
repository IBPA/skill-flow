"""Retrieval evaluation metrics: recall@k, hit@k, reciprocal rank."""

from __future__ import annotations


def recall_at_k(retrieved_keys: list[str], gt_keys: list[str], k: int) -> float:
    """Fraction of ground-truth keys found in top-k retrieved results."""
    if not gt_keys:
        return 0.0
    top_k = set(retrieved_keys[:k])
    hits = sum(1 for key in gt_keys if key in top_k)
    return hits / len(gt_keys)


def precision_at_k(retrieved_keys: list[str], gt_keys: list[str], k: int) -> float:
    """Fraction of top-k retrieved results that are ground-truth keys."""
    if k == 0:
        return 0.0
    top_k = set(retrieved_keys[:k])
    hits = sum(1 for key in top_k if key in set(gt_keys))
    return hits / k


def hit_at_k(retrieved_keys: list[str], gt_keys: list[str], k: int) -> float:
    """1.0 if any ground-truth key appears in top-k, else 0.0."""
    if not gt_keys:
        return 0.0
    top_k = set(retrieved_keys[:k])
    return 1.0 if any(key in top_k for key in gt_keys) else 0.0


def reciprocal_rank(retrieved_keys: list[str], gt_keys: list[str]) -> float:
    """Reciprocal of the rank of the first ground-truth key found."""
    if not gt_keys:
        return 0.0
    gt_set = set(gt_keys)
    for i, key in enumerate(retrieved_keys):
        if key in gt_set:
            return 1.0 / (i + 1)
    return 0.0

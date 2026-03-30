"""Shared helper to build a TaskResult with metrics from search results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skill_flow.eval.models import (
    PerQueryResult,
    RetrievedSkill,
    TaskResult,
)
from skill_flow.eval.utils.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

if TYPE_CHECKING:
    from skill_flow.eval.models import TaskGroundTruth
    from skill_flow.retriever.retriever import SearchResult


def build_task_result(
    task: TaskGroundTruth,
    results: list[SearchResult],
    ks: list[int],
    rerank_query: str = "",
    retrieval_query: str = "",
    retrieval_queries: list[PerQueryResult] | None = None,
    elapsed_ms: float = 0.0,
) -> TaskResult:
    """Build a TaskResult with metrics from search/rerank results."""
    keys = [r.key for r in results]
    return TaskResult(
        task_id=task.task_id,
        query=task.query,
        retrieval_query=retrieval_query,
        rerank_query=rerank_query,
        num_ground_truth=len(task.ground_truth_keys),
        num_injected=len(task.injected_skills),
        retrieved_skills=[
            RetrievedSkill(
                key=r.key,
                score=r.score,
                description=r.description,
                content=r.content,
                query_scores=r.query_scores,
            )
            for r in results
        ],
        retrieval_queries=retrieval_queries or [],
        recall_at={k: recall_at_k(keys, task.ground_truth_keys, k) for k in ks},
        precision_at={k: precision_at_k(keys, task.ground_truth_keys, k) for k in ks},
        hit_at={k: hit_at_k(keys, task.ground_truth_keys, k) for k in ks},
        reciprocal_rank=reciprocal_rank(keys, task.ground_truth_keys),
        elapsed_ms=elapsed_ms,
    )

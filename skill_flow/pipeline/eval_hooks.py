"""Eval hooks for pipeline stages -- compute metrics when GT is available."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skill_flow.eval.models import (
    RetrievedSkill,
    TaskResult,
    filter_ks,
)
from skill_flow.eval.utils.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from skill_flow.eval.utils.reporting import build_report, write_report

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.eval.models import EvalReport
    from skill_flow.pipeline.ground_truth import GroundTruthContext
    from skill_flow.retriever.retriever import SearchResult

logger = logging.getLogger(__name__)


def _gt_keys_for_task(
    task_id: str,
    gt_context: GroundTruthContext,
) -> list[str] | None:
    """Return GT keys for a task, or None if not in GT."""
    for t in gt_context.tasks:
        if t.task_id == task_id:
            return t.ground_truth_keys
    return None


def build_task_result(
    task_id: str,
    query: str,
    results: list[SearchResult],
    gt_context: GroundTruthContext,
    ks: list[int],
    retrieval_query: str = "",
    rerank_query: str = "",
) -> TaskResult | None:
    """Compute eval metrics for one task. Returns None if task has no GT."""
    gt_keys = _gt_keys_for_task(task_id, gt_context)
    if gt_keys is None:
        return None

    gt_task = next(t for t in gt_context.tasks if t.task_id == task_id)
    keys = [r.key for r in results]

    return TaskResult(
        task_id=task_id,
        query=query,
        retrieval_query=retrieval_query,
        rerank_query=rerank_query,
        num_ground_truth=len(gt_keys),
        num_injected=len(gt_task.injected_skills),
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
        recall_at={k: recall_at_k(keys, gt_keys, k) for k in ks},
        precision_at={k: precision_at_k(keys, gt_keys, k) for k in ks},
        hit_at={k: hit_at_k(keys, gt_keys, k) for k in ks},
        reciprocal_rank=reciprocal_rank(keys, gt_keys),
    )


def build_selector_task_result(
    task_id: str,
    query: str,
    candidates: list[SearchResult],
    selected: list[SearchResult],
    gt_context: GroundTruthContext,
    ks: list[int],
) -> TaskResult | None:
    """Compute eval metrics for selector stage, including select_recall/precision."""
    gt_keys = _gt_keys_for_task(task_id, gt_context)
    if gt_keys is None:
        return None

    gt_task = next(t for t in gt_context.tasks if t.task_id == task_id)
    selected_keys = {s.key for s in selected}
    gt_set = set(gt_keys)

    ranked = [
        RetrievedSkill(
            key=c.key,
            score=1.0 if c.key in selected_keys else 0.0,
            description=c.description,
            content=c.content,
        )
        for c in candidates
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)
    ranked_keys = [r.key for r in ranked]

    tp = len(selected_keys & gt_set)
    sel_recall = tp / len(gt_set) if gt_set else 0.0
    sel_prec = tp / len(selected_keys) if selected_keys else 0.0

    return TaskResult(
        task_id=task_id,
        query=query,
        num_ground_truth=len(gt_keys),
        num_injected=len(gt_task.injected_skills),
        retrieved_skills=ranked,
        recall_at={k: recall_at_k(ranked_keys, gt_keys, k) for k in ks},
        precision_at={k: precision_at_k(ranked_keys, gt_keys, k) for k in ks},
        hit_at={k: hit_at_k(ranked_keys, gt_keys, k) for k in ks},
        reciprocal_rank=reciprocal_rank(ranked_keys, gt_keys),
        select_recall=sel_recall,
        select_precision=sel_prec,
    )


def write_eval_report(
    task_results: list[TaskResult],
    gt_context: GroundTruthContext,
    ks: list[int],
    output_path: Path,
    tasks_dir: Path,
    index_dir: Path,
) -> EvalReport:
    """Build and write an eval report from task results."""
    num_total = len(gt_context.tasks) + len(gt_context.skipped)
    config: dict[str, object] = {
        "tasks_dir": str(tasks_dir),
        "index_dir": str(index_dir),
    }
    report = build_report(
        task_results,
        num_total,
        len(gt_context.skipped),
        len(gt_context.injected_skills),
        ks,
        config,
    )
    write_report(report, output_path)
    return report


def get_eval_ks(top_k: int) -> list[int]:
    """Return eval k values appropriate for the given top_k."""
    return filter_ks(top_k)

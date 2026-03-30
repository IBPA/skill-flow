"""Reciprocal Rank Fusion (RRF) for combining two retriever EvalReports.

Provides ``fuse_reports_rrf()`` which merges skill rankings from two
independent retrieval runs using standard RRF scoring, then recomputes
all evaluation metrics on the fused ranking.
"""

from __future__ import annotations

from skill_flow.eval.models import (
    EVAL_KS,
    EvalReport,
    EvalSummary,
    RetrievedSkill,
    TaskResult,
)
from skill_flow.eval.utils.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

_RRF_K = 60  # standard RRF constant


def fuse_reports_rrf(
    report_a: EvalReport,
    report_b: EvalReport,
) -> EvalReport:
    """Fuse two retriever EvalReports via Reciprocal Rank Fusion.

    For each task present in both reports, skill rankings are combined
    using RRF scores: ``1 / (k + rank)``.  The fused ranking is used to
    recompute all standard retrieval metrics.
    """
    map_a = {tr.task_id: tr for tr in report_a.task_results}
    map_b = {tr.task_id: tr for tr in report_b.task_results}
    shared = sorted(set(map_a) & set(map_b))

    task_results: list[TaskResult] = []
    for tid in shared:
        tr_a, tr_b = map_a[tid], map_b[tid]

        # Build RRF scores per skill key
        rrf_scores: dict[str, float] = {}
        for rank, sk in enumerate(tr_a.retrieved_skills, 1):
            rrf_scores[sk.key] = rrf_scores.get(sk.key, 0.0) + 1.0 / (_RRF_K + rank)
        for rank, sk in enumerate(tr_b.retrieved_skills, 1):
            rrf_scores[sk.key] = rrf_scores.get(sk.key, 0.0) + 1.0 / (_RRF_K + rank)

        # Sort by fused score descending
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused_keys = [k for k, _ in fused]
        fused_skills = [RetrievedSkill(key=k, score=s) for k, s in fused]

        # GT keys follow the pattern skillsbench/{task_id}/{skill_name}
        task_prefix = f"skillsbench/{tid}/"
        gt_set: set[str] = set()
        for sk in tr_a.retrieved_skills:
            if sk.key.startswith(task_prefix):
                gt_set.add(sk.key)
        for sk in tr_b.retrieved_skills:
            if sk.key.startswith(task_prefix):
                gt_set.add(sk.key)
        gt_keys = sorted(gt_set)

        ks = [k for k in EVAL_KS if k <= len(fused_keys)]
        recall = {k: recall_at_k(fused_keys, gt_keys, k) for k in ks}
        precision = {k: precision_at_k(fused_keys, gt_keys, k) for k in ks}
        hit = {k: hit_at_k(fused_keys, gt_keys, k) for k in ks}
        rr = reciprocal_rank(fused_keys, gt_keys)

        task_results.append(
            TaskResult(
                task_id=tid,
                query=tr_a.query,
                num_ground_truth=len(gt_keys),
                num_injected=tr_a.num_injected,
                retrieved_skills=fused_skills,
                recall_at=recall,
                precision_at=precision,
                hit_at=hit,
                reciprocal_rank=rr,
            )
        )

    n = len(task_results)
    summary = EvalSummary(
        num_tasks_total=n,
        num_tasks_evaluated=n,
        num_tasks_no_skills=0,
        num_skills_injected=sum(tr.num_injected for tr in task_results),
        mean_recall_at={
            k: sum(tr.recall_at.get(k, 0.0) for tr in task_results) / n for k in EVAL_KS
        },
        mean_precision_at={
            k: sum(tr.precision_at.get(k, 0.0) for tr in task_results) / n
            for k in EVAL_KS
        },
        mean_hit_at={
            k: sum(tr.hit_at.get(k, 0.0) for tr in task_results) / n for k in EVAL_KS
        },
        mrr=sum(tr.reciprocal_rank for tr in task_results) / n,
    )

    return EvalReport(
        summary=summary,
        task_results=task_results,
        config=report_a.config,
    )

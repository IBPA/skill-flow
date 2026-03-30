"""Report building, writing, and snapshot helpers for evaluation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from skill_flow.eval.models import (
    EvalReport,
    EvalSummary,
    PerQueryResult,
    RetrievedSkill,
    TaskResult,
)

if TYPE_CHECKING:
    from pathlib import Path


def build_summary(
    task_results: list[TaskResult],
    num_tasks_total: int,
    num_tasks_no_skills: int,
    num_skills_injected: int,
    ks: list[int],
) -> EvalSummary:
    """Aggregate per-task results into summary metrics."""
    n = len(task_results)

    mean_recall = {
        k: sum(r.recall_at[k] for r in task_results) / n if n else 0.0 for k in ks
    }
    mean_precision = {
        k: sum(r.precision_at[k] for r in task_results) / n if n else 0.0 for k in ks
    }
    mean_hit = {k: sum(r.hit_at[k] for r in task_results) / n if n else 0.0 for k in ks}
    mrr = sum(r.reciprocal_rank for r in task_results) / n if n else 0.0
    mean_sel_r = sum(r.select_recall for r in task_results) / n if n else 0.0
    mean_sel_p = sum(r.select_precision for r in task_results) / n if n else 0.0
    mean_elapsed = sum(r.elapsed_ms for r in task_results) / n if n else 0.0

    return EvalSummary(
        num_tasks_total=num_tasks_total,
        num_tasks_evaluated=n,
        num_tasks_no_skills=num_tasks_no_skills,
        num_skills_injected=num_skills_injected,
        mean_recall_at=mean_recall,
        mean_precision_at=mean_precision,
        mean_hit_at=mean_hit,
        mrr=mrr,
        mean_select_recall=mean_sel_r,
        mean_select_precision=mean_sel_p,
        mean_elapsed_ms=mean_elapsed,
    )


def build_report(
    task_results: list[TaskResult],
    num_tasks_total: int,
    num_tasks_no_skills: int,
    num_skills_injected: int,
    ks: list[int],
    config: dict[str, object],
) -> EvalReport:
    """Build an evaluation report from task results."""
    summary = build_summary(
        task_results, num_tasks_total, num_tasks_no_skills, num_skills_injected, ks
    )
    return EvalReport(summary=summary, task_results=task_results, config=config)


def write_report(report: EvalReport, output_path: Path | None) -> None:
    """Write an evaluation report to JSON (without skill content)."""
    if not output_path:
        return
    slim = report.model_copy(
        update={"task_results": _strip_skill_content(report.task_results)},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(slim.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def _strip_skill(s: RetrievedSkill) -> RetrievedSkill:
    """Strip content from a single RetrievedSkill (preserve scores)."""
    return RetrievedSkill(
        key=s.key,
        score=s.score,
        description=s.description,
        query_scores=s.query_scores,
    )


def _strip_skill_content(
    task_results: list[TaskResult],
) -> list[TaskResult]:
    """Strip SKILL.md content from results (keeps descriptions and scores)."""
    return [
        r.model_copy(
            update={
                "retrieved_skills": [_strip_skill(s) for s in r.retrieved_skills],
                "retrieval_queries": [
                    PerQueryResult(
                        query=pq.query,
                        retrieved_skills=[_strip_skill(s) for s in pq.retrieved_skills],
                    )
                    for pq in r.retrieval_queries
                ],
            }
        )
        for r in task_results
    ]


def print_reranker_comparison(results: list[tuple[str, EvalReport]]) -> None:
    """Print a side-by-side comparison table of reranker results."""
    header = f"{'Model':<35} {'MRR':>7} {'R@1':>7} {'R@2':>7} {'R@5':>7} {'R@10':>7}"
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")
    for label, report in results:
        s = report.summary
        print(
            f"{label:<35}"
            f" {s.mrr:>7.4f}"
            f" {s.mean_recall_at.get(1, 0.0):>7.4f}"
            f" {s.mean_recall_at.get(2, 0.0):>7.4f}"
            f" {s.mean_recall_at.get(5, 0.0):>7.4f}"
            f" {s.mean_recall_at.get(10, 0.0):>7.4f}"
        )
    print(f"{'=' * len(header)}")


def print_selector_comparison(results: list[tuple[str, EvalReport]]) -> None:
    """Print a side-by-side comparison table of selector results."""
    header = (
        f"{'Model':<35} {'SelR':>7} {'SelP':>7}"
        f" {'MRR':>7} {'R@1':>7} {'R@2':>7} {'R@5':>7} {'R@10':>7}"
    )
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")
    for label, report in results:
        s = report.summary
        print(
            f"{label:<35}"
            f" {s.mean_select_recall:>7.4f}"
            f" {s.mean_select_precision:>7.4f}"
            f" {s.mrr:>7.4f}"
            f" {s.mean_recall_at.get(1, 0.0):>7.4f}"
            f" {s.mean_recall_at.get(2, 0.0):>7.4f}"
            f" {s.mean_recall_at.get(5, 0.0):>7.4f}"
            f" {s.mean_recall_at.get(10, 0.0):>7.4f}"
        )
    print(f"{'=' * len(header)}")


def write_snapshot(
    task_results: list[TaskResult],
    num_tasks_total: int,
    num_tasks_no_skills: int,
    num_skills_injected: int,
    ks: list[int],
    config: dict[str, object],
    output_path: Path | None = None,
) -> None:
    """Write a lightweight incremental snapshot (without retrieved_skills)."""
    report = build_report(
        _strip_skill_content(task_results),
        num_tasks_total,
        num_tasks_no_skills,
        num_skills_injected,
        ks,
        config,
    )
    write_report(report, output_path)

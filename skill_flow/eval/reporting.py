"""Report building, writing, and snapshot helpers for evaluation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from skill_flow.eval.models import (
    EvalReport,
    EvalSummary,
    RetrievedSkill,
    TaskResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.eval.models import (
        EvalRunConfig,
        Reranker2EvalConfig,
        RerankerEvalConfig,
        SelectorEvalConfig,
    )

    type AnyEvalConfig = (
        EvalRunConfig
        | RerankerEvalConfig
        | Reranker2EvalConfig
        | SelectorEvalConfig
    )


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

    return EvalSummary(
        num_tasks_total=num_tasks_total,
        num_tasks_evaluated=n,
        num_tasks_no_skills=num_tasks_no_skills,
        num_skills_injected=num_skills_injected,
        mean_recall_at=mean_recall,
        mean_precision_at=mean_precision,
        mean_hit_at=mean_hit,
        mrr=mrr,
    )


def build_report(
    task_results: list[TaskResult],
    num_tasks_total: int,
    num_tasks_no_skills: int,
    num_skills_injected: int,
    ks: list[int],
    config: AnyEvalConfig,
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


def _strip_skill_content(
    task_results: list[TaskResult],
) -> list[TaskResult]:
    """Strip full SKILL.md content from retrieved_skills (keeps descriptions)."""
    return [
        r.model_copy(
            update={
                "retrieved_skills": [
                    RetrievedSkill(
                        key=s.key, score=s.score, description=s.description,
                    )
                    for s in r.retrieved_skills
                ]
            }
        )
        for r in task_results
    ]


def write_snapshot(
    task_results: list[TaskResult],
    num_tasks_total: int,
    num_tasks_no_skills: int,
    num_skills_injected: int,
    ks: list[int],
    config: AnyEvalConfig,
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
    write_report(report, config.output_path)

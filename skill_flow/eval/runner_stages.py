"""Stage 2-4 evaluation runners (reranker, deep reranker, selector)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from skill_flow.eval.models import (
    EvalReport,
    TaskResult,
    filter_ks,
)
from skill_flow.eval.utils.ground_truth import load_content_map, load_ground_truth
from skill_flow.eval.utils.reporting import build_report, write_report, write_snapshot
from skill_flow.eval.utils.task_result import build_task_result
from skill_flow.pipeline.query_gen_init import init_rerank_query_gen
from skill_flow.pipeline.stage_helpers import select_task
from skill_flow.reranker.reranker import Reranker
from skill_flow.retriever.retriever import SearchResult
from skill_flow.selector.selector import Selector

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.config import (
        DeepRerankerConfig,
        RerankerConfig,
        SelectorConfig,
    )

logger = logging.getLogger(__name__)


def _fmt_recall(result: TaskResult) -> str:
    parts = [f"R@{k}={v:.2f}" for k, v in sorted(result.recall_at.items())]
    return " ".join(parts)


def _run_rerank_stage(
    prev_report_path: Path,
    reranker_cfg: RerankerConfig | DeepRerankerConfig,
    tasks_dir: Path,
    index_dir: Path,
    max_tasks: int = 0,
    include_tasks: list[str] | None = None,
    output_path: Path | None = None,
) -> EvalReport:
    """Shared logic: load previous report, rerank, compute metrics, report."""
    prev_report = EvalReport.model_validate_json(
        prev_report_path.read_text(encoding="utf-8"),
    )
    tasks, injected_skills, skipped = load_ground_truth(tasks_dir)
    content_map = load_content_map(index_dir, injected_skills)
    reranker = Reranker(reranker_cfg)
    ks = filter_ks(reranker_cfg.input_top_k)
    query_gen, agg_method = init_rerank_query_gen(reranker_cfg.query_gen)

    task_map = {t.task_id: t for t in tasks}
    n_total = len(tasks) + len(skipped)
    cfg: dict[str, object] = {
        "prev_report_path": str(prev_report_path),
        "tasks_dir": str(tasks_dir),
        "index_dir": str(index_dir),
        "reranker": reranker_cfg.model_dump(mode="json"),
        "output_path": str(output_path) if output_path else None,
    }

    prev_tasks = prev_report.task_results
    if max_tasks > 0:
        prev_tasks = prev_tasks[:max_tasks]
    if include_tasks:
        inc = set(include_tasks)
        prev_tasks = [t for t in prev_tasks if t.task_id in inc]

    task_results: list[TaskResult] = []
    for prev_result in prev_tasks:
        task = task_map.get(prev_result.task_id)
        if task is None:
            logger.warning("Task %s not in GT, skipping", prev_result.task_id)
            continue

        prev_skills = prev_result.retrieved_skills
        if reranker_cfg.input_top_k > 0:
            prev_skills = prev_skills[: reranker_cfg.input_top_k]
        candidates = [
            SearchResult(
                key=s.key,
                score=s.score,
                description=s.description,
                content=content_map.get(s.key, s.content),
            )
            for s in prev_skills
        ]

        t0 = time.perf_counter()
        if query_gen:
            queries = query_gen.generate_multi(task.task_id, task.query)
            reranked = reranker.rerank_multi(
                queries,
                candidates,
                aggregation=agg_method,
            )
            rq = " | ".join(queries)
        else:
            reranked = reranker.rerank(task.query, candidates)
            rq = task.query
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result = build_task_result(
            task, reranked, ks, rerank_query=rq, elapsed_ms=elapsed_ms,
        )
        task_results.append(result)
        logger.info("Task %s: %s", task.task_id, _fmt_recall(result))
        write_snapshot(
            task_results,
            n_total,
            len(skipped),
            len(injected_skills),
            ks,
            cfg,
            output_path,
        )

    report = build_report(
        task_results,
        n_total,
        len(skipped),
        len(injected_skills),
        ks,
        cfg,
    )
    write_report(report, output_path)
    return report


def run_reranker_evaluation(
    prev_report_path: Path,
    reranker_cfg: RerankerConfig,
    tasks_dir: Path,
    index_dir: Path,
    max_tasks: int = 0,
    include_tasks: list[str] | None = None,
    output_path: Path | None = None,
) -> EvalReport:
    """Run a Stage 2 reranker evaluation on cached Stage 1 results."""
    return _run_rerank_stage(
        prev_report_path,
        reranker_cfg,
        tasks_dir,
        index_dir,
        max_tasks,
        include_tasks,
        output_path,
    )


def run_deep_reranker_evaluation(
    prev_report_path: Path,
    reranker_cfg: DeepRerankerConfig,
    tasks_dir: Path,
    index_dir: Path,
    max_tasks: int = 0,
    include_tasks: list[str] | None = None,
    output_path: Path | None = None,
) -> EvalReport:
    """Run a Stage 3 deep reranker evaluation on cached Stage 2 results."""
    return _run_rerank_stage(
        prev_report_path,
        reranker_cfg,
        tasks_dir,
        index_dir,
        max_tasks,
        include_tasks,
        output_path,
    )


def run_selector_evaluation(
    prev_report_path: Path,
    selector_cfg: SelectorConfig,
    tasks_dir: Path,
    index_dir: Path,
    max_tasks: int = 0,
    include_tasks: list[str] | None = None,
    output_path: Path | None = None,
) -> EvalReport:
    """Run a Stage 4 selector evaluation on cached Stage 3 results."""
    prev_report = EvalReport.model_validate_json(
        prev_report_path.read_text(encoding="utf-8"),
    )
    tasks, injected_skills, skipped = load_ground_truth(tasks_dir)
    content_map = load_content_map(index_dir, injected_skills)
    selector = Selector(selector_cfg)
    ks = filter_ks(selector_cfg.input_top_k)

    task_map = {t.task_id: t for t in tasks}
    n_total = len(tasks) + len(skipped)
    cfg: dict[str, object] = {
        "prev_report_path": str(prev_report_path),
        "tasks_dir": str(tasks_dir),
        "index_dir": str(index_dir),
        "selector": selector_cfg.model_dump(mode="json"),
        "output_path": str(output_path) if output_path else None,
    }

    prev_tasks = prev_report.task_results
    if max_tasks > 0:
        prev_tasks = prev_tasks[:max_tasks]
    if include_tasks:
        inc = set(include_tasks)
        prev_tasks = [t for t in prev_tasks if t.task_id in inc]

    task_results: list[TaskResult] = []
    for prev_result in prev_tasks:
        task = task_map.get(prev_result.task_id)
        if task is None:
            logger.warning("Task %s not in GT, skipping", prev_result.task_id)
            continue

        prev_skills = prev_result.retrieved_skills
        if selector_cfg.input_top_k > 0:
            prev_skills = prev_skills[: selector_cfg.input_top_k]
        candidates = [
            SearchResult(
                key=s.key,
                score=s.score,
                description=s.description,
                content=content_map.get(s.key, s.content),
            )
            for s in prev_skills
        ]

        t0 = time.perf_counter()
        selected = select_task(selector, task.query, task.task_id, candidates)
        selected_keys = {s.key for s in selected}
        ranked = sorted(
            [
                SearchResult(
                    key=c.key,
                    score=1.0 if c.key in selected_keys else 0.0,
                    description=c.description,
                    content=c.content,
                )
                for c in candidates
            ],
            key=lambda r: r.score,
            reverse=True,
        )
        gt_keys = set(task.ground_truth_keys)
        tp = len(selected_keys & gt_keys)
        sel_recall = tp / len(gt_keys) if gt_keys else 0.0
        sel_prec = tp / len(selected_keys) if selected_keys else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result = build_task_result(task, ranked, ks, elapsed_ms=elapsed_ms)
        result = result.model_copy(
            update={"select_recall": sel_recall, "select_precision": sel_prec},
        )
        task_results.append(result)
        logger.info("Task %s: %s", task.task_id, _fmt_recall(result))
        write_snapshot(
            task_results,
            n_total,
            len(skipped),
            len(injected_skills),
            ks,
            cfg,
            output_path,
        )

    report = build_report(
        task_results,
        n_total,
        len(skipped),
        len(injected_skills),
        ks,
        cfg,
    )
    write_report(report, output_path)
    return report

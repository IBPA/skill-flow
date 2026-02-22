"""Retriever evaluation orchestrator."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skill_flow.eval.ground_truth import load_content_map, load_ground_truth
from skill_flow.eval.metrics import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from skill_flow.eval.models import (
    EvalReport,
    EvalRunConfig,
    InjectedSkill,
    Reranker2EvalConfig,
    RerankerEvalConfig,
    RetrievedSkill,
    SelectorEvalConfig,
    TaskResult,
    filter_ks,
)
from skill_flow.eval.reporting import build_report, write_report, write_snapshot
from skill_flow.index.encoder import Encoder
from skill_flow.models.core import SkillFlow
from skill_flow.reranker.query_gen import QueryGenerator
from skill_flow.reranker.reranker import Reranker
from skill_flow.retriever.retriever import IndexSearcher, SearchResult
from skill_flow.selector.selector import Selector

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from skill_flow.eval.models import TaskGroundTruth

logger = logging.getLogger(__name__)


def _augment_index(
    searcher: IndexSearcher,
    encoder: Encoder,
    skills: list[InjectedSkill],
) -> None:
    """Inject GT skills into the FAISS index."""
    if not skills:
        return

    descriptions = [s.description for s in skills]
    vectors: np.ndarray = encoder.encode_documents(descriptions)
    searcher._index.add(vectors)
    searcher._skill_keys.extend(s.key for s in skills)
    searcher.add_descriptions({s.key: s.description for s in skills})
    searcher.add_contents({s.key: s.content for s in skills if s.content})

    logger.info(
        "Injected %d skills into index (now %d total)",
        len(skills),
        searcher._index.ntotal,
    )


def _build_task_result(
    task: TaskGroundTruth,
    results: list[SearchResult],
    ks: list[int],
    rerank_query: str = "",
) -> TaskResult:
    """Build a TaskResult with metrics from search/rerank results."""
    keys = [r.key for r in results]
    return TaskResult(
        task_id=task.task_id,
        query=task.query,
        rerank_query=rerank_query,
        num_ground_truth=len(task.ground_truth_keys),
        num_injected=len(task.injected_skills),
        retrieved_skills=[
            RetrievedSkill(
                key=r.key, score=r.score, description=r.description, content=r.content
            )
            for r in results
        ],
        recall_at={k: recall_at_k(keys, task.ground_truth_keys, k) for k in ks},
        precision_at={k: precision_at_k(keys, task.ground_truth_keys, k) for k in ks},
        hit_at={k: hit_at_k(keys, task.ground_truth_keys, k) for k in ks},
        reciprocal_rank=reciprocal_rank(keys, task.ground_truth_keys),
    )


def run_evaluation(config: EvalRunConfig) -> EvalReport:
    """Run a full retriever evaluation against SkillsBench ground truth."""
    tasks, injected_skills, skipped = load_ground_truth(
        config.tasks_dir, config.max_query_chars
    )

    encoder = Encoder(config.retriever)
    searcher = IndexSearcher(config.index_dir, encoder, config.retriever)

    _augment_index(searcher, encoder, injected_skills)

    reranker = Reranker(config.reranker) if config.reranker else None
    retriever = SkillFlow(searcher, reranker)
    ks = filter_ks(config.retriever.top_k)

    eval_tasks = tasks[: config.max_tasks] if config.max_tasks > 0 else tasks
    num_tasks_total = len(tasks) + len(skipped)
    task_results: list[TaskResult] = []
    for task in eval_tasks:
        results = retriever.search(task.query)
        result = _build_task_result(task, results, ks)
        task_results.append(result)
        logger.info(
            "Task %s: RR=%.3f, R@10=%.3f",
            task.task_id,
            result.reciprocal_rank,
            result.recall_at.get(10, 0.0),
        )
        write_snapshot(
            task_results, num_tasks_total, len(skipped),
            len(injected_skills), ks, config,
        )

    report = build_report(
        task_results, num_tasks_total, len(skipped),
        len(injected_skills), ks, config,
    )
    write_report(report, config.output_path)
    return report


def _run_rerank_stage(
    prev_report_path: Path,
    config: RerankerEvalConfig | Reranker2EvalConfig,
) -> EvalReport:
    """Shared logic: load previous report, rerank, compute metrics, report."""
    prev_report = EvalReport.model_validate_json(
        prev_report_path.read_text(encoding="utf-8"),
    )

    tasks, injected_skills, skipped = load_ground_truth(config.tasks_dir)
    content_map = load_content_map(config.index_dir, injected_skills)
    reranker = Reranker(config.reranker)
    top_k = config.reranker.top_k
    ks = filter_ks(top_k)

    query_gen: QueryGenerator | None = None
    qg_config = config.reranker.query_gen
    if qg_config and qg_config.enabled:
        query_gen = QueryGenerator(qg_config)

    task_query_map = {t.task_id: t for t in tasks}
    num_tasks_total = len(tasks) + len(skipped)

    prev_tasks = prev_report.task_results
    if config.max_tasks > 0:
        prev_tasks = prev_tasks[: config.max_tasks]
    task_results: list[TaskResult] = []
    for prev_result in prev_tasks:
        task = task_query_map.get(prev_result.task_id)
        if task is None:
            logger.warning("Task %s not found in GT, skipping", prev_result.task_id)
            continue

        candidates = [
            SearchResult(
                key=s.key,
                score=s.score,
                description=s.description,
                content=content_map.get(s.key, s.content),
            )
            for s in prev_result.retrieved_skills[:top_k]
        ]

        rerank_query = (
            query_gen.generate(task.task_id, task.query) if query_gen else task.query
        )
        reranked = reranker.rerank(rerank_query, candidates, top_k=top_k)
        result = _build_task_result(
            task, reranked, ks, rerank_query=rerank_query,
        )
        task_results.append(result)
        logger.info(
            "Task %s: RR=%.3f, R@10=%.3f",
            task.task_id,
            result.reciprocal_rank,
            result.recall_at.get(10, 0.0),
        )
        write_snapshot(
            task_results, num_tasks_total, len(skipped),
            len(injected_skills), ks, config,
        )

    report = build_report(
        task_results, num_tasks_total, len(skipped),
        len(injected_skills), ks, config,
    )
    write_report(report, config.output_path)
    return report


def run_reranker_evaluation(config: RerankerEvalConfig) -> EvalReport:
    """Run a Stage 2-only reranker evaluation on cached Stage 1 results."""
    return _run_rerank_stage(config.stage1_report_path, config)


def run_reranker2_evaluation(config: Reranker2EvalConfig) -> EvalReport:
    """Run a Stage 3 reranker2 evaluation on cached Stage 2 results."""
    return _run_rerank_stage(config.stage2_report_path, config)


def _run_selector_stage(
    prev_report_path: Path,
    config: SelectorEvalConfig,
) -> EvalReport:
    """Load previous report, run LLM selector, compute metrics, report."""
    prev_report = EvalReport.model_validate_json(
        prev_report_path.read_text(encoding="utf-8"),
    )

    tasks, injected_skills, skipped = load_ground_truth(config.tasks_dir)
    content_map = load_content_map(config.index_dir, injected_skills)
    selector = Selector(config.selector)
    top_k = config.selector.top_k
    ks = filter_ks(top_k)

    task_query_map = {t.task_id: t for t in tasks}
    num_tasks_total = len(tasks) + len(skipped)

    prev_tasks = prev_report.task_results
    if config.max_tasks > 0:
        prev_tasks = prev_tasks[: config.max_tasks]
    task_results: list[TaskResult] = []
    for prev_result in prev_tasks:
        task = task_query_map.get(prev_result.task_id)
        if task is None:
            logger.warning("Task %s not found in GT, skipping", prev_result.task_id)
            continue

        candidates = [
            SearchResult(
                key=s.key,
                score=s.score,
                description=s.description,
                content=content_map.get(s.key, s.content),
            )
            for s in prev_result.retrieved_skills[:top_k]
        ]

        selected = selector.select(task.query, candidates, task_id=task.task_id)
        result = _build_task_result(task, selected, ks)
        task_results.append(result)
        logger.info(
            "Task %s: RR=%.3f, selected=%d/%d",
            task.task_id,
            result.reciprocal_rank,
            len(selected),
            len(candidates),
        )
        write_snapshot(
            task_results, num_tasks_total, len(skipped),
            len(injected_skills), ks, config,
        )

    report = build_report(
        task_results, num_tasks_total, len(skipped),
        len(injected_skills), ks, config,
    )
    write_report(report, config.output_path)
    return report


def run_selector_evaluation(config: SelectorEvalConfig) -> EvalReport:
    """Run a Stage 4 selector evaluation on cached Stage 3 results."""
    return _run_selector_stage(config.stage3_report_path, config)

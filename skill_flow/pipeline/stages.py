"""Pipeline stage runners with optional eval-mode metrics."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from skill_flow.reranker.reranker import Reranker
from skill_flow.selector.selector import Selector

from .ground_truth import augment_searcher
from .models import _write_stage_results
from .query_gen_init import init_rerank_query_gen, init_retriever_query_gen
from .stage_helpers import (
    build_searcher,
    build_selector_stage_results,
    measure_call,
    process_retrieval_task,
    process_selector_task,
    rerank_task,
    retrieve_task,
    select_task,
    write_eval_if_needed,
)

if TYPE_CHECKING:
    from skill_flow.config import (
        Config,
        DeepRerankerConfig,
        RerankerConfig,
    )
    from skill_flow.eval.models import TaskResult
    from skill_flow.retriever.retriever import SearchResult

    from .ground_truth import GroundTruthContext
    from .models import PipelineStageResult, TaskInstruction

logger = logging.getLogger(__name__)


def run_stage1(
    config: Config,
    index_dir: Path,
    tasks: list[TaskInstruction],
    task_candidates: dict[str, list[SearchResult]],
    output_dir: Path,
    *,
    gt_context: GroundTruthContext | None = None,
    tasks_dir: Path | None = None,
) -> None:
    """Stage 1: dense retriever with optional query generation."""
    searcher = build_searcher(config.models.retriever, index_dir)
    if gt_context:
        augment_searcher(searcher, gt_context.injected_skills)
    query_gen, agg, tkpq = init_retriever_query_gen(
        config.models.retriever.query_gen,
        cache_path=str(output_dir / "retriever_query_gen_cache.json"),
    )

    stage: list[PipelineStageResult] = []
    eval_results: list[TaskResult] = []
    for task in tasks:
        results, elapsed_ms = measure_call(
            retrieve_task,
            searcher,
            task.query,
            config.models.retriever.top_k,
            query_gen,
            agg,
            tkpq,
            task_id=task.task_id,
        )
        process_retrieval_task(
            task,
            results,  # type: ignore[arg-type]
            task_candidates,
            gt_context,
            config.models.retriever.top_k,
            stage,
            eval_results,
            elapsed_ms,
        )
        logger.info("Stage 1 [%s]: %d results", task.task_id, len(results))  # type: ignore[arg-type]

    _write_stage_results(
        output_dir / "pipeline-stage1-retriever.json",
        "retriever",
        stage,
    )
    write_eval_if_needed(
        "stage1-retriever",
        eval_results,
        gt_context,
        config.models.retriever.top_k,
        output_dir,
        tasks_dir or output_dir,
        index_dir,
    )


def _run_rerank_stage(
    reranker_cfg: RerankerConfig | DeepRerankerConfig,
    stage_num: int,
    stage_name: str,
    config: Config,
    tasks: list[TaskInstruction],
    task_candidates: dict[str, list[SearchResult]],
    output_dir: Path,
    gt_context: GroundTruthContext | None,
    tasks_dir: Path | None,
) -> None:
    """Shared logic for stages 2 (reranker) and 3 (deep reranker)."""
    reranker = Reranker(reranker_cfg)
    qg, agg = init_rerank_query_gen(
        reranker_cfg.query_gen,
        cache_path=str(output_dir / "query_gen_cache.json"),
    )
    stage: list[PipelineStageResult] = []
    eval_results: list[TaskResult] = []
    for task in tasks:
        reranked, elapsed_ms = measure_call(
            rerank_task,
            reranker,
            task.query,
            task_candidates[task.task_id],
            qg,
            agg,
            task_id=task.task_id,
        )
        process_retrieval_task(
            task,
            reranked,  # type: ignore[arg-type]
            task_candidates,
            gt_context,
            reranker_cfg.input_top_k,
            stage,
            eval_results,
            elapsed_ms,
        )
        logger.info(
            "Stage %d [%s]: %d results",
            stage_num,
            task.task_id,
            len(reranked),  # type: ignore[arg-type]
        )
    _write_stage_results(
        output_dir / f"pipeline-stage{stage_num}-{stage_name}.json",
        stage_name,
        stage,
    )
    write_eval_if_needed(
        f"stage{stage_num}-{stage_name}",
        eval_results,
        gt_context,
        reranker_cfg.input_top_k,
        output_dir,
        tasks_dir or output_dir,
        Path(config.index.output_index_path),
    )


def run_stage2(
    config: Config,
    tasks: list[TaskInstruction],
    task_candidates: dict[str, list[SearchResult]],
    output_dir: Path,
    *,
    gt_context: GroundTruthContext | None = None,
    tasks_dir: Path | None = None,
) -> None:
    """Stage 2: cross-encoder reranker."""
    _run_rerank_stage(
        config.models.reranker,
        2,
        "reranker",
        config,
        tasks,
        task_candidates,
        output_dir,
        gt_context,
        tasks_dir,
    )


def run_stage3(
    config: Config,
    tasks: list[TaskInstruction],
    task_candidates: dict[str, list[SearchResult]],
    output_dir: Path,
    *,
    gt_context: GroundTruthContext | None = None,
    tasks_dir: Path | None = None,
) -> None:
    """Stage 3: deep reranker."""
    _run_rerank_stage(
        config.models.deep_reranker,
        3,
        "deep_reranker",
        config,
        tasks,
        task_candidates,
        output_dir,
        gt_context,
        tasks_dir,
    )


def run_stage4(
    config: Config,
    tasks: list[TaskInstruction],
    task_candidates: dict[str, list[SearchResult]],
    output_dir: Path,
    *,
    gt_context: GroundTruthContext | None = None,
    tasks_dir: Path | None = None,
) -> None:
    """Stage 4: LLM selector (parallel when max_workers > 1)."""
    sel_cfg = config.models.selector.model_copy(
        update={"cache_path": str(output_dir / "selector_llm_cache.json")},
    )
    selector, k = Selector(sel_cfg), sel_cfg.input_top_k
    inputs = {
        t.task_id: task_candidates[t.task_id][:k]
        if k > 0
        else task_candidates[t.task_id]
        for t in tasks
    }
    max_workers, total, done = max(sel_cfg.max_workers, 1), len(tasks), 0
    eval_results: list[TaskResult] = []
    timings: dict[str, float] = {}
    if max_workers > 1:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logger.info("Stage 4: using %d workers for %d tasks", max_workers, total)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                measure_call,
                select_task,
                selector,
                t.query,
                t.task_id,
                task_candidates[t.task_id],
            ): t
            for t in tasks
        }
        for fut in as_completed(futs):
            t = futs[fut]
            selected, elapsed_ms = fut.result()
            timings[t.task_id] = elapsed_ms
            process_selector_task(
                t,
                selected,  # type: ignore[arg-type]
                inputs,
                task_candidates,
                gt_context,
                sel_cfg.input_top_k,
                eval_results,
            )
            done += 1
            logger.info(
                "Stage 4 [%d/%d] %s: %d selected",
                done,
                total,
                t.task_id,
                len(selected),  # type: ignore[arg-type]
            )
    stage = build_selector_stage_results(tasks, inputs, selector, timings)
    _write_stage_results(
        output_dir / "pipeline-stage4-selector.json",
        "selector",
        stage,
    )
    write_eval_if_needed(
        "stage4-selector",
        eval_results,
        gt_context,
        sel_cfg.input_top_k,
        output_dir,
        tasks_dir or output_dir,
        Path(config.index.output_index_path),
    )

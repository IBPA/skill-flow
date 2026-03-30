"""Extracted helper functions for pipeline stage runners."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from skill_flow.index.encoder import Encoder
from skill_flow.retriever.bm25 import BM25Searcher
from skill_flow.retriever.multi_search import (
    collect_multi_scores,
    rank_from_scores,
    search_multi,
    union_from_collected,
)
from skill_flow.retriever.retriever import IndexSearcher

from .eval_hooks import (
    build_selector_task_result,
    build_task_result,
    get_eval_ks,
    write_eval_report,
)
from .models import (
    PipelineStageResult,
    _to_selector_results,
    _to_stage_results,
)

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.config import RetrieverConfig
    from skill_flow.eval.models import TaskResult
    from skill_flow.query_gen.query_gen import QueryGenerator
    from skill_flow.reranker.reranker import Reranker
    from skill_flow.retriever.protocol import Searcher
    from skill_flow.retriever.retriever import SearchResult
    from skill_flow.selector.selector import Selector

    from .ground_truth import GroundTruthContext
    from .models import TaskInstruction

logger = logging.getLogger(__name__)


# ── Searcher / task-level functions ─────────────────────────────


def build_searcher(config: RetrieverConfig, index_dir: Path) -> Searcher:
    """Build IndexSearcher or BM25Searcher from config."""
    if config.retriever_type == "bm25":
        return BM25Searcher.from_index_dir(index_dir, config)
    encoder = Encoder(config)
    return IndexSearcher(index_dir, encoder, config)


def retrieve_task(
    searcher: Searcher,
    query: str,
    top_k: int,
    query_gen: QueryGenerator | None = None,
    aggregation: str = "max",
    top_k_per_query: int = 0,
    task_id: str = "",
) -> list[SearchResult]:
    """Run retrieval for one task."""
    if query_gen:
        queries = query_gen.generate_multi(task_id, query)
        if len(queries) > 1:
            collected = collect_multi_scores(searcher, queries)
            if aggregation == "union":
                return union_from_collected(collected, top_k_per_query)
            return rank_from_scores(collected, aggregation, top_k)
        return search_multi(searcher, queries, top_k, aggregation)
    return searcher.search(query, top_k=top_k)


def rerank_task(
    reranker: Reranker,
    query: str,
    candidates: list[SearchResult],
    query_gen: QueryGenerator | None = None,
    aggregation: str = "max",
    task_id: str = "",
) -> list[SearchResult]:
    """Run reranking for one task."""
    if query_gen:
        queries = query_gen.generate_multi(task_id, query)
        return reranker.rerank_multi(
            queries,
            candidates,
            aggregation=aggregation,
        )
    return reranker.rerank(query, candidates)


def select_task(
    selector: Selector,
    query: str,
    task_id: str,
    candidates: list[SearchResult],
) -> list[SearchResult]:
    """Run LLM selection for one task."""
    return selector.select(query, candidates, task_id=task_id)


# ── Per-task result collection ──────────────────────────────────


def process_retrieval_task(
    task: TaskInstruction,
    results: list[SearchResult],
    task_candidates: dict[str, list[SearchResult]],
    gt_context: GroundTruthContext | None,
    top_k: int,
    stage_results: list[PipelineStageResult],
    eval_results: list[TaskResult],
    elapsed_ms: float = 0.0,
) -> None:
    """Collect stage result and optional eval metrics for a retrieval task."""
    stage_results.append(_to_stage_results(task, results, elapsed_ms))
    task_candidates[task.task_id] = results
    if gt_context:
        tr = build_task_result(
            task.task_id,
            task.query,
            results,
            gt_context,
            get_eval_ks(top_k),
        )
        if tr:
            eval_results.append(tr)


def process_selector_task(
    task: TaskInstruction,
    selected: list[SearchResult],
    inputs: dict[str, list[SearchResult]],
    task_candidates: dict[str, list[SearchResult]],
    gt_context: GroundTruthContext | None,
    input_top_k: int,
    eval_results: list[TaskResult],
) -> None:
    """Collect eval metrics for a selector task."""
    task_candidates[task.task_id] = selected
    if gt_context:
        tr = build_selector_task_result(
            task.task_id,
            task.query,
            inputs[task.task_id],
            selected,
            gt_context,
            get_eval_ks(input_top_k),
        )
        if tr:
            eval_results.append(tr)


def build_selector_stage_results(
    tasks: list[TaskInstruction],
    inputs: dict[str, list[SearchResult]],
    selector: Selector,
    timings: dict[str, float],
) -> list[PipelineStageResult]:
    """Build PipelineStageResult list for selector stage with timings."""
    return [
        _to_selector_results(
            t,
            inputs[t.task_id],
            *selector.get_labels(t.task_id),
            elapsed_ms=timings.get(t.task_id, 0.0),
        )
        for t in tasks
    ]


# ── Timing and eval utilities ───────────────────────────────────


def measure_call(
    func: object,
    *args: object,
    **kwargs: object,
) -> tuple[object, float]:
    """Call *func* and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)  # type: ignore[operator]
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def write_eval_if_needed(
    stage_name: str,
    eval_results: list[TaskResult],
    gt_context: GroundTruthContext | None,
    top_k: int,
    output_dir: Path,
    tasks_dir: Path,
    index_dir: Path,
) -> None:
    """Write eval report if GT context is present and results exist."""
    if not gt_context or not eval_results:
        return
    ks = get_eval_ks(top_k)
    write_eval_report(
        eval_results,
        gt_context,
        ks,
        output_dir / f"eval-{stage_name}.json",
        tasks_dir,
        index_dir,
    )

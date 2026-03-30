"""Shared grid-search helpers for experiment runners."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from skill_flow.eval.utils.reporting import build_report, write_report
from skill_flow.eval.utils.task_result import build_task_result
from skill_flow.reranker.reranker import aggregate_scores
from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.config import QueryGenConfig
    from skill_flow.eval.models import (
        EvalReport,
        TaskGroundTruth,
        TaskResult,
    )

logger = logging.getLogger(__name__)


def get_num_queries_list(qg: QueryGenConfig | None) -> list[int]:
    """Extract num_queries as a list."""
    if not qg or not qg.enabled:
        return [1]
    nq = qg.num_queries
    return nq if isinstance(nq, list) else [nq]


def get_aggregations(qg: QueryGenConfig | None) -> list[str]:
    """Extract aggregation list from query_gen config."""
    if not qg or not qg.enabled:
        return ["max"]
    agg = qg.aggregation
    return agg if isinstance(agg, list) else [agg]


def update_query_gen_num_queries(
    qg: QueryGenConfig,
    nq: int,
    cache_dir: str = "",
) -> QueryGenConfig:
    """Return a copy of *qg* with a single num_queries and derived cache path."""
    updates: dict[str, object] = {"num_queries": nq}
    if cache_dir:
        updates["cache_path"] = f"{cache_dir.rstrip('/')}/query_gen_cache_{nq}.json"
    return qg.model_copy(update=updates)


def get_top_k_per_query_list(qg: QueryGenConfig | None) -> list[int]:
    """Extract top_k_per_query as a list."""
    if not qg or not qg.enabled:
        return [0]
    tkpq = qg.top_k_per_query
    return tkpq if isinstance(tkpq, list) else [tkpq]


def update_query_gen_aggregation(
    qg: QueryGenConfig,
    agg: str,
) -> QueryGenConfig:
    """Return a copy of *qg* with a single aggregation string."""
    return qg.model_copy(update={"aggregation": agg})


def update_query_gen_top_k_per_query(
    qg: QueryGenConfig,
    tkpq: int,
) -> QueryGenConfig:
    """Return a copy of *qg* with a single top_k_per_query value."""
    return qg.model_copy(update={"top_k_per_query": tkpq})


def reaggregate_report(
    source: EvalReport,
    aggregation: str,
    task_gt_map: dict[str, TaskGroundTruth],
    ks: list[int],
    num_total: int,
    num_skipped: int,
    num_injected: int,
    config: dict[str, object],
    output_path: Path | None = None,
    query_field: str = "rerank_query",
) -> EvalReport:
    """Re-aggregate per-query scores from an existing report.

    *query_field* selects which query to forward to the new task result
    (``"rerank_query"`` for reranker, ``"retrieval_query"`` for retriever).
    """
    task_results: list[TaskResult] = []
    for prev in source.task_results:
        task = task_gt_map.get(prev.task_id)
        if task is None:
            continue
        per_key: dict[str, list[float]] = {
            s.key: s.query_scores for s in prev.retrieved_skills if s.query_scores
        }
        agg_scores = aggregate_scores(per_key, aggregation) if per_key else {}
        reranked = sorted(
            [
                SearchResult(
                    key=s.key,
                    score=agg_scores.get(s.key, s.score),
                    description=s.description,
                    content=s.content,
                    query_scores=s.query_scores,
                )
                for s in prev.retrieved_skills
            ],
            key=lambda r: r.score,
            reverse=True,
        )
        if query_field == "retrieval_query":
            task_results.append(
                build_task_result(
                    task,
                    reranked,
                    ks,
                    retrieval_query=prev.retrieval_query,
                    elapsed_ms=prev.elapsed_ms,
                )
            )
        else:
            task_results.append(
                build_task_result(
                    task,
                    reranked,
                    ks,
                    rerank_query=prev.rerank_query,
                    elapsed_ms=prev.elapsed_ms,
                )
            )

    report = build_report(
        task_results,
        num_total,
        num_skipped,
        num_injected,
        ks,
        config,
    )
    if output_path:
        write_report(report, output_path)
    return report

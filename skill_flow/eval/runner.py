"""Retriever evaluation orchestrator."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from skill_flow.eval.models import (
    EvalReport,
    InjectedSkill,
    PerQueryResult,
    RetrievedSkill,
    TaskResult,
    filter_ks,
)
from skill_flow.eval.utils.ground_truth import load_ground_truth
from skill_flow.eval.utils.reporting import build_report, write_report, write_snapshot
from skill_flow.eval.utils.task_result import build_task_result
from skill_flow.pipeline.query_gen_init import init_retriever_query_gen
from skill_flow.pipeline.stage_helpers import build_searcher
from skill_flow.retriever.multi_search import (
    collect_multi_scores,
    rank_from_scores,
    search_multi,
    union_from_collected,
)

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.config import RetrieverConfig
    from skill_flow.eval.models import TaskGroundTruth
    from skill_flow.query_gen.query_gen import QueryGenerator
    from skill_flow.retriever.protocol import Searcher
    from skill_flow.retriever.retriever import SearchResult

logger = logging.getLogger(__name__)


def _fmt_recall(result: TaskResult) -> str:
    parts = [f"R@{k}={v:.2f}" for k, v in sorted(result.recall_at.items())]
    return " ".join(parts)


def _augment_searcher(
    searcher: Searcher,
    skills: list[InjectedSkill],
) -> None:
    """Inject GT skills into the searcher via its protocol."""
    if not skills:
        return
    searcher.augment([s.key for s in skills], [s.description for s in skills])
    searcher.add_descriptions({s.key: s.description for s in skills})
    searcher.add_contents({s.key: s.content for s in skills if s.content})
    logger.info("Injected %d skills into searcher", len(skills))


def run_evaluation(
    retriever: RetrieverConfig,
    tasks_dir: Path,
    index_dir: Path,
    max_query_chars: int = 0,
    max_tasks: int = 0,
    include_tasks: list[str] | None = None,
    output_path: Path | None = None,
) -> EvalReport:
    """Run a full retriever evaluation against SkillsBench ground truth."""
    tasks, injected_skills, skipped = load_ground_truth(tasks_dir, max_query_chars)
    searcher = build_searcher(retriever, index_dir)
    _augment_searcher(searcher, injected_skills)

    qg, agg, tkpq = init_retriever_query_gen(retriever.query_gen)
    is_union = agg == "union"
    ks = filter_ks(0) if is_union else filter_ks(retriever.top_k)

    eval_tasks = tasks[:max_tasks] if max_tasks > 0 else tasks
    if include_tasks:
        inc = set(include_tasks)
        eval_tasks = [t for t in eval_tasks if t.task_id in inc]

    cfg: dict[str, object] = {
        "tasks_dir": str(tasks_dir),
        "index_dir": str(index_dir),
        "retriever": retriever.model_dump(mode="json"),
        "max_query_chars": max_query_chars,
        "output_path": str(output_path) if output_path else None,
    }
    n_total = len(tasks) + len(skipped)
    task_results: list[TaskResult] = []

    for task in eval_tasks:
        t0 = time.perf_counter()
        results, rq, pq = _retrieve_one(
            searcher,
            task,
            retriever.top_k,
            qg,
            agg,
            tkpq,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result = build_task_result(
            task,
            results,
            ks,
            retrieval_query=rq,
            retrieval_queries=pq,
            elapsed_ms=elapsed_ms,
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


def _retrieve_one(
    searcher: Searcher,
    task: TaskGroundTruth,
    top_k: int,
    qg: QueryGenerator | None,
    agg: str,
    tkpq: int,
) -> tuple[list[SearchResult], str, list[PerQueryResult]]:
    """Retrieve results for one task, returning (results, query, per_query)."""
    per_query: list[PerQueryResult] = []
    if qg:
        queries = qg.generate_multi(task.task_id, task.query)
        if len(queries) > 1:
            collected = collect_multi_scores(searcher, queries)
            if agg == "union":
                results = union_from_collected(collected, tkpq)
            else:
                results = rank_from_scores(collected, agg, top_k)
            per_query = [
                PerQueryResult(
                    query=q,
                    retrieved_skills=[
                        RetrievedSkill(
                            key=r.key,
                            score=r.score,
                            description=r.description,
                        )
                        for r in qr[:top_k]
                    ],
                )
                for q, qr in zip(
                    collected.queries,
                    collected.per_query_results,
                    strict=True,
                )
            ]
        else:
            results = search_multi(searcher, queries, top_k, agg)
        return results, " | ".join(queries), per_query
    results = searcher.search(task.query, top_k=top_k)
    return results, "", per_query

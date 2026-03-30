"""Reranker comparison experiment runner with grid-search aggregation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from skill_flow.config import RerankerConfig
from skill_flow.eval.experiments.grid import (
    get_aggregations,
    get_num_queries_list,
    reaggregate_report,
    update_query_gen_aggregation,
    update_query_gen_num_queries,
)
from skill_flow.eval.models import filter_ks
from skill_flow.eval.runner_stages import run_reranker_evaluation
from skill_flow.eval.utils.ground_truth import load_ground_truth
from skill_flow.eval.utils.helpers import slug

if TYPE_CHECKING:
    from skill_flow.config import RerankerExperimentConfig, RerankerVariant
    from skill_flow.eval.models import EvalReport, TaskGroundTruth

logger = logging.getLogger(__name__)


def _get_content_chars_list(variant: RerankerVariant) -> list[int]:
    cc = variant.max_content_chars
    return cc if isinstance(cc, list) else [cc]


def _set_content_chars(variant: RerankerVariant, chars: int) -> RerankerVariant:
    return variant.model_copy(update={"max_content_chars": chars})


def _set_variant_num_queries(
    variant: RerankerVariant,
    nq: int,
    cache_dir: str = "",
) -> RerankerVariant:
    qg = variant.query_gen
    if not qg:
        return variant
    return variant.model_copy(
        update={"query_gen": update_query_gen_num_queries(qg, nq, cache_dir)},
    )


def _set_variant_aggregation(variant: RerankerVariant, agg: str) -> RerankerVariant:
    qg = variant.query_gen
    if not qg:
        return variant
    return variant.model_copy(
        update={"query_gen": update_query_gen_aggregation(qg, agg)},
    )


def _to_reranker_config(variant: RerankerVariant) -> RerankerConfig:
    cc = variant.max_content_chars
    if isinstance(cc, list):
        msg = "max_content_chars must be a single int; expand list variants first"
        raise TypeError(msg)
    if variant.query_gen and isinstance(variant.query_gen.num_queries, list):
        msg = "num_queries must be a single int; expand list variants first"
        raise TypeError(msg)
    return RerankerConfig(
        enabled=True,
        model_name=variant.model_name,
        input_top_k=variant.input_top_k,
        batch_size=variant.batch_size,
        max_content_chars=cc,
        query_gen=variant.query_gen,
    )


def _reranker_label(
    variant: RerankerVariant,
    include_tasks: list[str] | None = None,
) -> str:
    label = slug(variant.model_name)
    cc = variant.max_content_chars
    if isinstance(cc, int) and cc > 0:
        label = f"{label}-{cc}chars"
    qg = variant.query_gen
    if qg and qg.enabled and isinstance(qg.num_queries, int):
        label = f"{label}-q{qg.num_queries}-{qg.aggregation}"
    if include_tasks:
        label = f"{label}-i{len(include_tasks)}"
    return label


def _score_and_reaggregate(
    variant: RerankerVariant,
    aggregations: list[str],
    config: RerankerExperimentConfig,
    input_path: Path,
    index_dir: Path,
    output_dir: Path,
    results: list[tuple[str, EvalReport]],
) -> EvalReport | None:
    first = _set_variant_aggregation(variant, aggregations[0])
    label = _reranker_label(first, config.include_tasks)
    out = output_dir / f"{label}.json"
    reranker_cfg = _to_reranker_config(first)

    report = run_reranker_evaluation(
        input_path,
        reranker_cfg,
        Path(config.tasks_dir),
        index_dir,
        max_tasks=config.max_tasks,
        include_tasks=config.include_tasks or None,
        output_path=out,
    )
    results.append((label, report))

    if len(aggregations) > 1:
        tasks, injected, skipped = load_ground_truth(Path(config.tasks_dir))
        gt_map: dict[str, TaskGroundTruth] = {t.task_id: t for t in tasks}
        ks = filter_ks(variant.input_top_k)
        n_total = len(tasks) + len(skipped)

        for agg in aggregations[1:]:
            agg_v = _set_variant_aggregation(variant, agg)
            agg_label = _reranker_label(agg_v, config.include_tasks)
            agg_out = output_dir / f"{agg_label}.json"
            agg_cfg: dict[str, object] = {
                "prev_report_path": str(input_path),
                "tasks_dir": config.tasks_dir,
                "index_dir": str(index_dir),
                "reranker": _to_reranker_config(agg_v).model_dump(mode="json"),
                "output_path": str(agg_out),
            }
            agg_report = reaggregate_report(
                report,
                agg,
                gt_map,
                ks,
                n_total,
                len(skipped),
                len(injected),
                agg_cfg,
                agg_out,
            )
            results.append((agg_label, agg_report))
            logger.info("Re-aggregated %s with %s", agg_label, agg)

    return report


def run_reranker_experiment(
    config: RerankerExperimentConfig,
) -> list[tuple[str, EvalReport]]:
    """Run reranker eval for each variant, return (label, report) pairs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(config.input_report_path)
    index_dir = Path(config.index_dir)

    results: list[tuple[str, EvalReport]] = []
    for variant in config.rerankers:
        if not variant.enabled:
            continue
        for chars in _get_content_chars_list(variant):
            for nq in get_num_queries_list(variant.query_gen):
                expanded = _set_variant_num_queries(
                    _set_content_chars(variant, chars),
                    nq,
                    config.cache_dir,
                )
                aggs = ["max"] if nq == 1 else get_aggregations(variant.query_gen)
                _score_and_reaggregate(
                    expanded,
                    aggs,
                    config,
                    input_path,
                    index_dir,
                    output_dir,
                    results,
                )

    return results

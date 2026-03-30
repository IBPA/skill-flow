"""Retriever comparison experiment runner."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from skill_flow.config import RetrieverConfig
from skill_flow.corpus.loader import load_corpus
from skill_flow.eval.experiments.grid import (
    get_aggregations,
    get_num_queries_list,
    get_top_k_per_query_list,
    reaggregate_report,
    update_query_gen_aggregation,
    update_query_gen_num_queries,
    update_query_gen_top_k_per_query,
)
from skill_flow.eval.models import PerQueryResult, RetrievedSkill, filter_ks
from skill_flow.eval.runner import _augment_searcher, run_evaluation
from skill_flow.eval.utils.ground_truth import load_ground_truth
from skill_flow.eval.utils.helpers import slug
from skill_flow.eval.utils.reporting import build_report, write_report
from skill_flow.eval.utils.task_result import build_task_result
from skill_flow.index.builder import build_index
from skill_flow.index.encoder import Encoder, pick_device
from skill_flow.pipeline.stage_helpers import build_searcher
from skill_flow.query_gen.query_gen import QueryGenerator
from skill_flow.retriever.multi_search import (
    CollectedScores,
    collect_multi_scores,
    rank_from_scores,
    union_from_collected,
)

if TYPE_CHECKING:
    from skill_flow.config import RetrieverExperimentConfig, RetrieverVariant
    from skill_flow.eval.models import EvalReport, TaskGroundTruth, TaskResult

logger = logging.getLogger(__name__)


def _to_cfg(v: RetrieverVariant) -> RetrieverConfig:
    return RetrieverConfig(
        model_name=v.model_name,
        query_prompt=v.query_prompt,
        doc_prompt=v.doc_prompt,
        retriever_type=v.retriever_type,
        top_k=v.top_k,
        query_gen=v.query_gen,
    )


def _ensure_index(variant: RetrieverVariant, corpus_path: str) -> None:
    idx = Path(variant.index_dir)
    if variant.retriever_type == "bm25":
        if not (idx / "skill_ids.json").exists():
            msg = f"BM25 requires pre-built artifacts in {idx}."
            raise FileNotFoundError(msg)
        return
    if (idx / "faiss.index").exists():
        return
    logger.info("Index not found at %s, building ...", idx)
    rc = _to_cfg(variant)
    cp = Path(corpus_path)
    enc = Encoder(rc, device=variant.device or pick_device())
    build_index(
        load_corpus(cp),
        enc,
        idx,
        batch_size=variant.batch_size,
        corpus_path=cp,
        max_content_tokens=variant.max_content_tokens,
    )


def _retriever_label(v: RetrieverVariant, inc: list[str] | None = None) -> str:
    base = slug(Path(v.index_dir).name)
    label = f"bm25-{base}" if v.retriever_type == "bm25" else base
    qg = v.query_gen
    if qg and qg.enabled and isinstance(qg.num_queries, int):
        label = f"{label}-q{qg.num_queries}-{qg.aggregation}"
        if (
            qg.aggregation == "union"
            and isinstance(qg.top_k_per_query, int)
            and qg.top_k_per_query > 0
        ):
            label = f"{label}-tk{qg.top_k_per_query}"
    if inc:
        label = f"{label}-i{len(inc)}"
    return label


def _with_num_queries(v: RetrieverVariant, nq: int, cd: str = "") -> RetrieverVariant:
    if not v.query_gen:
        return v
    return v.model_copy(
        update={"query_gen": update_query_gen_num_queries(v.query_gen, nq, cd)}
    )


def _with_top_k_per_query(v: RetrieverVariant, tkpq: int) -> RetrieverVariant:
    if not v.query_gen:
        return v
    return v.model_copy(
        update={"query_gen": update_query_gen_top_k_per_query(v.query_gen, tkpq)}
    )


def _with_aggregation(v: RetrieverVariant, agg: str) -> RetrieverVariant:
    if not v.query_gen:
        return v
    return v.model_copy(
        update={"query_gen": update_query_gen_aggregation(v.query_gen, agg)}
    )


def _cfg_dict(
    v: RetrieverVariant, c: RetrieverExperimentConfig, out: Path,
) -> dict[str, object]:
    return {
        "tasks_dir": c.tasks_dir,
        "index_dir": v.index_dir,
        "retriever": _to_cfg(v).model_dump(mode="json"),
        "max_tasks": c.max_tasks,
        "include_tasks": c.include_tasks,
        "output_path": str(out),
    }


def _score_grid(
    exp: RetrieverVariant,
    aggs: list[str],
    cache: list[tuple[TaskGroundTruth, CollectedScores, str, float]],
    tasks: list[TaskGroundTruth],
    cfg: RetrieverExperimentConfig,
    out_dir: Path,
    ks: list[int],
    n_total: int,
    n_skip: int,
    n_inj: int,
    results: list[tuple[str, EvalReport]],
) -> None:
    first_v = _with_aggregation(exp, aggs[0])
    lbl = _retriever_label(first_v, cfg.include_tasks)
    out = out_dir / f"{lbl}.json"
    cd = _cfg_dict(first_v, cfg, out)
    trs: list[TaskResult] = []
    for task, collected, rq, elapsed_ms in cache:
        ranked = rank_from_scores(collected, aggs[0], exp.top_k)
        pq = [
            PerQueryResult(
                query=q,
                retrieved_skills=[
                    RetrievedSkill(key=r.key, score=r.score, description=r.description)
                    for r in qr[: exp.top_k]
                ],
            )
            for q, qr in zip(
                collected.queries, collected.per_query_results, strict=True
            )
        ]
        trs.append(
            build_task_result(
                task, ranked, ks, retrieval_query=rq, retrieval_queries=pq,
                elapsed_ms=elapsed_ms,
            )
        )
    first_rpt = build_report(trs, n_total, n_skip, n_inj, ks, cd)
    write_report(first_rpt, out)
    results.append((lbl, first_rpt))
    if len(aggs) > 1:
        gt_map = {t.task_id: t for t in tasks}
        for agg in aggs[1:]:
            av = _with_aggregation(exp, agg)
            al = _retriever_label(av, cfg.include_tasks)
            ao = out_dir / f"{al}.json"
            ac = _cfg_dict(av, cfg, ao)
            rpt = reaggregate_report(
                first_rpt, agg, gt_map, ks, n_total, n_skip, n_inj,
                ac, ao, "retrieval_query",
            )
            results.append((al, rpt))


def _union_grid(
    exp: RetrieverVariant,
    cache: list[tuple[TaskGroundTruth, CollectedScores, str, float]],
    cfg: RetrieverExperimentConfig,
    out_dir: Path,
    n_total: int,
    n_skip: int,
    n_inj: int,
    results: list[tuple[str, EvalReport]],
) -> None:
    for tkpq in get_top_k_per_query_list(exp.query_gen):
        if tkpq <= 0:
            continue
        uv = _with_top_k_per_query(_with_aggregation(exp, "union"), tkpq)
        lbl = _retriever_label(uv, cfg.include_tasks)
        out = out_dir / f"{lbl}.json"
        cd = _cfg_dict(uv, cfg, out)
        uks = filter_ks(0)
        trs: list[TaskResult] = []
        for task, collected, rq, elapsed_ms in cache:
            trs.append(
                build_task_result(
                    task, union_from_collected(collected, tkpq), uks,
                    retrieval_query=rq, elapsed_ms=elapsed_ms,
                )
            )
        rpt = build_report(trs, n_total, n_skip, n_inj, uks, cd)
        write_report(rpt, out)
        results.append((lbl, rpt))


def _run_grid(
    exp: RetrieverVariant,
    aggs: list[str],
    cfg: RetrieverExperimentConfig,
    out_dir: Path,
    results: list[tuple[str, EvalReport]],
) -> None:
    tasks, inj, skipped = load_ground_truth(Path(cfg.tasks_dir))
    ks = filter_ks(exp.top_k)
    nt = len(tasks) + len(skipped)
    et = tasks[: cfg.max_tasks] if cfg.max_tasks > 0 else tasks
    if cfg.include_tasks:
        inc = set(cfg.include_tasks)
        et = [t for t in et if t.task_id in inc]
    searcher = build_searcher(_to_cfg(exp), Path(exp.index_dir))
    _augment_searcher(searcher, inj)
    qg = (
        QueryGenerator(exp.query_gen)
        if exp.query_gen and exp.query_gen.enabled
        else None
    )
    tc: list[tuple[TaskGroundTruth, CollectedScores, str, float]] = []
    for task in et:
        t0 = time.perf_counter()
        qs = qg.generate_multi(task.task_id, task.query) if qg else [task.query]
        collected = collect_multi_scores(searcher, qs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        tc.append((task, collected, " | ".join(qs), elapsed_ms))
    sa = [a for a in aggs if a != "union"]
    if sa:
        _score_grid(
            exp, sa, tc, tasks, cfg, out_dir, ks, nt, len(skipped), len(inj), results,
        )
    if "union" in aggs:
        _union_grid(exp, tc, cfg, out_dir, nt, len(skipped), len(inj), results)


def run_experiment(cfg: RetrieverExperimentConfig) -> list[tuple[str, EvalReport]]:
    """Run retriever eval for each variant, return (label, report) pairs."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, EvalReport]] = []
    for v in cfg.retrievers:
        if not v.enabled:
            continue
        _ensure_index(v, cfg.corpus_path)
        for nq in get_num_queries_list(v.query_gen):
            exp = _with_num_queries(v, nq, cfg.cache_dir)
            aggs = ["max"] if nq == 1 else get_aggregations(v.query_gen)
            if nq > 1 and (len(aggs) > 1 or "union" in aggs):
                _run_grid(exp, aggs, cfg, out_dir, results)
            else:
                first = _with_aggregation(exp, aggs[0])
                lbl = _retriever_label(first, cfg.include_tasks)
                out = out_dir / f"{lbl}.json"
                rpt = run_evaluation(
                    _to_cfg(first),
                    Path(cfg.tasks_dir),
                    Path(first.index_dir),
                    max_tasks=cfg.max_tasks,
                    include_tasks=cfg.include_tasks or None,
                    output_path=out,
                )
                results.append((lbl, rpt))
    return results


def print_comparison(results: list[tuple[str, EvalReport]]) -> None:
    """Print a comparison table of retriever results."""
    ks = [1, 5, 10, 100]
    h = f"{'Model':<25} {'MRR':>7}"
    h += "".join(f" {'R@' + str(k):>7}" for k in ks)
    sep = "=" * len(h)
    print(f"\n{sep}\n{h}\n{'-' * len(h)}")
    for lbl, rpt in results:
        s = rpt.summary
        vals = " ".join(f"{s.mean_recall_at.get(k, 0.0):>7.4f}" for k in ks)
        print(f"{lbl:<25} {s.mrr:>7.4f} {vals}")
    print(sep)

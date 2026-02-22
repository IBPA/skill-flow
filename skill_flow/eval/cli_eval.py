"""CLI helpers for evaluation commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skill_flow.eval.models import (
    EvalRunConfig,
    Reranker2EvalConfig,
    RerankerEvalConfig,
    SelectorEvalConfig,
)
from skill_flow.eval.runner import (
    run_evaluation,
    run_reranker2_evaluation,
    run_reranker_evaluation,
    run_selector_evaluation,
)

if TYPE_CHECKING:
    import argparse

    from skill_flow.config import Config
    from skill_flow.eval.models import EvalSummary


def print_summary(
    s: EvalSummary,
    output_path: Path | None,
    label: str = "",
) -> None:
    """Print evaluation summary to stdout."""
    if label:
        print(f"\n  [{label}]")
    print(f"\n{'=' * 50}")
    print(f"  Tasks: {s.num_tasks_evaluated} evaluated / {s.num_tasks_total} total")
    print(f"  Skipped (no skills): {s.num_tasks_no_skills}")
    print(f"  Skills injected: {s.num_skills_injected}")
    print(f"  MRR: {s.mrr:.4f}")
    print(f"{'=' * 50}")
    print(f"  {'k':>5}  {'Recall@k':>10}  {'Prec@k':>10}  {'Hit@k':>10}")
    print(f"  {'-' * 42}")
    for k in sorted(s.mean_recall_at):
        print(
            f"  {k:>5}  {s.mean_recall_at[k]:>10.4f}"
            f"  {s.mean_precision_at[k]:>10.4f}"
            f"  {s.mean_hit_at[k]:>10.4f}"
        )
    print(f"{'=' * 50}")
    if output_path:
        print(f"\n  Report: {output_path}")


def run_retriever_eval(args: argparse.Namespace, config: Config) -> None:
    """Run Stage 1 retriever evaluation."""
    eval_settings = config.models.retriever.eval

    if args.tasks_dir is not None:
        tasks_dir: str = args.tasks_dir
    elif eval_settings is not None:
        tasks_dir = eval_settings.tasks_dir
    else:
        raise SystemExit(
            "Error: --tasks-dir is required when models.retriever.eval "
            "is not configured"
        )

    rerank_enabled = (
        args.rerank if args.rerank is not None else config.models.reranker.enabled
    )
    reranker_cfg = config.models.reranker if rerank_enabled else None

    output: str | None = args.output
    if output is None and eval_settings is not None:
        output = eval_settings.output_path

    eval_config = EvalRunConfig(
        tasks_dir=Path(tasks_dir),
        index_dir=Path(args.index_dir or config.index.output_index_path),
        retriever=config.models.retriever,
        max_query_chars=args.max_query_chars,
        max_tasks=args.max_tasks,
        output_path=Path(output) if output else None,
        reranker=reranker_cfg,
    )

    report = run_evaluation(eval_config)
    print_summary(report.summary, eval_config.output_path, label="Retriever")


def run_reranker_eval(args: argparse.Namespace, config: Config) -> None:
    """Run Stage 2 reranker evaluation on cached Stage 1 results."""
    eval_settings = config.models.reranker.eval
    if eval_settings is None:
        raise SystemExit(
            "Error: models.reranker.eval is not configured"
        )

    eval_config = RerankerEvalConfig(
        stage1_report_path=Path(eval_settings.stage1_report_path),
        tasks_dir=Path(eval_settings.tasks_dir),
        index_dir=Path(config.index.output_index_path),
        reranker=config.models.reranker.model_copy(update={"enabled": True}),
        max_tasks=args.max_tasks,
        output_path=Path(eval_settings.output_path),
    )

    report = run_reranker_evaluation(eval_config)
    print_summary(report.summary, eval_config.output_path, label="Reranker")


def run_reranker2_eval(args: argparse.Namespace, config: Config) -> None:
    """Run Stage 3 reranker2 evaluation on cached Stage 2 results."""
    eval_settings = config.models.reranker2.eval
    if eval_settings is None:
        raise SystemExit(
            "Error: models.reranker2.eval is not configured"
        )

    eval_config = Reranker2EvalConfig(
        stage2_report_path=Path(eval_settings.stage2_report_path),
        tasks_dir=Path(eval_settings.tasks_dir),
        index_dir=Path(config.index.output_index_path),
        reranker=config.models.reranker2.model_copy(update={"enabled": True}),
        max_tasks=args.max_tasks,
        output_path=Path(eval_settings.output_path),
    )

    report = run_reranker2_evaluation(eval_config)
    print_summary(report.summary, eval_config.output_path, label="Reranker2")


def run_selector_eval(args: argparse.Namespace, config: Config) -> None:
    """Run Stage 4 selector evaluation on cached Stage 3 results."""
    eval_settings = config.models.selector.eval
    if eval_settings is None:
        raise SystemExit(
            "Error: models.selector.eval is not configured"
        )

    eval_config = SelectorEvalConfig(
        stage3_report_path=Path(eval_settings.stage3_report_path),
        tasks_dir=Path(eval_settings.tasks_dir),
        index_dir=Path(config.index.output_index_path),
        selector=config.models.selector.model_copy(update={"enabled": True}),
        max_tasks=args.max_tasks,
        output_path=Path(eval_settings.output_path),
    )

    report = run_selector_evaluation(eval_config)
    print_summary(report.summary, eval_config.output_path, label="Selector")


def run_eval(args: argparse.Namespace, config: Config) -> None:
    """Run evaluation stages based on config."""
    retriever_eval = config.models.retriever.eval
    reranker_eval = config.models.reranker.eval
    reranker2_eval = config.models.reranker2.eval
    selector_eval = config.models.selector.eval

    run_retriever = retriever_eval is not None and retriever_eval.enabled
    run_reranker = reranker_eval is not None and reranker_eval.enabled
    run_reranker2 = reranker2_eval is not None and reranker2_eval.enabled
    run_selector = selector_eval is not None and selector_eval.enabled

    if not any([run_retriever, run_reranker, run_reranker2, run_selector]):
        raise SystemExit("Error: no eval stages enabled in config")

    if run_retriever:
        run_retriever_eval(args, config)

    if run_reranker:
        run_reranker_eval(args, config)

    if run_reranker2:
        run_reranker2_eval(args, config)

    if run_selector:
        run_selector_eval(args, config)

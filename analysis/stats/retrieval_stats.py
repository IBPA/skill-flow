"""Bridge between EvalReport/TaskResult and statistical utilities.

Extracts per-task retrieval metrics from evaluation reports and computes
bootstrap CIs and paired hypothesis tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from analysis.stats.bootstrap import bootstrap_ci, paired_bootstrap_test

if TYPE_CHECKING:
    from skill_flow.eval.models import EvalReport

    from analysis.stats.types import ConfidenceInterval, PairedTestResult


def _extract_metric(
    report: EvalReport,
    metric: str,
    k: int | None = None,
) -> list[float]:
    """Extract per-task values for a retrieval metric.

    Args:
        report: Evaluation report with per-task results.
        metric: One of "recall", "precision", "hit", "mrr",
                "select_recall", "select_precision".
        k: Required for recall/precision/hit; ignored for mrr.

    Returns:
        List of per-task metric values.
    """
    values: list[float] = []
    for tr in report.task_results:
        if metric == "recall":
            values.append(tr.recall_at.get(k, 0.0) if k is not None else 0.0)
        elif metric == "precision":
            values.append(tr.precision_at.get(k, 0.0) if k is not None else 0.0)
        elif metric == "hit":
            values.append(tr.hit_at.get(k, 0.0) if k is not None else 0.0)
        elif metric == "mrr":
            values.append(tr.reciprocal_rank)
        elif metric == "select_recall":
            values.append(tr.select_recall)
        elif metric == "select_precision":
            values.append(tr.select_precision)
        else:
            msg = f"Unknown metric: {metric!r}"
            raise ValueError(msg)
    return values


def retrieval_ci(
    report: EvalReport,
    metric: str,
    k: int | None = None,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ConfidenceInterval:
    """Bootstrap CI for a retrieval metric averaged across tasks.

    Args:
        report: Evaluation report.
        metric: Metric name (see ``_extract_metric``).
        k: The k parameter for recall/precision/hit.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level.
        seed: RNG seed.

    Returns:
        ConfidenceInterval for the mean metric across tasks.
    """
    values = _extract_metric(report, metric, k)
    return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)


def retrieval_paired_test(
    report_a: EvalReport,
    report_b: EvalReport,
    metric: str,
    k: int | None = None,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairedTestResult:
    """Paired bootstrap test comparing two reports on a retrieval metric.

    Reports are aligned by task_id. Only tasks present in both are compared.

    Args:
        report_a: First report (e.g. treatment).
        report_b: Second report (e.g. baseline).
        metric: Metric name.
        k: The k parameter for recall/precision/hit.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level.
        seed: RNG seed.

    Returns:
        PairedTestResult for the mean difference (A - B).
    """
    map_a = {tr.task_id: tr for tr in report_a.task_results}
    map_b = {tr.task_id: tr for tr in report_b.task_results}
    shared = sorted(set(map_a) & set(map_b))

    vals_a: list[float] = []
    vals_b: list[float] = []
    for tid in shared:
        tr_a = map_a[tid]
        tr_b = map_b[tid]
        if metric == "mrr":
            vals_a.append(tr_a.reciprocal_rank)
            vals_b.append(tr_b.reciprocal_rank)
        elif metric in ("recall", "precision", "hit") and k is not None:
            attr = {"recall": "recall_at", "precision": "precision_at", "hit": "hit_at"}
            vals_a.append(getattr(tr_a, attr[metric]).get(k, 0.0))
            vals_b.append(getattr(tr_b, attr[metric]).get(k, 0.0))
        elif metric == "select_recall":
            vals_a.append(tr_a.select_recall)
            vals_b.append(tr_b.select_recall)
        elif metric == "select_precision":
            vals_a.append(tr_a.select_precision)
            vals_b.append(tr_b.select_precision)
        else:
            msg = f"Unknown metric: {metric!r}"
            raise ValueError(msg)

    return paired_bootstrap_test(vals_a, vals_b, n_boot=n_boot, alpha=alpha, seed=seed)


def all_retrieval_cis(
    report: EvalReport,
    ks: list[int],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    """Compute bootstrap CIs for all retrieval metrics at all k values.

    Returns a dict keyed by metric labels like "R@10", "P@5", "MRR".
    """
    cis: dict[str, ConfidenceInterval] = {}
    for k in ks:
        cis[f"R@{k}"] = retrieval_ci(
            report, "recall", k, n_boot=n_boot, alpha=alpha, seed=seed
        )
        cis[f"P@{k}"] = retrieval_ci(
            report, "precision", k, n_boot=n_boot, alpha=alpha, seed=seed
        )
    cis["MRR"] = retrieval_ci(report, "mrr", n_boot=n_boot, alpha=alpha, seed=seed)
    return cis

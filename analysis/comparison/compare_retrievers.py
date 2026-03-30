"""Compare per-task retrieval results between two retriever reports.

Loads two EvalReport JSON files, aligns them by task_id, and prints:
  1. Summary metrics side-by-side
  2. Per-task win/loss/tie breakdown at a chosen k
  3. Biggest wins and losses (sorted by RR delta)
  4. Scatter plot of reciprocal rank (A vs B)

Usage::

    uv run python -m analysis.eda.compare_retrievers \\
        outputs/experiments/retriever-comparison/bge-m3.json \\
        outputs/experiments/retriever-comparison/bm25-bge-base.json \\
        --k 10 --top-n 10 --output-dir outputs/eda/compare
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel
from skill_flow.eval.models import EvalReport, TaskResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------


class TaskComparison(BaseModel, frozen=True):
    """Per-task delta between two retrievers."""

    task_id: str
    query: str
    rr_a: float
    rr_b: float
    rr_delta: float  # A - B  (positive = A wins)
    recall_a: float
    recall_b: float
    recall_delta: float
    hit_a: bool
    hit_b: bool
    rank_a: int  # 0 = not found
    rank_b: int


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _first_gt_rank(task: TaskResult) -> int:
    """Return 1-based rank of first GT skill, or 0 if none found."""
    rr = task.reciprocal_rank
    return round(1 / rr) if rr > 0 else 0


def _load_report(path: Path) -> EvalReport:
    return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))


def _label(path: Path) -> str:
    return path.stem


def _build_comparisons(
    report_a: EvalReport,
    report_b: EvalReport,
    k: int,
) -> list[TaskComparison]:
    map_b = {t.task_id: t for t in report_b.task_results}
    comparisons: list[TaskComparison] = []
    for ta in report_a.task_results:
        tb = map_b.get(ta.task_id)
        if tb is None:
            continue
        rr_a, rr_b = ta.reciprocal_rank, tb.reciprocal_rank
        recall_a = ta.recall_at.get(k, 0.0)
        recall_b = tb.recall_at.get(k, 0.0)
        hit_a = recall_a > 0
        hit_b = recall_b > 0
        comparisons.append(
            TaskComparison(
                task_id=ta.task_id,
                query=ta.query[:120],
                rr_a=rr_a,
                rr_b=rr_b,
                rr_delta=rr_a - rr_b,
                recall_a=recall_a,
                recall_b=recall_b,
                recall_delta=recall_a - recall_b,
                hit_a=hit_a,
                hit_b=hit_b,
                rank_a=_first_gt_rank(ta),
                rank_b=_first_gt_rank(tb),
            )
        )
    return comparisons


# ------------------------------------------------------------------
# Printing
# ------------------------------------------------------------------


def _print_summary(
    report_a: EvalReport,
    report_b: EvalReport,
    label_a: str,
    label_b: str,
) -> None:
    sa, sb = report_a.summary, report_b.summary
    w = max(len(label_a), len(label_b), 12)
    print(f"\n{'=' * (w + 50)}")
    print(f"{'Metric':<15} {label_a:>{w}}  {label_b:>{w}}  {'Delta':>8}")
    print(f"{'-' * (w + 50)}")

    rows = [("MRR", sa.mrr, sb.mrr)]
    for k_val in sorted(sa.mean_recall_at, key=lambda x: int(x)):
        ra = sa.mean_recall_at.get(k_val, 0.0)
        rb = sb.mean_recall_at.get(k_val, 0.0)
        rows.append((f"R@{k_val}", ra, rb))
    for k_val in sorted(sa.mean_hit_at, key=lambda x: int(x)):
        ha = sa.mean_hit_at.get(k_val, 0.0)
        hb = sb.mean_hit_at.get(k_val, 0.0)
        rows.append((f"Hit@{k_val}", ha, hb))

    for label, va, vb in rows:
        delta = va - vb
        sign = "+" if delta > 0 else ""
        print(f"{label:<15} {va:>{w}.4f}  {vb:>{w}.4f}  {sign}{delta:>7.4f}")
    print(f"{'=' * (w + 50)}")


def _print_win_loss(
    comparisons: list[TaskComparison],
    label_a: str,
    label_b: str,
    k: int,
) -> None:
    wins_a = sum(1 for c in comparisons if c.rr_delta > 1e-9)
    wins_b = sum(1 for c in comparisons if c.rr_delta < -1e-9)
    ties = len(comparisons) - wins_a - wins_b

    print(f"\nWin/Loss/Tie (by reciprocal rank, k={k}):")
    print(f"  {label_a} wins: {wins_a}")
    print(f"  {label_b} wins: {wins_b}")
    print(f"  Ties:           {ties}")

    only_a = sum(1 for c in comparisons if c.hit_a and not c.hit_b)
    only_b = sum(1 for c in comparisons if c.hit_b and not c.hit_a)
    both = sum(1 for c in comparisons if c.hit_a and c.hit_b)
    neither = sum(1 for c in comparisons if not c.hit_a and not c.hit_b)

    print(f"\nHit@{k} breakdown:")
    print(f"  Both hit:       {both}")
    print(f"  Only {label_a}: {only_a}")
    print(f"  Only {label_b}: {only_b}")
    print(f"  Neither:        {neither}")


def _print_top_deltas(
    comparisons: list[TaskComparison],
    label_a: str,
    label_b: str,
    top_n: int,
) -> None:
    sorted_by_delta = sorted(comparisons, key=lambda c: c.rr_delta, reverse=True)

    col_w = 30
    hdr = (
        f"  {'task_id':<{col_w}}"
        f" {'rank_A':>6} {'rank_B':>6}"
        f" {'RR_A':>7} {'RR_B':>7} {'delta':>7}"
    )

    print(f"\nTop {top_n} wins for {label_a} (A ranks better):")
    print(hdr)
    for c in sorted_by_delta[:top_n]:
        print(
            f"  {c.task_id:<{col_w}}"
            f" {c.rank_a or '-':>6} {c.rank_b or '-':>6}"
            f" {c.rr_a:>7.4f} {c.rr_b:>7.4f} {c.rr_delta:>+7.4f}"
        )

    print(f"\nTop {top_n} wins for {label_b} (B ranks better):")
    print(hdr)
    for c in sorted_by_delta[-top_n:][::-1]:
        print(
            f"  {c.task_id:<{col_w}}"
            f" {c.rank_a or '-':>6} {c.rank_b or '-':>6}"
            f" {c.rr_a:>7.4f} {c.rr_b:>7.4f} {c.rr_delta:>+7.4f}"
        )


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------


def _plot_rr_scatter(
    comparisons: list[TaskComparison],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    rng = np.random.default_rng(42)
    jitter = 0.012
    rr_a = np.array([c.rr_a for c in comparisons])
    rr_b = np.array([c.rr_b for c in comparisons])
    rr_a_j = rr_a + rng.uniform(-jitter, jitter, size=len(rr_a))
    rr_b_j = rr_b + rng.uniform(-jitter, jitter, size=len(rr_b))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(rr_a_j, rr_b_j, alpha=0.5, s=30, edgecolors="black", linewidths=0.3)
    ax.plot([0, 1], [0, 1], "r--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(f"RR — {label_a}")
    ax.set_ylabel(f"RR — {label_b}")
    ax.set_title("Reciprocal Rank: A vs B")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()

    path = output_dir / "rr_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {path}")


def _plot_rank_delta_bar(
    comparisons: list[TaskComparison],
    label_a: str,
    label_b: str,
    output_dir: Path,
) -> None:
    sorted_c = sorted(comparisons, key=lambda c: c.rr_delta, reverse=True)
    task_ids = [c.task_id for c in sorted_c]
    deltas = [c.rr_delta for c in sorted_c]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" if d < 0 else "#95a5a6" for d in deltas]

    fig, ax = plt.subplots(figsize=(12, max(5, len(task_ids) * 0.22)))
    ax.barh(
        range(len(task_ids)), deltas, color=colors, edgecolor="black", linewidth=0.3
    )
    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels(task_ids, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"RR delta ({label_a} - {label_b})")
    ax.set_title(f"Per-task RR delta (green = {label_a} wins)")
    fig.tight_layout()

    path = output_dir / "rr_delta_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-task retrieval results between two retrievers",
    )
    parser.add_argument("report_a", type=Path, help="First report JSON (treated as A)")
    parser.add_argument("report_b", type=Path, help="Second report JSON (treated as B)")
    parser.add_argument("--k", type=int, default=10, help="k for recall/hit breakdown")
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of top wins/losses"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Directory for plots"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    report_a = _load_report(args.report_a)
    report_b = _load_report(args.report_b)
    label_a = _label(args.report_a)
    label_b = _label(args.report_b)

    comparisons = _build_comparisons(report_a, report_b, args.k)
    if not comparisons:
        print("No overlapping tasks found between the two reports.")
        return

    _print_summary(report_a, report_b, label_a, label_b)
    _print_win_loss(comparisons, label_a, label_b, args.k)
    _print_top_deltas(comparisons, label_a, label_b, args.top_n)

    output_dir = args.output_dir or Path(f"outputs/eda/compare-{label_a}-vs-{label_b}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_rr_scatter(comparisons, label_a, label_b, output_dir)
    _plot_rank_delta_bar(comparisons, label_a, label_b, output_dir)


if __name__ == "__main__":
    main()

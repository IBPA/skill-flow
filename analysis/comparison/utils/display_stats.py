"""CI-enhanced display helpers for condition comparison output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from analysis.stats.benchmark_stats import (
    benchmark_ci,
    benchmark_paired_test,
    significance_stars,
)

if TYPE_CHECKING:
    from analysis.comparison.utils.loader import ConditionResults
    from analysis.stats.types import ConfidenceInterval


def _fmt_ci(ci: ConfidenceInterval, fmt: str = ".1%") -> str:
    """Format a CI as 'mean [lo, hi]'."""
    return f"{ci.mean:{fmt}} [{ci.ci_lo:{fmt}}, {ci.ci_hi:{fmt}}]"


def _print_row(
    label: str,
    values: list[int] | list[float],
    header_w: int,
    col_w: int,
    *,
    fmt: str = "",
) -> None:
    row = f"{label:>{header_w}}"
    for v in values:
        cell = f"{v:{fmt}}"
        row += f"  {cell:>{col_w}}"
    print(row)


def _print_ci_row(
    label: str,
    cis: list[ConfidenceInterval],
    header_w: int,
    col_w: int,
    *,
    fmt: str = ".1%",
) -> None:
    row = f"{label:>{header_w}}"
    for ci in cis:
        cell = _fmt_ci(ci, fmt)
        row += f"  {cell:>{col_w}}"
    print(row)


def print_summary_with_ci(
    conditions: list[ConditionResults],
    threshold: float,
) -> None:
    """Print aggregate summary table with bootstrap 95% CIs."""
    col_w = max((len(c.label) for c in conditions), default=30)
    col_w = max(col_w, 30)
    header_w = 20
    sep = "-" * (header_w + (col_w + 2) * len(conditions))

    print("\n1. Aggregate Summary (with 95% CI)")
    print(sep)
    row_header = f"{'':>{header_w}}"
    for c in conditions:
        row_header += f"  {c.label:>{col_w}}"
    print(row_header)
    print(sep)

    _print_row("Runs", [len(c.runs) for c in conditions], header_w, col_w, fmt="d")

    for k in (1, 2, 3):
        cis = [benchmark_ci(c, "pass_at", k, threshold=threshold) for c in conditions]
        _print_ci_row(f"Pass@{k}", cis, header_w, col_w)
    for k in (1, 2, 3):
        cis = [benchmark_ci(c, "pass_pow", k, threshold=threshold) for c in conditions]
        _print_ci_row(f"Pass^{k}", cis, header_w, col_w)

    cis = [benchmark_ci(c, "mean_reward") for c in conditions]
    _print_ci_row("Mean reward", cis, header_w, col_w, fmt=".3f")
    cis = [benchmark_ci(c, "mean_steps") for c in conditions]
    _print_ci_row("Steps/task", cis, header_w, col_w, fmt=".1f")
    cis = [benchmark_ci(c, "mean_cost") for c in conditions]
    _print_ci_row("Cost $/task", cis, header_w, col_w, fmt=".4f")
    print(sep)


def print_significance(
    baseline: ConditionResults,
    others: list[ConditionResults],
    threshold: float,
) -> None:
    """Print paired bootstrap significance tests vs baseline."""
    print(f"\n4. Significance Tests (vs {baseline.label})")
    for c in others:
        print(f"\n  {baseline.label} → {c.label}:")
        for k in (1, 2, 3):
            result = benchmark_paired_test(
                c, baseline, "pass_at", k, threshold=threshold
            )
            stars = significance_stars(result.p_value)
            print(
                f"    Pass@{k}: diff={result.observed_diff:+.1%}, "
                f"p={result.p_value:.4f} {stars}"
            )

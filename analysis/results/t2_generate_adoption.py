"""Generate the skill adoption statistics table (tab:adoption).

Produces a complete LaTeX ``table`` environment showing skill retrieval and
agent use statistics across SkillsBench and Terminal-Bench conditions.

Usage::

    uv run python -m analysis.results.t2_generate_adoption

    uv run python -m analysis.results.t2_generate_adoption \
        --output paper/tables/2_adoption.tex

    uv run python -m analysis.results.t2_generate_adoption --print-pvalues
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.results.utils.adoption_utils import (
    SK_ADOPTION_GROUPS,
    TB_ADOPTION_GROUPS,
    Row,
    SignificanceResult,
    apply_bold,
    apply_significance,
    compute_row,
    format_row,
)
from analysis.results.utils.latex_utils import write_or_print

# -- Group builder -----------------------------------------------------


def _build_group(
    groups: list[dict[str, object]],
    *,
    is_tb: bool,
    exclude: set[int] | None = None,
    baseline_label: str = "Vercel",
) -> tuple[list[Row], list[SignificanceResult]]:
    """Compute rows for a benchmark group and apply bold + significance.

    Returns the formatted rows and the significance test results.
    """
    rows: list[Row] = []
    for group in groups:
        row = compute_row(group, is_tb=is_tb)
        if row is not None:
            rows.append(row)
    apply_bold(rows, exclude=exclude or set())
    # Count non-baseline, non-oracle rows for Holm-Bonferroni correction
    n_cmp = sum(1 for r in rows if r.label not in {baseline_label, "Oracle"})
    sig_results = apply_significance(
        rows,
        baseline_label=baseline_label,
        col="loaded",
        n_comparisons=max(n_cmp, 1),
    )
    return rows, sig_results


# -- Full table --------------------------------------------------------


def render_table() -> tuple[
    list[str], list[SignificanceResult], list[SignificanceResult]
]:
    """Return LaTeX lines and significance results for both benchmarks."""
    ncols = 5
    lines: list[str] = [
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{rcccc}",
        r"  \toprule",
        (
            r"  \textbf{Condition}"
            r" & \textbf{Mean Skills/Task} & \textbf{Tasks Retrieved (\%)}"
            r" & \textbf{Oracle Skills Retrieved (\%)}"
            r" & \textbf{Tasks Used (\%)} \\"
        ),
        r"  \midrule",
        rf"  \multicolumn{{{ncols}}}{{l}}{{\textit{{SkillsBench}}}} \\",
        r"  \midrule",
    ]

    # SkillsBench group
    sk_rows, sk_sig = _build_group(SK_ADOPTION_GROUPS, is_tb=False, exclude={0})
    oracle_idx = next(
        (i for i, r in enumerate(sk_rows) if r.label == "Oracle"),
        None,
    )
    for i, row in enumerate(sk_rows):
        lines.append(format_row(row, is_oracle=(i == oracle_idx)))
    lines.append(r"  \midrule")
    lines.append(
        rf"  \multicolumn{{{ncols}}}{{l}}{{\textit{{Terminal-Bench}}}} \\",
    )
    lines.append(r"  \midrule")

    # Terminal-Bench group
    tb_rows, tb_sig = _build_group(TB_ADOPTION_GROUPS, is_tb=True)
    for row in tb_rows:
        lines.append(format_row(row))

    lines.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}%",
            r"}",
        ]
    )
    return lines, sk_sig, tb_sig


# -- P-value printing --------------------------------------------------


def _fmt_pvalue(p: float) -> str:
    """Format a p-value in scientific notation, clamping small values."""
    if p < 1e-6:
        return "<1.00e-06"
    return f"{p:.2e}"


def _print_pvalues(
    sk_sig: list[SignificanceResult],
    tb_sig: list[SignificanceResult],
) -> None:
    """Print raw and Holm-Bonferroni adjusted p-values to stdout."""
    print("\n=== Paired Bootstrap Test P-Values (Tasks Used vs Vercel) ===\n")
    for bench_name, results in [
        ("SkillsBench", sk_sig),
        ("Terminal-Bench", tb_sig),
    ]:
        print(f"--- {bench_name} ---")
        if not results:
            print("  (no comparisons available)\n")
            continue
        for r in results:
            raw_str = _fmt_pvalue(r.raw_p)
            adj_str = _fmt_pvalue(r.adjusted_p)
            print(f"  {r.label:30s}  raw p = {raw_str:>12s}  adj p = {adj_str:>12s}")
        print()


# -- CLI ---------------------------------------------------------------


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate tab:adoption skill statistics table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/2_adoption.tex"),
    )
    parser.add_argument(
        "--print-pvalues",
        action="store_true",
        help="Print raw and Holm-Bonferroni adjusted p-values to stdout",
    )
    args = parser.parse_args()
    table_lines, sk_sig, tb_sig = render_table()
    write_or_print(table_lines, args.output)
    if args.print_pvalues:
        _print_pvalues(sk_sig, tb_sig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

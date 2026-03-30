"""Generate the benchmark performance table (tab:results) with bootstrap CIs.

Usage::

    uv run python -m analysis.results.t1_generate_results \
        [--eval-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.comparison.compare_conditions import align_conditions
from analysis.comparison.utils.loader import load_condition
from analysis.results.utils.format_utils import mark_best, mark_best_whole
from analysis.results.utils.latex_utils import fmt_ci_pct, write_or_print
from analysis.stats.benchmark_stats import benchmark_ci, benchmark_paired_test
from analysis.stats.proportions import cohens_h, holm_bonferroni

if TYPE_CHECKING:
    from analysis.comparison.utils.loader import ConditionResults
    from analysis.stats.types import ConfidenceInterval

_SK_ORDER: list[tuple[str, str]] = [
    ("skillsbench-inject", "Oracle"),
    ("baseline", "No Skills"),
    ("vercel-find-skills", "Vercel"),
    ("skillflow-inject", "SkillFlow (Ours)"),
]

_TB_ORDER: list[tuple[str, str]] = [
    ("baseline", "No Skills"),
    ("vercel-find-skills", "Vercel"),
    ("skillflow-inject", "SkillFlow (Ours)"),
    (
        "skillflow-inject-specificity-v3.0-no-letta",
        "SkillFlow-specific (Ours)",
    ),
]

_LABEL_PAD = 16


def _stars_from_adj_p(p_adj: float) -> str:
    """Convert Holm-Bonferroni-adjusted p-value to LaTeX significance marker."""
    if p_adj < 0.01:
        return "$^{**}$"
    if p_adj < 0.05:
        return "$^{*}$"
    return ""


def _fmt_plain_steps(ci: ConfidenceInterval) -> str:
    """Format steps/task as a plain value (no CI)."""
    return f"{ci.mean:.1f}"


def _fmt_plain_cost(ci: ConfidenceInterval) -> str:
    """Format cost/task as a plain dollar value (no CI)."""
    return f"\\${ci.mean:.3f}"


def _italicize_ci_cell(cell: str) -> str:
    """Wrap only the number portion in italic, leaving CI brackets outside."""
    idx = cell.find("{\\scriptsize")
    if idx == -1:
        return f"\\textit{{{cell}}}"
    return f"\\textit{{{cell[:idx]}}}{cell[idx:]}"


def _compute_significance(
    loaded: list[tuple[str, ConditionResults]],
    aligned_dict: dict[str, ConditionResults],
    baseline_label: str,
    baseline_cond: ConditionResults | None,
    bench_name: str,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Compute paired bootstrap significance and p-value records."""
    non_base_labels = [lbl for lbl, _ in loaded if lbl != baseline_label]
    raw_p: list[float] = [
        benchmark_paired_test(
            aligned_dict[lbl],
            baseline_cond,
            "pass_at",
            k=1,
        ).p_value
        if baseline_cond is not None
        else 1.0
        for lbl in non_base_labels
    ]
    adj_p = holm_bonferroni(raw_p)
    sig_map = dict(zip(non_base_labels, adj_p, strict=True))

    base_mean = (
        benchmark_ci(baseline_cond, "pass_at", k=1).mean
        if baseline_cond is not None
        else 0.0
    )
    pvalue_records: list[dict[str, object]] = []
    for i, lbl in enumerate(non_base_labels):
        cond_mean = benchmark_ci(aligned_dict[lbl], "pass_at", k=1).mean
        es = cohens_h(cond_mean, base_mean)
        pvalue_records.append(
            {
                "benchmark": bench_name,
                "condition": lbl,
                "raw_p": raw_p[i],
                "adj_p": adj_p[i],
                "cohens_h": es.cohens_h,
                "interpretation": es.interpretation,
            }
        )
    return sig_map, pvalue_records


def _build_section(
    eval_dir: Path,
    cond_order: list[tuple[str, str]],
    bench_name: str,
    prefix: str,
) -> tuple[list[str] | None, list[dict[str, object]]]:
    """Build lines for one benchmark section.

    Returns:
        Tuple of (LaTeX lines or None, list of p-value records).
    """
    # Load conditions in order
    loaded: list[tuple[str, ConditionResults]] = []
    for cond_name, label in cond_order:
        c = load_condition(eval_dir, cond_name, label, prefix=prefix)
        if c.runs:
            loaded.append((label, c))
    if not loaded:
        return None, []

    conds_list = [c for _, c in loaded]
    aligned = align_conditions(conds_list)
    aligned_dict = {c.label: c for c in aligned}

    baseline_label = "No Skills"
    baseline_cond = aligned_dict.get(baseline_label)
    oracle_label = "Oracle" if "Oracle" in aligned_dict else None

    sig_map, pvalue_records = _compute_significance(
        loaded,
        aligned_dict,
        baseline_label,
        baseline_cond,
        bench_name,
    )

    row_data: list[dict[str, str]] = []
    for label, _cond in loaded:
        cond = aligned_dict[label]
        ci_p1 = benchmark_ci(cond, "pass_at", k=1)
        ci_p3 = benchmark_ci(cond, "pass_at", k=3)
        ci_pp3 = benchmark_ci(cond, "pass_pow", k=3)
        ci_steps = benchmark_ci(cond, "mean_steps")
        ci_cost = benchmark_ci(cond, "mean_cost")

        sig = _stars_from_adj_p(sig_map[label]) if label in sig_map else ""

        row_data.append(
            {
                "label": label,
                "p1": fmt_ci_pct(ci_p1),
                "p3": fmt_ci_pct(ci_p3),
                "pp3": fmt_ci_pct(ci_pp3),
                "steps": _fmt_plain_steps(ci_steps),
                "cost": _fmt_plain_cost(ci_cost),
                "sig": sig,
            }
        )

    oracle_idx = next(
        (i for i, r in enumerate(row_data) if r["label"] == oracle_label),
        None,
    )
    exclude = {oracle_idx} if oracle_idx is not None else set()
    for col in ("p1", "p3", "pp3"):
        cells = [r[col] for r in row_data]
        bolded = mark_best(cells, exclude=exclude, direction="max")
        for i, r in enumerate(row_data):
            r[col] = bolded[i]
    for col in ("steps", "cost"):
        cells = [r[col] for r in row_data]
        bolded = mark_best_whole(cells, exclude=exclude, direction="min")
        for i, r in enumerate(row_data):
            r[col] = bolded[i]

    lines: list[str] = [
        f"  \\multicolumn{{6}}{{l}}{{\\textit{{{bench_name}}}}} \\\\",
        r"  \midrule",
    ]

    for row in row_data:
        lbl, sig = row["label"], row["sig"]
        p1, p3, pp3 = row["p1"], row["p3"], row["pp3"]
        steps, cost = row["steps"], row["cost"]
        if lbl == oracle_label:
            lbl = f"\\textit{{{lbl}}}"
            p1 = _insert_sig(_italicize_ci_cell(p1), sig)
            p3, pp3 = _italicize_ci_cell(p3), _italicize_ci_cell(pp3)
            steps, cost = f"\\textit{{{steps}}}", f"\\textit{{{cost}}}"
            end = " \\\\[2pt]"
        else:
            p1, end = _insert_sig(p1, sig), " \\\\"
        lines.append(
            f"  {lbl:<{_LABEL_PAD}s} & {p1} & {p3} & {pp3} & {steps} & {cost}{end}",
        )

    return lines, pvalue_records


def _insert_sig(cell: str, sig: str) -> str:
    """Insert significance marker after the number but before the CI."""
    if not sig:
        return cell
    idx = cell.find("{\\scriptsize")
    if idx == -1:
        return cell + sig
    return cell[:idx] + sig + cell[idx:]


def render_table(
    eval_dir: Path,
) -> tuple[list[str], list[dict[str, object]]]:
    """Return LaTeX tabular content and collected p-value records."""
    lines: list[str] = [
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{rccccc}",
        r"  \toprule",
        (
            r"  \textbf{Condition} & \textbf{Pass@1} & \textbf{Pass@3}"
            r" & \textbf{Pass\textasciicircum3} & \textbf{Steps/Task}"
            r" & \textbf{Cost/Task} \\"
        ),
        r"  \midrule",
    ]

    all_pvalues: list[dict[str, object]] = []
    for prefix, cond_order, bench_name in [
        ("sk", _SK_ORDER, "SkillsBench"),
        ("tb", _TB_ORDER, "Terminal-Bench"),
    ]:
        section, pv_records = _build_section(eval_dir, cond_order, bench_name, prefix)
        all_pvalues.extend(pv_records)
        if section:
            lines.extend(section)
            lines.append(r"  \midrule")

    # Replace last \midrule with \bottomrule
    if lines and lines[-1].strip() == r"\midrule":
        lines[-1] = r"  \bottomrule"

    lines.extend([r"\end{tabular}%", r"}"])
    return lines, all_pvalues


def _print_pvalues(records: list[dict[str, object]]) -> None:
    """Print raw and adjusted p-values with Cohen's h to stdout."""
    if not records:
        print("No p-value records to display.")
        return
    current_bench = ""
    for rec in records:
        bench = str(rec["benchmark"])
        if bench != current_bench:
            if current_bench:
                print()
            print(f"--- {bench} (vs No Skills baseline) ---")
            current_bench = bench
        raw = float(str(rec["raw_p"]))
        adj = float(str(rec["adj_p"]))
        h = float(str(rec["cohens_h"]))
        interp = rec["interpretation"]
        print(
            f"  {rec['condition']:<30s}  "
            f"raw p={raw:.2e}  adj p={adj:.2e}  "
            f"Cohen's h={h:+.3f} ({interp})"
        )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate tab:results benchmark performance table",
    )
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/1_results.tex"),
    )
    parser.add_argument(
        "--print-pvalues",
        action="store_true",
        help="Print raw and Holm-Bonferroni adjusted p-values to stdout",
    )
    args = parser.parse_args()

    table_lines, pvalue_records = render_table(args.eval_dir)
    if args.print_pvalues:
        _print_pvalues(pvalue_records)
        print()
    write_or_print(table_lines, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

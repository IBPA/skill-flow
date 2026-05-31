"""Generate the selector ablation table (tab:selector_ablation).

Compares the deployed SkillFlow pipeline (with the gpt-4o-mini selector) against a
no-selector variant that injects the deep-reranker top-10 directly, end-to-end on
SkillsBench (Codex GPT-5-mini). Reports Pass@1 and cost/task with bootstrap CIs and
the paired equivalence/cost tests (no-selector vs full SkillFlow).

The ``--print-stats`` output also documents why the no-selector condition is kept
out of the main results-table (tab:results) multiple-comparison family: folding it
in as a fourth vs-baseline test pushes the headline SkillFlow result past the
Holm-Bonferroni 0.05 threshold, so the ablation is reported separately.

Usage::

    uv run python -m analysis.results.t20_generate_selector_ablation \
        [--eval-dir DIR] [--model gpt5mini] [--output PATH] [--print-stats]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.comparison.compare_conditions import align_conditions
from analysis.comparison.utils.loader import load_condition
from analysis.results.utils.latex_utils import fmt_ci_pct, write_or_print
from analysis.stats.benchmark_stats import benchmark_ci, benchmark_paired_test
from analysis.stats.proportions import holm_bonferroni

if TYPE_CHECKING:
    from analysis.comparison.utils.loader import ConditionResults

# All conditions loaded; canonical labels keyed below.
_CONDS: list[tuple[str, str]] = [
    ("baseline", "No Skills"),
    ("skillsbench-inject", "Oracle"),
    ("vercel-find-skills", "Vercel"),
    ("skillflow-inject", "SkillFlow"),
    ("skillflow-deep-reranker-inject", "No-Selector"),
]
# Rows shown in the LaTeX table, with their display names.
_TABLE_DISPLAY: list[tuple[str, str]] = [
    ("No Skills", "No Skills"),
    ("SkillFlow", "SkillFlow (with selector)"),
    ("No-Selector", "Deep-reranker top-10 (no selector)"),
]
# The vs-baseline family used by tab:results (excludes No-Selector).
_MAIN_FAMILY: list[str] = ["Oracle", "Vercel", "SkillFlow"]


def _load_aligned(eval_dir: Path, model: str | None) -> dict[str, ConditionResults]:
    """Load every condition, align to the shared task set, and key by label."""
    loaded = [
        c
        for cond, label in _CONDS
        if (c := load_condition(eval_dir, cond, label, model=model)).runs
    ]
    return {c.label: c for c in align_conditions(loaded)}


def render_table(eval_dir: Path, model: str | None = None) -> list[str]:
    """Return LaTeX tabular content for the selector ablation."""
    aligned = _load_aligned(eval_dir, model)
    lines: list[str] = [
        r"\begin{tabular}{lcc}",
        r"  \toprule",
        r"  \textbf{Pipeline} & \textbf{Pass@1} & \textbf{Cost/Task} \\",
        r"  \midrule",
    ]
    for label, display in _TABLE_DISPLAY:
        cond = aligned.get(label)
        if cond is None:
            continue
        p1 = fmt_ci_pct(benchmark_ci(cond, "pass_at", k=1))
        cost = benchmark_ci(cond, "mean_cost").mean
        lines.append(f"  {display} & {p1} & \\${cost:.3f} \\\\")
    lines.extend([r"  \bottomrule", r"\end{tabular}"])
    return lines


def _print_stats(eval_dir: Path, model: str | None = None) -> None:
    """Print Pass@1/cost, the ablation paired tests, and the Holm-family demo."""
    aligned = _load_aligned(eval_dir, model)
    base = aligned["No Skills"]
    print(f"Aligned tasks: {len(base.task_rewards)}\n")
    for label, _display in _TABLE_DISPLAY:
        if label not in aligned:
            continue
        ci = benchmark_ci(aligned[label], "pass_at", k=1)
        cost = benchmark_ci(aligned[label], "mean_cost").mean
        print(
            f"  {label:12s} Pass@1={ci.mean * 100:5.1f} "
            f"[{ci.ci_lo * 100:.1f}, {ci.ci_hi * 100:.1f}]  cost/task=${cost:.4f}"
        )

    full, nosel = aligned["SkillFlow"], aligned["No-Selector"]
    eq = benchmark_paired_test(nosel, full, "pass_at", k=1)
    cost_test = benchmark_paired_test(nosel, full, "mean_cost")
    print("\nAblation-internal paired tests (no-selector vs full SkillFlow):")
    print(f"  Pass@1 equivalence: diff={eq.observed_diff:+.4f}  p={eq.p_value:.4f}")
    print(f"  cost/task: diff={cost_test.observed_diff:+.4f} p={cost_test.p_value:.4f}")

    if not all(lbl in aligned for lbl in _MAIN_FAMILY):
        return

    def raw_p(label: str) -> float:
        return benchmark_paired_test(aligned[label], base, "pass_at", k=1).p_value

    fam4 = [*_MAIN_FAMILY, "No-Selector"]
    raw_main = [raw_p(x) for x in _MAIN_FAMILY]
    adj3 = dict(zip(_MAIN_FAMILY, holm_bonferroni(raw_main), strict=True))
    adj4 = dict(zip(fam4, holm_bonferroni([raw_p(x) for x in fam4]), strict=True))
    print("\nHolm-Bonferroni over the vs-baseline family (why we keep it separate):")
    print(f"  {'condition':12s} {'raw p':>8s} {'adj m=3':>9s} {'adj m=4':>9s}")
    for label in fam4:
        a3 = f"{adj3[label]:.4f}" if label in adj3 else "--"
        print(f"  {label:12s} {raw_p(label):8.4f} {a3:>9s} {adj4[label]:9.4f}")
    print(
        "  Adding No-Selector pushes SkillFlow past the 0.05 Holm threshold, so the\n"
        "  ablation is reported separately from tab:results."
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate the selector ablation table")
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument(
        "--model",
        type=str,
        default="gpt5mini",
        help="Model substring to filter run directories; empty string disables it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/20_selector_ablation.tex"),
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print Pass@1/cost, equivalence/cost p-values, and the Holm-family demo.",
    )
    args = parser.parse_args()

    model = args.model or None
    if args.print_stats:
        _print_stats(args.eval_dir, model)
        print()
    write_or_print(render_table(args.eval_dir, model), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

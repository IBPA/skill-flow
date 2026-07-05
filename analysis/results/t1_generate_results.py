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

# Second-agent SkillsBench panel (no Vercel/oracle-free benchmark). Used only when
# a second-agent model is supplied, to fold a transferability panel into tab:results.
_SK_SECOND_AGENT_ORDER: list[tuple[str, str]] = [
    ("skillsbench-inject", "Oracle"),
    ("baseline", "No Skills"),
    ("skillflow-inject", "SkillFlow (Ours)"),
]

_LABEL_PAD = 16
_MODEL_PAD = 16


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


def _score_group(
    eval_dir: Path,
    cond_order: list[tuple[str, str]],
    stat_label: str,
    prefix: str,
    model: str | None,
) -> tuple[list[dict[str, str]], str | None, list[dict[str, object]]]:
    """Load one (benchmark, model) group; return bolded rows + p-value records.

    Bolding (best per column, excluding Oracle) and significance (Holm-Bonferroni
    vs the group's own No Skills baseline) are computed within the group, so the
    two models never compete for bold or share a correction family.
    """
    loaded: list[tuple[str, ConditionResults]] = []
    for cond_name, label in cond_order:
        c = load_condition(eval_dir, cond_name, label, prefix=prefix, model=model)
        if c.runs:
            loaded.append((label, c))
    if not loaded:
        return [], None, []

    aligned_dict = {c.label: c for c in align_conditions([c for _, c in loaded])}
    baseline_cond = aligned_dict.get("No Skills")
    oracle_label = "Oracle" if "Oracle" in aligned_dict else None

    sig_map, pvalue_records = _compute_significance(
        loaded, aligned_dict, "No Skills", baseline_cond, stat_label
    )

    row_data: list[dict[str, str]] = []
    for label, _cond in loaded:
        cond = aligned_dict[label]
        row_data.append(
            {
                "label": label,
                "p1": fmt_ci_pct(benchmark_ci(cond, "pass_at", k=1)),
                "p3": fmt_ci_pct(benchmark_ci(cond, "pass_at", k=3)),
                "pp3": fmt_ci_pct(benchmark_ci(cond, "pass_pow", k=3)),
                "steps": _fmt_plain_steps(benchmark_ci(cond, "mean_steps")),
                "cost": _fmt_plain_cost(benchmark_ci(cond, "mean_cost")),
                "sig": _stars_from_adj_p(sig_map[label]) if label in sig_map else "",
            }
        )

    oracle_idx = next(
        (i for i, r in enumerate(row_data) if r["label"] == oracle_label), None
    )
    exclude = {oracle_idx} if oracle_idx is not None else set()
    for col in ("p1", "p3", "pp3"):
        bolded = mark_best([r[col] for r in row_data], exclude=exclude, direction="max")
        for i, r in enumerate(row_data):
            r[col] = bolded[i]
    for col in ("steps", "cost"):
        bolded = mark_best_whole(
            [r[col] for r in row_data], exclude=exclude, direction="min"
        )
        for i, r in enumerate(row_data):
            r[col] = bolded[i]

    return row_data, oracle_label, pvalue_records


def _render_group_rows(
    row_data: list[dict[str, str]],
    oracle_label: str | None,
    model_label: str,
) -> list[str]:
    """Render a model group's rows; the model label appears on the first row."""
    lines: list[str] = []
    for idx, row in enumerate(row_data):
        model_cell = model_label if idx == 0 else ""
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
            f"  {model_cell:<{_MODEL_PAD}s} & {lbl:<{_LABEL_PAD}s} "
            f"& {p1} & {p3} & {pp3} & {steps} & {cost}{end}"
        )
    return lines


def _build_section(
    eval_dir: Path,
    bench_name: str,
    groups: list[tuple[str, str | None, list[tuple[str, str]], str]],
) -> tuple[list[str] | None, list[dict[str, object]]]:
    """Build a benchmark section spanning one or more model groups.

    ``groups`` is a list of ``(model_label, model, cond_order, prefix)``. Each
    model's rows are stacked under a single section header, with the model shown
    once in the first column.
    """
    rendered: list[str] = []
    pvalues: list[dict[str, object]] = []
    for model_label, model, cond_order, prefix in groups:
        stat_label = f"{bench_name} ({model_label})"
        row_data, oracle_label, pv = _score_group(
            eval_dir, cond_order, stat_label, prefix, model
        )
        pvalues.extend(pv)
        if row_data:
            if rendered:  # light divider between model groups in the same section
                rendered.append(r"  \cmidrule(lr){1-7}")
            rendered.extend(_render_group_rows(row_data, oracle_label, model_label))
    if not rendered:
        return None, pvalues

    lines = [
        f"  \\multicolumn{{7}}{{l}}{{\\textit{{{bench_name}}}}} \\\\",
        r"  \midrule",
        *rendered,
    ]
    return lines, pvalues


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
    model: str | None = None,
    model_label: str = "GPT-5-mini",
    second_agent_model: str | None = None,
    second_agent_name: str = "Claude Haiku 4.5",
) -> tuple[list[str], list[dict[str, object]]]:
    """Return LaTeX tabular content and collected p-value records.

    When ``second_agent_model`` is set, its SkillsBench runs are added as a
    second model group inside the SkillsBench section (a ``Model`` column
    distinguishes the agents) to demonstrate transferability across backbones.
    """
    lines: list[str] = [
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"  \toprule",
        (
            r"  \textbf{Model} & \textbf{Condition} & \textbf{Pass@1}"
            r" & \textbf{Pass@3} & \textbf{Pass\textasciicircum3}"
            r" & \textbf{Steps/Task} & \textbf{Cost/Task} \\"
        ),
        r"  \midrule",
    ]

    sk_groups: list[tuple[str, str | None, list[tuple[str, str]], str]] = [
        (model_label, model, _SK_ORDER, "sk"),
    ]
    if second_agent_model:
        sk_groups.append(
            (second_agent_name, second_agent_model, _SK_SECOND_AGENT_ORDER, "sk")
        )

    all_pvalues: list[dict[str, object]] = []
    for bench_name, groups in [
        ("SkillsBench", sk_groups),
        ("Terminal-Bench", [(model_label, model, _TB_ORDER, "tb")]),
    ]:
        section, pv_records = _build_section(eval_dir, bench_name, groups)
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
        "--model",
        type=str,
        default="gpt5mini",
        help=(
            "Model substring for the primary run set (Codex GPT-5-mini); pass an "
            "empty string to disable filtering."
        ),
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default="GPT-5-mini",
        help="Display name for the primary model in the Model column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/1_results.tex"),
    )
    parser.add_argument(
        "--second-agent-model",
        type=str,
        default="claudehaiku4520251001",
        help=(
            "Model substring for a second SkillsBench model group added under the "
            "SkillsBench section. Pass an empty string to omit it; it is also "
            "omitted automatically if no matching runs are found."
        ),
    )
    parser.add_argument(
        "--second-agent-name",
        type=str,
        default="Claude Haiku 4.5",
        help="Display name for the second agent in the Model column.",
    )
    parser.add_argument(
        "--print-pvalues",
        action="store_true",
        help="Print raw and Holm-Bonferroni adjusted p-values to stdout",
    )
    args = parser.parse_args()

    model = args.model or None
    table_lines, pvalue_records = render_table(
        args.eval_dir,
        model,
        model_label=args.model_label,
        second_agent_model=args.second_agent_model,
        second_agent_name=args.second_agent_name,
    )
    if args.print_pvalues:
        _print_pvalues(pvalue_records)
        print()
    write_or_print(table_lines, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

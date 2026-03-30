"""Generate the retrieval pipeline stages table (tab:retrieval_stages) with CIs.

Produces a complete LaTeX ``table`` environment showing recall, precision,
and MRR across each stage of the SkillFlow retrieval pipeline.

Each stage reads from a separate eval report JSON file produced by the
corresponding experiment runner.

Usage::

    uv run python -m analysis.results.t3_generate_retrieval_stages

    uv run python -m analysis.results.t3_generate_retrieval_stages \
        --output paper/tables/3_retrieval_stages.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analysis.results.utils.latex_utils import (
    fmt_ci_abbrev,
    load_report,
    write_or_print,
)
from analysis.stats.retrieval_stats import all_retrieval_cis

# ------------------------------------------------------------------
# Stage definitions — each stage reads from a different eval report
# ------------------------------------------------------------------

_STAGE_REPORTS: dict[str, str] = {
    "Retriever": ("outputs/experiments/retriever-union/bge-base-q5-union-tk200.json"),
    "Shallow Reranker": (
        "outputs/experiments/reranker-comparison/"
        "cross-encoder-ms-marco-minilm-l-6-v2-512chars-q1-max.json"
    ),
    "Deep Reranker": (
        "outputs/experiments/deep-reranker-comparison/"
        "baai-bge-reranker-v2-m3-4096chars-q1-max.json"
    ),
    "Selector": (
        "outputs/pipeline/skillsbench/specificity-v2.0/eval-stage4-selector.json"
    ),
}

_KS_BY_STAGE: dict[str, list[int]] = {
    "Retriever": [1, 5, 10, 50, 100, 500, 1000],
    "Shallow Reranker": [1, 5, 10, 50, 100],
    "Deep Reranker": [1, 5, 10],
    "Selector": [1, 5],
}

_K_OUTPUT_LABELS: dict[str, str] = {
    "Retriever": "1000",
    "Shallow Reranker": "100",
    "Deep Reranker": "10",
    "Selector": r"$\leq$5",
}

_ALL_KS = [1, 5, 10, 50, 100, 500, 1000]

# Padding widths for alignment (measured from authoritative table)
_STAGE_PAD = 33
_K_PAD = 24


# ------------------------------------------------------------------
# Row generation
# ------------------------------------------------------------------


def _render_stage_rows(
    report_paths: dict[str, Path],
    ks_by_stage: dict[str, list[int]],
    lines: list[str],
) -> None:
    """Append retrieval stage data rows (R + P per stage, MRR in last column)."""
    first_stage = True
    for stage_name, report_path in report_paths.items():
        report = load_report(report_path)
        if report is None:
            lines.append(
                f"% SKIPPED {stage_name}: report not found at {report_path}",
            )
            continue

        if not first_stage:
            lines.append(r"  \midrule")
        first_stage = False

        ks = ks_by_stage.get(stage_name, [1, 5, 10])
        cis = all_retrieval_cis(report, ks)
        k_label = _K_OUTPUT_LABELS.get(stage_name, "---")
        mrr_ci = cis.get("MRR")
        mrr_str = fmt_ci_abbrev(mrr_ci) if mrr_ci else "---"

        for row_idx, metric_prefix in enumerate(("R", "P")):
            cells: list[str] = []
            for k in _ALL_KS:
                key = f"{metric_prefix}@{k}"
                if key in cis:
                    cells.append(fmt_ci_abbrev(cis[key]))
                else:
                    cells.append("---")

            if row_idx == 0:
                stage_cell = f"\\multirow{{2}}{{*}}{{{stage_name}}}"
                k_cell = f"\\multirow{{2}}{{*}}{{{k_label}}}"
                mrr_cell = f"\\multirow{{2}}{{*}}{{{mrr_str}}}"
            else:
                stage_cell = ""
                k_cell = ""
                mrr_cell = ""

            padded_cells = [f"{c:<28s}" if c != "---" else "---  " for c in cells]
            cell_str = " & ".join(padded_cells)
            lines.append(
                f"  {stage_cell:<{_STAGE_PAD}s}"
                f" & {k_cell:<{_K_PAD}s}"
                f" & {metric_prefix:<3s}"
                f" & {cell_str}"
                f" & {mrr_cell} \\\\",
            )


# ------------------------------------------------------------------
# Caption
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Full table
# ------------------------------------------------------------------


def render_table() -> list[str]:
    """Return LaTeX tabular content for tab:retrieval_stages (no table wrapper)."""
    lines: list[str] = [
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ccccccccccc}",
        r"  \toprule",
        (
            r"  \textbf{Stage} & \textbf{K output} & \textbf{Metric}"
            r" & \textbf{@1} & \textbf{@5} & \textbf{@10} & \textbf{@50}"
            r" & \textbf{@100} & \textbf{@500} & \textbf{@1000}"
            r" & \textbf{MRR} \\"
        ),
        r"  \midrule",
    ]

    report_paths = {stage: Path(fpath) for stage, fpath in _STAGE_REPORTS.items()}
    _render_stage_rows(report_paths, _KS_BY_STAGE, lines)

    lines.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}%",
            r"}",
        ]
    )
    return lines


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate tab:retrieval_stages pipeline table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/3_retrieval_stages.tex"),
    )
    args = parser.parse_args()

    table_lines = render_table()
    write_or_print(table_lines, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

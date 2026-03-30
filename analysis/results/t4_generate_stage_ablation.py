"""Generate the stage-ablation retrieval metrics table (tab:stage_ablation).

Produces a complete LaTeX ``table`` environment comparing retrieval methods
and cumulative pipeline stages, including a hybrid (Dense + BM25) RRF
fusion baseline.

Usage::

    uv run python -m analysis.results.t4_generate_stage_ablation

    uv run python -m analysis.results.t4_generate_stage_ablation \
        --output paper/tables/4_stage_ablation.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.results.utils.format_utils import mark_best
from analysis.results.utils.fusion_utils import fuse_reports_rrf
from analysis.results.utils.latex_utils import (
    fmt_ci_abbrev,
    load_report,
    write_or_print,
)
from analysis.stats.retrieval_stats import all_retrieval_cis, retrieval_ci

if TYPE_CHECKING:
    from skill_flow.eval.models import EvalReport

    from analysis.stats.types import ConfidenceInterval

# ------------------------------------------------------------------
# Report paths and constants
# ------------------------------------------------------------------

_BM25_REPORT = "outputs/experiments/retriever-ablation/bm25-bge-base.json"
_DENSE_REPORT = "outputs/experiments/retriever-ablation/bge-base.json"

_HYBRID_SENTINEL = "__hybrid__"

# First group: retrieval methods; second group: cumulative pipeline.
# A \midrule is inserted between the two groups.
_RETRIEVAL_METHODS: dict[str, str] = {
    "BM25 only": _BM25_REPORT,
    "Hybrid (Dense $+$ BM25)": _HYBRID_SENTINEL,
    "Dense only": _DENSE_REPORT,
}

_PIPELINE_STAGES: dict[str, str] = {
    r"$+$ Shallow reranker (1--2)": (
        "outputs/experiments/reranker-comparison/"
        "cross-encoder-ms-marco-minilm-l-6-v2-512chars-q1-max.json"
    ),
    r"$+$ Deep reranker (1--3)": (
        "outputs/experiments/deep-reranker-comparison/"
        "baai-bge-reranker-v2-m3-4096chars-q1-max.json"
    ),
}

_LABEL_PAD = 32


# ------------------------------------------------------------------
# Row generation
# ------------------------------------------------------------------


def _compute_ablation_cells(
    report: EvalReport,
) -> list[str]:
    """Return [MRR, R@10, P@10, Hit@10] formatted with abbreviated CIs."""
    cis = all_retrieval_cis(report, [10])
    hit_ci: ConfidenceInterval = retrieval_ci(report, "hit", 10)
    return [
        fmt_ci_abbrev(cis["MRR"]) if "MRR" in cis else "---",
        fmt_ci_abbrev(cis["R@10"]) if "R@10" in cis else "---",
        fmt_ci_abbrev(cis["P@10"]) if "P@10" in cis else "---",
        fmt_ci_abbrev(hit_ci),
    ]


def _build_rows(
    computed_reports: dict[str, EvalReport],
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    """Build (retrieval_rows, pipeline_rows) with label and cells."""
    method_rows: list[tuple[str, list[str]]] = []
    for label, report_path in _RETRIEVAL_METHODS.items():
        if str(report_path) == _HYBRID_SENTINEL:
            if label in computed_reports:
                method_rows.append(
                    (
                        label,
                        _compute_ablation_cells(
                            computed_reports[label],
                        ),
                    )
                )
            continue
        report = load_report(Path(report_path))
        if report is not None:
            method_rows.append((label, _compute_ablation_cells(report)))

    pipeline_rows: list[tuple[str, list[str]]] = []
    for label, report_path in _PIPELINE_STAGES.items():
        report = load_report(Path(report_path))
        if report is not None:
            pipeline_rows.append((label, _compute_ablation_cells(report)))

    return method_rows, pipeline_rows


# ------------------------------------------------------------------
# Full table
# ------------------------------------------------------------------


def render_table() -> list[str]:
    """Return LaTeX tabular content for tab:stage_ablation (no table wrapper)."""
    # Build hybrid (Dense + BM25) via RRF fusion
    computed: dict[str, EvalReport] = {}
    dense_report = load_report(Path(_DENSE_REPORT))
    bm25_report = load_report(Path(_BM25_REPORT))
    if dense_report and bm25_report:
        computed["Hybrid (Dense $+$ BM25)"] = fuse_reports_rrf(
            dense_report,
            bm25_report,
        )

    method_rows, pipeline_rows = _build_rows(computed)
    all_rows = method_rows + pipeline_rows

    # Apply bold to best value in each metric column (across all rows)
    for col_idx in range(4):
        cells = [row_cells[col_idx] for _, row_cells in all_rows]
        bolded = mark_best(cells, direction="max")
        for i, (_lbl, row_cells) in enumerate(all_rows):
            row_cells[col_idx] = bolded[i]

    lines: list[str] = [
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"  \toprule",
        (
            r"  \textbf{Pipeline} & \textbf{MRR} & \textbf{R@10}"
            r" & \textbf{P@10} & \textbf{Hit@10} \\"
        ),
        r"  \midrule",
    ]

    for label, cells in method_rows:
        cell_str = " & ".join(cells)
        lines.append(f"  {label:<{_LABEL_PAD}s} & {cell_str} \\\\")

    lines.append(r"  \midrule")

    for label, cells in pipeline_rows:
        cell_str = " & ".join(cells)
        lines.append(f"  {label:<{_LABEL_PAD}s} & {cell_str} \\\\")

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
        description="Generate tab:stage_ablation pipeline ablation table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/4_stage_ablation.tex"),
    )
    args = parser.parse_args()

    table_lines = render_table()
    write_or_print(table_lines, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

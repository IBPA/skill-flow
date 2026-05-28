"""Generate the structure-aware retrieval comparison table.

Reads all 11 variant reports from
``outputs/experiments/structure-aware/*.json`` (produced by
``skill_flow.config.experiments.structure-aware.json``) and emits a markdown
or LaTeX comparison table. Variants:

* description-only baseline (bge-base)
* C2: description + bge-code-v1
* C1: full content + bge-code-v1 (max 2048 tokens)
* Strategy A: 3-way YAML/prose/code, all bge-base, x {rrf, max, mean, sum_norm}
* Strategy B: 3-way YAML/prose with bge-base, code with bge-code-v1,
  x {rrf, max, mean, sum_norm}

Usage::

    uv run python -m analysis.results.t17_generate_structure_aware_comparison

    # LaTeX appendix table to a file
    uv run python -m analysis.results.t17_generate_structure_aware_comparison \\
        --format latex --output paper/tables/t17_structure_aware.tex
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.results.utils.format_utils import mark_best
from analysis.results.utils.latex_utils import (
    available_ks,
    load_reports_from_dir,
    table_env,
    write_or_print,
)

if TYPE_CHECKING:
    from skill_flow.eval.models import EvalReport

logger = logging.getLogger(__name__)

_KS = [5, 10, 100, 1000]

# Display order and labels — keyed by JSON stem (which equals the slugified
# variant ``label`` field from skill_flow/config/experiments/structure-aware.json).
_ROW_ORDER: list[tuple[str, str]] = [
    ("desc-bge-base", "desc-only (bge-base, baseline)"),
    ("c2-desc-bge-code", "C2: desc + bge-code-v1"),
    ("c1-content-bge-code", "C1: content + bge-code-v1"),
    ("strategy-a-3way-bge-base-rrf", "A: 3-way (bge-base) · RRF"),
    ("strategy-a-3way-bge-base-max", "A: 3-way (bge-base) · max"),
    ("strategy-a-3way-bge-base-mean", "A: 3-way (bge-base) · mean"),
    ("strategy-a-3way-bge-base-sumnorm", "A: 3-way (bge-base) · sum-norm"),
    ("strategy-b-3way-mixed-rrf", "B: 3-way (mixed) · RRF"),
    ("strategy-b-3way-mixed-max", "B: 3-way (mixed) · max"),
    ("strategy-b-3way-mixed-mean", "B: 3-way (mixed) · mean"),
    ("strategy-b-3way-mixed-sumnorm", "B: 3-way (mixed) · sum-norm"),
]


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _row_metrics(report: EvalReport, ks: list[int]) -> dict[str, str]:
    s = report.summary
    cells: dict[str, str] = {"mrr": _fmt(s.mrr)}
    for k in ks:
        cells[f"r@{k}"] = _fmt(s.mean_recall_at.get(k, 0.0))
    return cells


def _ordered_rows(
    reports: dict[str, EvalReport],
    ks: list[int],
) -> tuple[list[str], list[dict[str, str]]]:
    labels: list[str] = []
    rows: list[dict[str, str]] = []
    for stem, label in _ROW_ORDER:
        if stem not in reports:
            logger.warning("missing report: %s.json", stem)
            continue
        labels.append(label)
        rows.append(_row_metrics(reports[stem], ks))
    return labels, rows


def render_markdown(
    reports: dict[str, EvalReport],
    ks: list[int],
) -> list[str]:
    labels, rows = _ordered_rows(reports, ks)
    if not rows:
        return ["(no reports found)"]

    columns = ["mrr", *[f"r@{k}" for k in ks]]
    column_cells = {col: [r[col] for r in rows] for col in columns}
    for col in columns:
        column_cells[col] = mark_best(column_cells[col], fmt="bold")
    # Rewrite ``\textbf{...}`` to markdown ``**...**`` for readability.
    for col, cells in column_cells.items():
        column_cells[col] = [
            c.replace("\\textbf{", "**").replace("}", "**") if "\\textbf" in c else c
            for c in cells
        ]

    header_cells = ["MRR"] + [f"R@{k}" for k in ks]
    lines = [
        "| Variant | " + " | ".join(header_cells) + " |",
        "| --- | " + " | ".join(["---"] * len(header_cells)) + " |",
    ]
    for i, label in enumerate(labels):
        row_cells = [column_cells[col][i] for col in columns]
        lines.append(f"| {label} | " + " | ".join(row_cells) + " |")
    return lines


def render_latex(
    reports: dict[str, EvalReport],
    ks: list[int],
) -> list[str]:
    labels, rows = _ordered_rows(reports, ks)
    if not rows:
        return ["% (no reports found)"]

    columns = ["mrr", *[f"r@{k}" for k in ks]]
    column_cells = {col: [r[col] for r in rows] for col in columns}
    for col in columns:
        column_cells[col] = mark_best(column_cells[col], fmt="bold")

    header_cells = ["\\textbf{MRR}"] + [f"\\textbf{{R@{k}}}" for k in ks]
    header = "\\textbf{Variant} & " + " & ".join(header_cells)
    ncols = header.count("&") + 1
    top, bot = table_env("l" + "c" * (ncols - 1), header)

    body: list[str] = []
    for i, label in enumerate(labels):
        cells = [column_cells[col][i] for col in columns]
        body.append("  " + label + " & " + " & ".join(cells) + " \\\\")
    return [*top, *body, *bot]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("outputs/experiments/structure-aware/"),
    )
    ap.add_argument("--format", choices=("markdown", "latex"), default="markdown")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    reports = load_reports_from_dir(args.reports_dir)
    if not reports:
        logger.error("no reports found in %s", args.reports_dir)
        return 1

    ks = available_ks(reports, _KS)
    if args.format == "markdown":
        lines = render_markdown(reports, ks)
    else:
        lines = render_latex(reports, ks)
    write_or_print(lines, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

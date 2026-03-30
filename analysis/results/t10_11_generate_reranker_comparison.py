"""Generate reranker comparison tables (T10, T11) for the paper appendix.

Usage::

    uv run python -m analysis.results.t10_11_generate_reranker_comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.results.utils.format_utils import mark_best
from analysis.results.utils.latency_utils import load_latency_map
from analysis.results.utils.latex_utils import (
    load_reports_from_dir,
    table_env,
    write_or_print,
)

if TYPE_CHECKING:
    from skill_flow.eval.models import EvalReport

_RERANKER_KS = [1, 5, 10, 50, 100]
_DEEP_KS = [1, 5, 10]

_MODEL_DISPLAY: dict[str, str] = {
    "bge-reranker-v2-m3": "BGE-reranker-v2-m3",
    "ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM",
}


_AGG_ORDER = {"---": 0, "max": 1, "mean": 2, "rrf": 3}

_DEFAULT_LATENCY_RERANKER_DIR = Path(
    "outputs/experiments/latency/latency-reranker-comparison",
)
_DEFAULT_LATENCY_DEEP_DIR = Path(
    "outputs/experiments/latency/latency-deep-reranker-comparison",
)


def _reranker_parts(report: EvalReport) -> tuple[str, str, int, str]:
    """Return (model_short, max_content_chars, num_queries, aggregation)."""
    cfg = report.config
    reranker = cfg.get("reranker") if isinstance(cfg, dict) else None
    if not isinstance(reranker, dict):
        return ("unknown", "?", 1, "---")
    model_short = str(reranker.get("model_name", "unknown")).rsplit("/", 1)[-1]
    chars = str(reranker.get("max_content_chars", "?"))
    qg = reranker.get("query_gen")
    if isinstance(qg, dict) and qg.get("enabled"):
        nq = int(qg.get("num_queries", 1))
        agg = str(qg.get("aggregation", "---")) if nq > 1 else "---"
    else:
        nq, agg = 1, "---"
    return model_short, chars, nq, agg


def _collect_cells(
    report: EvalReport,
    ks: list[int],
    ms: float,
) -> list[str]:
    """Build [MRR, R@k1, ..., R@kN, ms/task] cell strings."""
    cells = [f"{report.summary.mrr:.3f}"]
    for k in ks:
        cells.append(f"{report.summary.mean_recall_at.get(k, 0.0):.3f}")
    cells.append(f"{ms:.1f}" if ms > 0 else "---")
    return cells


def _bold_columns(
    rows: list[tuple[str, str, int, list[str]]],
    n_cols: int,
) -> None:
    """In-place bold the best value per column (max for metrics, min for ms)."""
    for col in range(n_cols):
        col_cells = [r[3][col] for r in rows]
        direction = "min" if col == n_cols - 1 else "max"
        bolded = mark_best(col_cells, direction=direction)
        for i, cell in enumerate(bolded):
            rows[i][3][col] = cell


# ------------------------------------------------------------------
# Table 10: Reranker model comparison
# ------------------------------------------------------------------


def generate_reranker_comparison(
    reports: dict[str, EvalReport],
    latency_map: dict[str, float],
) -> list[str]:
    """Render complete tab:reranker_comparison table."""
    ks = _RERANKER_KS
    k_hdr = " & ".join(f"\\textbf{{R@{k}}}" for k in ks)
    header = (
        f"\\textbf{{Model}} & \\textbf{{Agg.}} & \\textbf{{$M$}}"
        f" & \\textbf{{MRR}} & {k_hdr} & \\textbf{{ms/task}}"
    )
    ncols = header.count("&") + 1
    top, bot = table_env("c" * ncols, header)

    # Collect rows: (model_short, agg, nq, [MRR, R@ks..., ms])
    rows: list[tuple[str, str, int, list[str]]] = []
    for stem, report in sorted(reports.items()):
        model, _chars, nq, agg = _reranker_parts(report)
        ms = latency_map.get(stem, 0.0)
        rows.append((model, agg, nq, _collect_cells(report, ks, ms)))

    # Sort by model name, then (agg_order, num_queries)
    rows.sort(key=lambda r: (r[0], _AGG_ORDER.get(r[1], 99), r[2]))
    _bold_columns(rows, 1 + len(ks) + 1)

    # Identify ms-marco rows and the M=1 row to underline
    ms_idx = [i for i, r in enumerate(rows) if "marco" in r[0].lower()]
    ul_idx = next((i for i in ms_idx if rows[i][2] == 1), None)
    if ul_idx is not None:
        rows[ul_idx][3][:] = [f"\\underline{{{c}}}" for c in rows[ul_idx][3]]

    # Render
    lines = list(top)
    first_ms = True
    prev_display: str | None = None
    for row_idx, (model, agg, nq, cells) in enumerate(rows):
        display = _MODEL_DISPLAY.get(model, model)
        if prev_display is not None and display != prev_display:
            lines.append(r"    \midrule")
        # Underline agg/M for selected row
        agg_s = f"\\underline{{{agg}}}" if row_idx == ul_idx else agg
        nq_s = f"\\underline{{{nq}}}" if row_idx == ul_idx else str(nq)
        # Multirow for ms-marco group
        if row_idx in ms_idx and first_ms:
            model_col = f"\\multirow{{{len(ms_idx)}}}{{*}}{{{display}}}"
            first_ms = False
        elif row_idx in ms_idx:
            model_col = ""
        else:
            model_col = display
        lines.append(f"    {model_col} & {agg_s}  & {nq_s} & {' & '.join(cells)} \\\\")
        prev_display = display

    lines.extend(bot)
    return lines


# ------------------------------------------------------------------
# Table 11: Deep reranker configuration comparison
# ------------------------------------------------------------------


def generate_deep_reranker_config(
    reports: dict[str, EvalReport],
    latency_map: dict[str, float],
) -> list[str]:
    """Render complete tab:deep_reranker_config table."""
    ks = _DEEP_KS
    k_hdr = " & ".join(f"\\textbf{{R@{k}}}" for k in ks)
    header = (
        f"\\textbf{{Chunk Size}} & \\textbf{{Agg.}}"
        f" & \\textbf{{$M$}} & \\textbf{{MRR}}"
        f" & {k_hdr} & \\textbf{{ms/task}}"
    )
    ncols = header.count("&") + 1
    top, bot = table_env(
        "c" * ncols,
        header,
        extra_preamble=r"\renewcommand{\arraystretch}{0.82}%",
    )

    # Collect rows: (chars, agg, nq, [MRR, R@ks..., ms])
    rows: list[tuple[str, str, int, list[str]]] = []
    for stem, report in sorted(reports.items()):
        _model, chars, nq, agg = _reranker_parts(report)
        ms = latency_map.get(stem, 0.0)
        rows.append((chars, agg, nq, _collect_cells(report, ks, ms)))

    rows.sort(key=lambda r: (int(r[0]), _AGG_ORDER.get(r[1], 99), r[2]))
    _bold_columns(rows, 1 + len(ks) + 1)

    # Render
    lines = list(top)
    prev_chars: str | None = None
    for chars, agg, nq, cells in rows:
        if prev_chars is not None and chars != prev_chars:
            lines.append(r"    \addlinespace[2pt]")
        prev_chars = chars
        lines.append(f"    {chars} & {agg} & {nq} & {' & '.join(cells)} \\\\")

    lines.extend(bot)
    return lines


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

_DEFAULT_RERANKER_DIR = Path("outputs/experiments/reranker-comparison")
_DEFAULT_DEEP_DIR = Path("outputs/experiments/deep-reranker-comparison")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate reranker comparison tables",
    )
    parser.add_argument(
        "--reranker-dir",
        type=Path,
        default=_DEFAULT_RERANKER_DIR,
    )
    parser.add_argument(
        "--deep-dir",
        type=Path,
        default=_DEFAULT_DEEP_DIR,
    )
    parser.add_argument(
        "--latency-reranker-dir",
        type=Path,
        default=_DEFAULT_LATENCY_RERANKER_DIR,
    )
    parser.add_argument(
        "--latency-deep-dir",
        type=Path,
        default=_DEFAULT_LATENCY_DEEP_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/tables"),
    )
    args = parser.parse_args()
    out: Path = args.output_dir

    reports = load_reports_from_dir(args.reranker_dir)
    if reports:
        t10_latency = load_latency_map(args.latency_reranker_dir)
        write_or_print(
            generate_reranker_comparison(reports, t10_latency),
            out / "10_reranker_comparison.tex",
        )

    reports = load_reports_from_dir(args.deep_dir)
    if reports:
        t11_latency = load_latency_map(args.latency_deep_dir)
        write_or_print(
            generate_deep_reranker_config(reports, t11_latency),
            out / "11_deep_reranker_config.tex",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

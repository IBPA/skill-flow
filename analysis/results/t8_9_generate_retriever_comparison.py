"""Generate retriever comparison tables (T8, T9) for the paper appendix.

Usage::

    uv run python -m analysis.results.t8_9_generate_retriever_comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.results.utils.format_utils import mark_best
from analysis.results.utils.latency_utils import load_latency_map
from analysis.results.utils.latex_utils import (
    available_ks,
    load_reports_from_dir,
    table_env,
    write_or_print,
)

if TYPE_CHECKING:
    from skill_flow.eval.models import EvalReport

_ALL_KS = [1, 5, 10, 50, 100, 500, 1000]

_MODEL_PARAMS: dict[str, str] = {
    "BAAI/bge-base-en-v1.5": "110M",
    "BAAI/bge-m3": "568M",
    "intfloat/e5-base-v2": "110M",
    "bm25": "---",
}

_MODEL_SHORT: dict[str, str] = {
    "BAAI/bge-base-en-v1.5": "bge-base",
    "BAAI/bge-m3": "bge-m3",
    "intfloat/e5-base-v2": "e5-base",
    "bm25": "bm25",
}

_DEFAULT_LATENCY_RETRIEVER_DIR = Path(
    "outputs/experiments/latency/latency-retriever-comparison",
)
_DEFAULT_LATENCY_QUERYGEN_DIRS = [
    Path("outputs/experiments/latency/latency-retriever-union"),
    Path("outputs/experiments/latency/latency-retriever-querygen"),
]


_T8_ORDER = [
    "bge-base",
    "bge-m3",
    "e5-base",
    "bge-m3 + content",
    "bge-base + content",
    "bm25",
]


def _retriever_label(report: EvalReport) -> tuple[str, str]:
    """Extract short model label and param count from a report config dict."""
    cfg = report.config
    retriever = cfg.get("retriever") if isinstance(cfg, dict) else None
    if not isinstance(retriever, dict):
        return "unknown", "?"
    model = retriever.get("model_name", "unknown")
    short = _MODEL_SHORT.get(model, model.rsplit("/", 1)[-1])
    # Detect content mode from index_dir
    index_dir = cfg.get("index_dir", "") if isinstance(cfg, dict) else ""
    if isinstance(index_dir, str) and "content" in index_dir:
        short = f"{short} + content"
    return short, _MODEL_PARAMS.get(model, "?")


def _fmt(v: float) -> str:
    """Format a metric value as 0.xxx."""
    return f"{v:.3f}"


# ------------------------------------------------------------------
# Table generators
# ------------------------------------------------------------------


def generate_retriever_comparison(
    reports: dict[str, EvalReport],
    latency_map: dict[str, float],
) -> list[str]:
    """Render complete tab:retriever_comparison table."""
    ks = available_ks(reports, _ALL_KS)
    k_hdr = " & ".join(f"\\textbf{{@{k}}}" for k in ks)
    header = (
        f"\\textbf{{Retriever}} & \\textbf{{Params}} & \\textbf{{Metric}}"
        f" & {k_hdr} & \\textbf{{ms/query}}"
    )
    ncols = header.count("&") + 1
    top, bot = table_env("c" * ncols, header)

    labelled: list[tuple[str, str, str, EvalReport]] = []
    for stem, report in reports.items():
        label, params = _retriever_label(report)
        labelled.append((label, params, stem, report))

    order = {name: i for i, name in enumerate(_T8_ORDER)}
    labelled.sort(key=lambda x: order.get(x[0], 999))

    n = len(labelled)
    r_cols: dict[int, list[str]] = {k: [] for k in ks}
    p_cols: dict[int, list[str]] = {k: [] for k in ks}
    mrr_vals: list[str] = []
    ms_vals: list[str] = []

    for _label, _params, stem, report in labelled:
        s = report.summary
        for k in ks:
            r_cols[k].append(_fmt(s.mean_recall_at.get(k, 0.0)))
            p_cols[k].append(_fmt(s.mean_precision_at.get(k, 0.0)))
        mrr_vals.append(_fmt(s.mrr))
        ms = latency_map.get(stem, 0.0)
        ms_vals.append(f"{ms:.1f}" if ms > 0 else "---")

    for k in ks:
        r_cols[k] = mark_best(r_cols[k])
        p_cols[k] = mark_best(p_cols[k])
    mrr_vals = mark_best(mrr_vals)
    ms_vals = mark_best(ms_vals, direction="min")

    lines = list(top)
    for idx, (label, params, _stem, _report) in enumerate(labelled):
        r_str = " & ".join(r_cols[k][idx] for k in ks)
        p_str = " & ".join(p_cols[k][idx] for k in ks)
        ms_s = ms_vals[idx]
        mrr_s = mrr_vals[idx]
        pad = " " * max(1, 37 - len(label))
        lines.append(
            f"    \\multirow{{3}}{{*}}{{{label}}}"
            f"{pad}& \\multirow{{3}}{{*}}{{{params}}} & R"
            f"   & {r_str} & \\multirow{{3}}{{*}}{{{ms_s}}} \\\\"
        )
        blank = " " * 37
        lines.append(
            f"    {blank} & {' ' * max(1, len(params) + 22)} & P   & {p_str} & \\\\"
        )
        lines.append(
            f"    {blank} & {' ' * max(1, len(params) + 22)} & MRR"
            f" & \\multicolumn{{{len(ks)}}}{{c}}{{{mrr_s}}} & \\\\"
        )
        if idx < n - 1:
            lines.append(r"    \midrule")
    lines.extend(bot)
    return lines


def generate_retriever_query_config(
    reports: dict[str, EvalReport],
    latency_map: dict[str, float],
) -> list[str]:
    """Render complete tab:retriever_query_config table."""
    ks = available_ks(reports, _ALL_KS)
    k_hdr = " & ".join(f"\\textbf{{R@{k}}}" for k in ks)
    header = (
        f"\\textbf{{Agg.}} & \\textbf{{$M$}} & \\textbf{{$tk$}}"
        f" & {k_hdr} & \\textbf{{MRR}} & \\textbf{{ms/task}}"
    )
    ncols = header.count("&") + 1
    top, bot = table_env("c" * ncols, header)

    rows: list[tuple[str, int, str, str, EvalReport]] = []
    for stem, report in reports.items():
        cfg = report.config
        retriever = cfg.get("retriever", {}) if isinstance(cfg, dict) else {}
        qg = retriever.get("query_gen") if isinstance(retriever, dict) else None
        if isinstance(qg, dict) and qg.get("enabled"):
            nq = qg.get("num_queries", 1)
            agg = str(qg.get("aggregation", "---")) if nq > 1 else "---"
            tkpq = qg.get("top_k_per_query", None)
            # Skip grid-search variants (top_k_per_query is a list)
            if isinstance(tkpq, list):
                continue
            tk_s = str(tkpq) if tkpq else "---"
        else:
            agg, nq, tk_s = "---", 1, "---"
        rows.append((agg, nq, tk_s, stem, report))

    _agg_order = {"---": 0, "rrf": 1, "union": 2}

    def _sort_key(
        row: tuple[str, int, str, str, EvalReport],
    ) -> tuple[int, int, int]:
        agg, m, tk_s, _, _ = row
        tk_n = int(tk_s) if tk_s != "---" else 0
        return (_agg_order.get(agg, 99), m, tk_n)

    rows.sort(key=_sort_key)

    r_cols: dict[int, list[str]] = {k: [] for k in ks}
    mrr_vals: list[str] = []
    ms_vals: list[str] = []

    for _agg, _nq, _tk_s, stem, report in rows:
        s = report.summary
        for k in ks:
            r_cols[k].append(_fmt(s.mean_recall_at.get(k, 0.0)))
        mrr_vals.append(_fmt(s.mrr))
        ms = latency_map.get(stem, 0.0)
        ms_vals.append(f"{ms:.1f}" if ms > 0 else "---")

    for k in ks:
        r_cols[k] = mark_best(r_cols[k])
    mrr_vals = mark_best(mrr_vals)
    ms_vals = mark_best(ms_vals, direction="min")

    lines = list(top)
    for idx, (agg, nq, tk_s, _stem, _report) in enumerate(rows):
        r_str = " & ".join(r_cols[k][idx] for k in ks)
        ms_s = ms_vals[idx]
        mrr_s = mrr_vals[idx]
        lines.append(
            f"    {agg:<5} & {nq} & {tk_s:<4} & {r_str} & {mrr_s} & {ms_s} \\\\"
        )
    lines.extend(bot)
    return lines


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

_DEFAULT_RETRIEVER_DIR = Path("outputs/experiments/retriever-comparison")
_DEFAULT_QUERYGEN_DIRS = [
    Path("outputs/experiments/retriever-union"),
    Path("outputs/experiments/retriever-querygen-v2"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate retriever comparison tables",
    )
    parser.add_argument(
        "--retriever-dir",
        type=Path,
        default=_DEFAULT_RETRIEVER_DIR,
    )
    parser.add_argument(
        "--querygen-dir",
        type=Path,
        nargs="+",
        default=_DEFAULT_QUERYGEN_DIRS,
    )
    parser.add_argument(
        "--latency-retriever-dir",
        type=Path,
        default=_DEFAULT_LATENCY_RETRIEVER_DIR,
    )
    parser.add_argument(
        "--latency-querygen-dir",
        type=Path,
        nargs="+",
        default=_DEFAULT_LATENCY_QUERYGEN_DIRS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/tables"))
    args = parser.parse_args()
    out: Path = args.output_dir

    reports = load_reports_from_dir(args.retriever_dir)
    if reports:
        t8_latency = load_latency_map(args.latency_retriever_dir)
        write_or_print(
            generate_retriever_comparison(reports, t8_latency),
            out / "8_retriever_comparison.tex",
        )

    qg_reports: dict[str, EvalReport] = {}
    for d in args.querygen_dir:
        qg_reports.update(load_reports_from_dir(d))
    if qg_reports:
        t9_latency = load_latency_map(*args.latency_querygen_dir)
        write_or_print(
            generate_retriever_query_config(qg_reports, t9_latency),
            out / "9_retriever_query_config.tex",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate a LaTeX latency table from pipeline stage timing data.

Reads ``latency_summary.json`` (or falls back to individual stage JSON files)
from a pipeline run directory and outputs a LaTeX table with per-stage
median, mean, and p95 latency in seconds.

Usage::

    uv run python -m analysis.results.t5_generate_latency \
        --run-dir outputs/pipeline/skillsbench/run-dir

    uv run python -m analysis.results.t5_generate_latency \
        --run-dir outputs/pipeline/skillsbench/run-dir \
        --output paper/tables/5_latency.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from skill_flow.pipeline.models import _compute_latency_stats

from analysis.results.utils.latex_utils import write_or_print

_DEFAULT_RUN_DIR = Path(
    "outputs/pipeline/skillsbench/latency-run",
)

_STAGE_LABELS: dict[str, str] = {
    "retriever": "Stage 1: Dense Retriever",
    "reranker": "Stage 2: Cross-Encoder Reranker",
    "deep_reranker": "Stage 3: Deep Reranker",
    "selector": "Stage 4: LLM Selector",
}

_LABEL_PAD = 32


def _load_summary(path: Path) -> dict[str, dict[str, object]]:
    """Load a latency_summary.json and return the ``stages`` dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("stages", {}))


def _load_stage_files(run_dir: Path) -> dict[str, dict[str, object]]:
    """Fallback: compute per-task ms from individual pipeline-stage files."""
    result: dict[str, dict[str, object]] = {}
    for stage_file in sorted(run_dir.glob("pipeline-stage*.json")):
        stage_data = json.loads(stage_file.read_text(encoding="utf-8"))
        stage_name = stage_data.get("stage", stage_file.stem)
        per_task: dict[str, float] = {}
        values: list[float] = []
        for tr in stage_data.get("task_results", []):
            ms = tr.get("elapsed_ms", 0.0)
            per_task[tr["task_id"]] = ms
            values.append(ms)
        result[stage_name] = {
            "per_task_ms": per_task,
            "aggregate": _compute_latency_stats(values),
        }
    return result


def load_latency_data(
    run_dir: Path,
) -> dict[str, dict[str, object]]:
    """Load per-stage latency data from a pipeline run directory."""
    summary_path = run_dir / "latency_summary.json"
    if summary_path.exists():
        return _load_summary(summary_path)
    return _load_stage_files(run_dir)


def _compute_total_stats(
    stages: dict[str, dict[str, object]],
) -> dict[str, float]:
    """Compute total latency stats from per-task sums across all stages."""
    # Collect all task IDs present in any stage
    task_ids: set[str] = set()
    for stage_info in stages.values():
        per_task = stage_info.get("per_task_ms", {})
        if isinstance(per_task, dict):
            task_ids.update(per_task.keys())

    if not task_ids:
        return {"median_ms": 0.0, "mean_ms": 0.0, "p95_ms": 0.0}

    # Sum per-task latencies across stages
    totals: list[float] = []
    for tid in task_ids:
        total = 0.0
        for stage_info in stages.values():
            per_task = stage_info.get("per_task_ms", {})
            if isinstance(per_task, dict):
                total += per_task.get(tid, 0.0)
        totals.append(total)

    return _compute_latency_stats(totals)


def render_latex_table(stages: dict[str, dict[str, object]]) -> str:
    """Render LaTeX tabular content from per-stage latency stats (seconds)."""
    lines = [
        r"\begin{tabular}{lrrr}",
        r"  \toprule",
        r"  \textbf{Stage} & \textbf{Median} & \textbf{Mean} & \textbf{P95} \\",
        r"  \midrule",
    ]

    for stage_key, label in _STAGE_LABELS.items():
        if stage_key not in stages:
            continue
        raw_agg = stages[stage_key]["aggregate"]
        agg = dict(raw_agg) if isinstance(raw_agg, dict) else {}
        median_s = agg["median_ms"] / 1000
        mean_s = agg["mean_ms"] / 1000
        p95_s = agg["p95_ms"] / 1000
        lines.append(
            f"  {label:<{_LABEL_PAD}}& {median_s:.1f} & {mean_s:.1f} & {p95_s:.1f} \\\\"
        )

    total = _compute_total_stats(stages)
    t_med = total["median_ms"] / 1000
    t_mean = total["mean_ms"] / 1000
    t_p95 = total["p95_ms"] / 1000
    total_label = r"\textbf{Total}"

    lines.extend(
        [
            r"  \midrule",
            f"  {total_label:<{_LABEL_PAD}}"
            f"& \\textbf{{{t_med:.1f}}}"
            f" & \\textbf{{{t_mean:.1f}}}"
            f" & \\textbf{{{t_p95:.1f}}} \\\\",
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate latency LaTeX table")
    parser.add_argument(
        "--run-dir",
        default=str(_DEFAULT_RUN_DIR),
        help="Pipeline run directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="paper/tables/5_latency.tex",
        help="Output .tex file (stdout if empty string)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    stages = load_latency_data(run_dir)
    if not stages:
        print("No stage data found in", run_dir, file=sys.stderr)
        sys.exit(1)

    table = render_latex_table(stages)
    output = Path(args.output) if args.output else None
    write_or_print(table.splitlines(), output)


if __name__ == "__main__":
    main()

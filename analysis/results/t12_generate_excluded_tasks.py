"""Generate the excluded benchmark tasks table (tab:excluded_tasks).

Renders a LaTeX table of excluded benchmark tasks grouped by benchmark and
exclusion category.

Usage::

    uv run python -m analysis.results.t12_generate_excluded_tasks

    uv run python -m analysis.results.t12_generate_excluded_tasks \
        --output paper/tables/12_excluded_tasks.tex
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

from analysis.results.utils.latex_utils import write_or_print

_DEFAULT_OUTPUT = Path("paper/tables/12_excluded_tasks.tex")


class ExcludedTask(TypedDict):
    benchmark: str
    task: str
    category: str
    reason: str


EXCLUDED_TASKS: list[ExcludedTask] = [
    # Infrastructure — Daytona sandbox errors
    {
        "benchmark": "SkillsBench",
        "task": "earthquake-phase-association",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "fix-druid-loophole-cve",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "fix-erlang-ssh-cve",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "latex-formula-extraction",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "organize-messy-files",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "parallel-tfidf-search",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "quantum-numerical-simulation",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "setup-fuzzing-py",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "shock-analysis-demand",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "syzkaller-ppdev-syzlang",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "taxonomy-tree-merge",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "video-filler-word-remover",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    {
        "benchmark": "SkillsBench",
        "task": "video-tutorial-indexer",
        "category": "Infrastructure",
        "reason": "Daytona sandbox error",
    },
    # Infrastructure — Docker build failures
    {
        "benchmark": "SkillsBench",
        "task": "multilingual-video-dubbing",
        "category": "Infrastructure",
        "reason": "TTS model download fails intermittently during Docker build",
    },
    {
        "benchmark": "SkillsBench",
        "task": "scheduling-email-assistant",
        "category": "Infrastructure",
        "reason": "Docker compose mounts hardcoded host path",
    },
    {
        "benchmark": "SkillsBench",
        "task": "speaker-diarization-subtitles",
        "category": "Infrastructure",
        "reason": "Whisper model loading triggers OOM during Docker build",
    },
    # Reproducibility — intermittent oracle failures
    {
        "benchmark": "SkillsBench",
        "task": "dynamic-object-aware-egomotion",
        "category": "Reproducibility",
        "reason": "Oracle outputs non-serializable numpy types",
    },
    {
        "benchmark": "SkillsBench",
        "task": "fix-build-google-auto",
        "category": "Reproducibility",
        "reason": "Maven build flaky under Docker networking",
    },
    {
        "benchmark": "SkillsBench",
        "task": "pedestrian-traffic-counting",
        "category": "Reproducibility",
        "reason": "Oracle depends on external vision API keys",
    },
    {
        "benchmark": "SkillsBench",
        "task": "r2r-mpc-control",
        "category": "Reproducibility",
        "reason": "MPC settling time sensitive to Docker CPU scheduling",
    },
    {
        "benchmark": "SkillsBench",
        "task": "reserves-at-risk-calc",
        "category": "Reproducibility",
        "reason": "Numerical precision varies across platforms",
    },
    {
        "benchmark": "SkillsBench",
        "task": "simpo-code-reproduction",
        "category": "Reproducibility",
        "reason": "Build timeout; passes with extended timeout",
    },
    # Terminal-Bench
    {
        "benchmark": "Terminal-Bench",
        "task": "train-fasttext",
        "category": "Resource",
        "reason": "Large corpus download and model training",
    },
]


_BENCH_MIN: dict[str, int] = {"SkillsBench": 30, "Terminal-Bench": 28}
_CAT_MIN = 13


def _compute_task_pads(tasks: list[ExcludedTask]) -> list[int]:
    """Compute per-task column widths for aligned output.

    Within each (benchmark, category) group the task column is padded to
    ``max(longest_task + 1, benchmark_min)``.  The last task in a group
    falls back to its individual width, while the first task in a group
    also considers the previous group's width for visual continuity.
    """
    # Identify groups in order
    groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    group_order: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for idx, t in enumerate(tasks):
        key = (t["benchmark"], t["category"])
        groups[key].append(idx)
        if key not in seen:
            group_order.append(key)
            seen.add(key)

    group_pad: dict[tuple[str, str], int] = {}
    for key, idxs in groups.items():
        bmin = _BENCH_MIN.get(key[0], 30)
        longest = max(len(tasks[i]["task"]) for i in idxs)
        group_pad[key] = max(longest + 1, bmin)

    pads = [0] * len(tasks)
    for gi, gkey in enumerate(group_order):
        bmin = _BENCH_MIN.get(gkey[0], 30)
        prev_gpad = group_pad[group_order[gi - 1]] if gi > 0 else bmin
        gpad = group_pad[gkey]
        idxs = groups[gkey]
        for pos, idx in enumerate(idxs):
            ind = max(len(tasks[idx]["task"]) + 1, bmin)
            is_first = pos == 0
            is_last = pos == len(idxs) - 1
            if is_last:
                pads[idx] = ind
            elif is_first:
                pads[idx] = max(ind, gpad, prev_gpad)
            else:
                pads[idx] = gpad
    return pads


def render_table(tasks: list[ExcludedTask]) -> list[str]:
    """Return LaTeX tabular content for tab:excluded_tasks (no table wrapper)."""
    lines: list[str] = [
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llll}",
        r"  \toprule",
        (
            r"  \textbf{Benchmark} & \textbf{Task}"
            r" & \textbf{Category} & \textbf{Reason} \\"
        ),
        r"  \midrule",
    ]

    task_pads = _compute_task_pads(tasks)

    prev_benchmark = ""
    prev_category = ""

    for i, task in enumerate(tasks):
        bench = task["benchmark"]
        category = task["category"]

        if prev_benchmark and bench != prev_benchmark:
            lines.append(r"  \midrule")
            prev_category = ""

        if prev_category and category != prev_category and bench == prev_benchmark:
            lines.append(r"  \midrule")

        tpad = task_pads[i]
        cpad = max(len(category), _CAT_MIN)
        lines.append(
            f"  {bench} & {task['task']:<{tpad}}"
            f" & {category:<{cpad}} & {task['reason']} \\\\"
        )

        prev_benchmark = bench
        prev_category = category

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
        description="Generate tab:excluded_tasks excluded benchmark tasks table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    table_lines = render_table(EXCLUDED_TASKS)
    write_or_print(table_lines, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

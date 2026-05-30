"""Retrieved-skill usage and help/hurt analysis on SkillsBench.

How often do agents use retrieved *non-oracle* skills, and do skills help or
hurt downstream success? Both are answered from the existing SkillsBench
trajectories (no new agent runs):

* Part A: oracle vs. non-oracle injection share and per-skill load rates.
* Part B: SkillFlow-vs-baseline help/hurt, partitioned by what retrieval
  injected (oracle-present / non-oracle-only / no-skills) so the non-oracle
  effect is isolated, with an overall paired-bootstrap test.

Writes the help/hurt partition table to ``paper/tables/18_skill_usage.tex``
(camera-ready artifact) plus a JSON record of all numbers.

Usage::

    uv run python -m analysis.results.t18_generate_skill_usage

    uv run python -m analysis.results.t18_generate_skill_usage \\
        --sf-prefix sk-skillflow-inject-gpt5mini-medium- \\
        --bl-prefix sk-baseline-gpt5mini-medium- \\
        --tasks-dir integration/skillsbench/tasks \\
        --latex-out paper/tables/18_skill_usage.tex \\
        --out outputs/analysis/skill_usage_helphurt.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from analysis.results.utils.latex_utils import table_env, write_or_print
from analysis.results.utils.usage_utils import (
    PartitionStat,
    collect_outcomes,
    collect_usage,
    helphurt,
    usage_summary,
)
from analysis.results.utils.usage_utils import _run_dirs as run_dirs

logger = logging.getLogger(__name__)

_GROUP_LABELS = {
    "all": "All tasks (full SkillFlow effect)",
    "oracle_present": "Oracle-present (>=1 oracle injected)",
    "nonoracle_only": "Non-oracle-only (isolates non-oracle effect)",
    "no_skills": "No skills injected",
}

_LATEX_GROUP_LABELS = {
    "all": "All tasks",
    "oracle_present": "Oracle present",
    "nonoracle_only": "Non-oracle only",
    "no_skills": "No skills",
}


def render_usage_table(s: dict[str, float | int]) -> str:
    """Part A markdown table."""
    return "\n".join(
        [
            "| Skill class | Injected | Per task | Load rate |",
            "| --- | --- | --- | --- |",
            f"| Oracle | {s['n_oracle']} | "
            f"{(s['n_oracle'] / s['n_task_runs']):.2f} | "
            f"{s['oracle_load_rate']:.0%} |",
            f"| Non-oracle | {s['n_nonoracle']} | "
            f"{s['nonoracle_per_task']:.2f} | {s['nonoracle_load_rate']:.0%} |",
        ]
    )


def render_partition_table(parts: list[PartitionStat]) -> str:
    """Part B markdown table (SkillFlow vs. baseline by injection group)."""
    lines = [
        "| Group | n | SF pass | Baseline pass | Helped | Hurt | Tie |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in parts:
        lines.append(
            f"| {_GROUP_LABELS.get(p.group, p.group)} | {p.n} | "
            f"{p.sf_passrate:.3f} | {p.bl_passrate:.3f} | "
            f"{p.helped} | {p.hurt} | {p.tie} |"
        )
    return "\n".join(lines)


def render_latex(parts: list[PartitionStat]) -> list[str]:
    """Part B as a paper table body (tabular only; wrapper lives in main.tex)."""
    header = (
        r"\textbf{Injected} & \textbf{Tasks} & \textbf{SkillFlow} & "
        r"\textbf{Baseline} & \textbf{Helped} & \textbf{Hurt}"
    )
    top, bot = table_env("lccccc", header)
    body = [
        f"  {_LATEX_GROUP_LABELS.get(p.group, p.group)} & "
        f"{p.n} & {p.sf_passrate:.3f} & "
        f"{p.bl_passrate:.3f} & {p.helped} & {p.hurt} \\\\"
        for p in parts
    ]
    return [*top, *body, *bot]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sf-prefix", default="sk-skillflow-inject-gpt5mini-medium-")
    ap.add_argument("--bl-prefix", default="sk-baseline-gpt5mini-medium-")
    ap.add_argument(
        "--tasks-dir", type=Path, default=Path("integration/skillsbench/tasks")
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/skill_usage_helphurt.json"),
    )
    ap.add_argument(
        "--latex-out",
        type=Path,
        default=Path("paper/tables/18_skill_usage.tex"),
    )
    args = ap.parse_args()

    sf = run_dirs(args.sf_prefix)
    bl = run_dirs(args.bl_prefix)
    if not sf or not bl:
        logger.error("Missing runs (sf=%d, bl=%d).", len(sf), len(bl))
        return 1

    records = collect_usage(sf, args.tasks_dir)
    summary = usage_summary(records)
    outcomes = collect_outcomes(sf, bl, args.tasks_dir)
    hh = helphurt(outcomes)

    print(f"SkillFlow runs: {len(sf)}   Baseline runs: {len(bl)}")
    print(
        f"Task-runs analyzed: {summary['n_task_runs']}   "
        f"Common tasks: {len(outcomes)}\n"
    )
    print("## Part A — retrieved-skill use rate")
    print(
        f"Non-oracle skills are {summary['nonoracle_share']:.0%} of all "
        f"injected ({summary['skills_per_task']:.2f} skills/task).\n"
    )
    print(render_usage_table(summary))
    print("\n## Part B — help/hurt by injection group (mean pass-rate over runs)\n")
    print(render_partition_table(hh.partitions))
    print(
        f"\nOverall SkillFlow - baseline pass-rate diff: "
        f"{hh.overall_diff:+.3f} (paired bootstrap p={hh.overall_p:.4f})"
    )
    if hh.nonoracle_only_tasks:
        print("Non-oracle-only tasks: " + ", ".join(hh.nonoracle_only_tasks))
    if hh.hurt_tasks:
        print("Hurt tasks (SF < baseline): " + ", ".join(hh.hurt_tasks))

    write_or_print(render_latex(hh.partitions), args.latex_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "sf_runs": [d.name for d in sf],
                "bl_runs": [d.name for d in bl],
                "part_a_use_rate": summary,
                "part_b_helphurt": hh.model_dump(),
                "outcomes": [o.model_dump() for o in outcomes],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared skill stats utilities for computing per-task injection/loading metrics.

Used by both ``generate_adoption`` (for bootstrap CIs) and
``summarize_skill_stats`` (for aggregate TSV output).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from analysis.utils.find_skill_patterns import scan_trajectory

logger = logging.getLogger(__name__)

_EVAL_DIR = Path("outputs/evaluation")


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _find_runs(group: dict[str, Any]) -> list[Path]:
    if "runs" in group:
        return [_EVAL_DIR / r for r in group["runs"] if (_EVAL_DIR / r).exists()]
    prefix = group["prefix"]
    return sorted(
        d for d in _EVAL_DIR.iterdir() if d.name.startswith(prefix) and d.is_dir()
    )


def _read_skill_name(skill_md: Path) -> str:
    """Read the ``name`` field from SKILL.md YAML frontmatter."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("\"'")
        if line == "---" and text.index(line) > 0:
            break
    return ""


def _get_retrieved_skills(skills_dir: Path) -> tuple[set[str], dict[str, str]]:
    """Get real skill names and folder-to-name mapping from agent/skills/.

    Returns:
        (real_names, folder_to_name) where real_names is the set of actual
        skill names from SKILL.md frontmatter, and folder_to_name maps
        folder names to real names (for matching trajectory references).
    """
    real_names: set[str] = set()
    folder_to_name: dict[str, str] = {}
    if not skills_dir.exists():
        return real_names, folder_to_name
    for child in skills_dir.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        skill_md = child / "SKILL.md"
        name = _read_skill_name(skill_md) if skill_md.exists() else child.name
        name = name or child.name
        real_names.add(name)
        folder_to_name[child.name] = name
    return real_names, folder_to_name


def _load_gt_names(gt_tasks_dir: Path, task_name: str) -> set[str]:
    """Load GT skill names for a task from SkillsBench."""
    gt_dir = gt_tasks_dir / task_name / "environment" / "skills"
    if not gt_dir.exists():
        return set()
    return {s.name for s in gt_dir.iterdir() if s.is_dir()}


def _scan_run(
    run_dir: Path,
    *,
    all_injected: bool = False,
    gt_tasks_dir: Path | None = None,
    letta_gt: bool = False,
) -> dict[str, Any]:
    """Scan one run for pass/inject/load/gt-hit counts."""
    total = 0
    injected_tasks = 0
    loaded_tasks = 0
    total_skills = 0
    gt_recalls: list[float] = []
    per_task_skills: list[int] = []
    per_task_retrieved: list[float] = []
    per_task_loaded: list[float] = []

    for td in sorted(run_dir.iterdir()):
        if not td.is_dir() or not (td / "result.json").exists():
            continue
        total += 1
        task_name = td.name.rsplit("__", 1)[0]

        skills_dir = td / "agent" / "skills"
        real_names, folder_to_name = _get_retrieved_skills(skills_dir)
        folder_names = set(folder_to_name.keys())

        # For all_injected (Golden), the GT skills ARE the retrieved skills
        if all_injected and gt_tasks_dir and not folder_names:
            gt_as_retrieved = _load_gt_names(gt_tasks_dir, task_name)
            folder_names = gt_as_retrieved
            real_names = gt_as_retrieved

        n_skills = len(folder_names)
        has_skill = n_skills > 0 or all_injected
        if has_skill:
            injected_tasks += 1
            total_skills += n_skills

        traj = td / "agent" / "trajectory.json"
        loaded = False
        if traj.exists() and has_skill:
            findings = scan_trajectory(traj)
            if findings:
                loaded_tasks += 1
                loaded = True

        per_task_skills.append(n_skills)
        per_task_retrieved.append(1.0 if has_skill else 0.0)
        per_task_loaded.append(1.0 if loaded else 0.0)

        if gt_tasks_dir:
            gt_names = _load_gt_names(gt_tasks_dir, task_name)
            if gt_names:
                gt_recalls.append(len(real_names & gt_names) / len(gt_names))
        elif letta_gt:
            gt_recalls.append(1.0 if task_name in real_names else 0.0)

    return {
        "total": total,
        "injected_tasks": injected_tasks,
        "loaded_tasks": loaded_tasks,
        "total_skills": total_skills,
        "gt_recalls": gt_recalls,
        "per_task_skills": per_task_skills,
        "per_task_retrieved": per_task_retrieved,
        "per_task_loaded": per_task_loaded,
    }


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def compute_group_stats(group: dict[str, Any]) -> dict[str, str]:
    """Compute aggregate stats for a run group."""
    runs = _find_runs(group)
    if not runs:
        return {"label": group["label"], "n_runs": "0"}

    agg_keys = [
        "total",
        "injected_tasks",
        "loaded_tasks",
        "total_skills",
    ]
    agg = dict.fromkeys(agg_keys, 0)
    all_gt_recalls: list[float] = []

    all_injected_flag = group.get("all_injected", False)
    letta_gt = group.get("letta_gt", False)
    gt_tasks_dir_str = group.get("gt_tasks_dir")
    gt_tasks_dir = Path(gt_tasks_dir_str) if gt_tasks_dir_str else None
    for run_dir in runs:
        stats = _scan_run(
            run_dir,
            all_injected=all_injected_flag,
            gt_tasks_dir=gt_tasks_dir,
            letta_gt=letta_gt,
        )
        for k in agg_keys:
            agg[k] += stats[k]
        all_gt_recalls.extend(stats["gt_recalls"])

    n_runs = len(runs)
    tasks_per_run = agg["total"] // n_runs if n_runs else 0

    # Metric 1: retrieval rate (injected_tasks / total)
    retr_rate = agg["injected_tasks"] / agg["total"] * 100 if agg["total"] else 0
    retrieved_str = (
        f"{agg['injected_tasks']}/{agg['total']} ({retr_rate:.0f}%)"
        if agg["total"]
        else "-"
    )

    # Metric 2: avg skills per task (total_skills / total)
    avg_skills = agg["total_skills"] / agg["total"] if agg["total"] else 0

    # Metric 3: GT recall (mean of per-task recalls)
    gt_recall = (
        f"{sum(all_gt_recalls) / len(all_gt_recalls) * 100:.0f}%"
        if all_gt_recalls
        else "0%"
    )

    # Metric 4: task use rate (loaded_tasks / injected_tasks)
    task_use = (
        f"{agg['loaded_tasks'] / agg['injected_tasks'] * 100:.0f}%"
        if agg["injected_tasks"]
        else "0%"
    )

    return {
        "label": group["label"],
        "n_runs": str(n_runs),
        "tasks": str(tasks_per_run),
        "retrieved": retrieved_str,
        "avg_skills": f"{avg_skills:.1f}",
        "gt_recall": gt_recall,
        "task_use": task_use,
    }


class PerTaskStats(BaseModel, frozen=True):
    """Per-task arrays for a run group, pooled across runs."""

    label: str
    skills_per_task: list[float]
    retrieved: list[float]
    loaded: list[float]
    gt_recalls: list[float]


def compute_per_task_stats(group: dict[str, Any]) -> PerTaskStats | None:
    """Return per-task arrays pooled across runs for bootstrap CIs."""
    runs = _find_runs(group)
    if not runs:
        return None

    all_skills: list[float] = []
    all_retrieved: list[float] = []
    all_loaded: list[float] = []
    all_gt: list[float] = []

    all_injected_flag = group.get("all_injected", False)
    letta_gt = group.get("letta_gt", False)
    gt_tasks_dir_str = group.get("gt_tasks_dir")
    gt_tasks_dir = Path(gt_tasks_dir_str) if gt_tasks_dir_str else None

    for run_dir in runs:
        stats = _scan_run(
            run_dir,
            all_injected=all_injected_flag,
            gt_tasks_dir=gt_tasks_dir,
            letta_gt=letta_gt,
        )
        all_skills.extend(float(x) for x in stats["per_task_skills"])
        all_retrieved.extend(stats["per_task_retrieved"])
        all_loaded.extend(stats["per_task_loaded"])
        all_gt.extend(stats["gt_recalls"])

    return PerTaskStats(
        label=group["label"],
        skills_per_task=all_skills,
        retrieved=all_retrieved,
        loaded=all_loaded,
        gt_recalls=all_gt,
    )

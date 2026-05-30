"""Per-skill usage + help/hurt analysis over SkillsBench benchmark runs.

Quantifies how often agents use retrieved non-oracle skills, and whether
skills help or hurt downstream success, from existing trajectories — no new
runs. Part A classifies each injected skill oracle/non-oracle and whether the
agent loaded it (opened SKILL.md outside the "Available skills" listing).
Part B partitions tasks by what retrieval injected (oracle-present /
non-oracle-only / no-skills) and compares SkillFlow vs. baseline within each;
the non-oracle-only group isolates the non-oracle effect. Observational.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from analysis.stats.bootstrap import paired_bootstrap_test
from analysis.utils.find_skill_patterns import scan_trajectory

if TYPE_CHECKING:
    from collections.abc import Iterable

_EVAL_DIR = Path("outputs/evaluation")
# Folder name embedded in a real SKILL.md read path (the listing is excluded
# by scan_trajectory, so any folder we see here was genuinely opened).
_SKILL_PATH = re.compile(r"/skills/([^/\"'\s]+)/SKILL\.md")
_PASS_THRESHOLD = 1.0


class SkillUsage(BaseModel, frozen=True):
    """One injected skill in one (task, run)."""

    run: str
    task: str
    folder: str
    name: str
    is_oracle: bool
    loaded: bool


class TaskOutcome(BaseModel, frozen=True):
    """Per-task pass vectors (one entry per run) plus injected composition.

    The selector cache is keyed per task, so the injected skill set is fixed
    across runs; ``n_injected`` / ``n_oracle`` are therefore per-task counts.
    """

    task: str
    baseline: tuple[bool, ...]
    skillflow: tuple[bool, ...]
    n_injected: int
    n_oracle: int


class PartitionStat(BaseModel, frozen=True):
    """SkillFlow vs. baseline within one injection group."""

    group: str
    n: int
    sf_passrate: float
    bl_passrate: float
    helped: int
    hurt: int
    tie: int


class HelpHurt(BaseModel, frozen=True):
    """Partitioned help/hurt result + overall paired-bootstrap test."""

    partitions: list[PartitionStat]
    overall_diff: float
    overall_p: float
    nonoracle_only_tasks: list[str]
    hurt_tasks: list[str]


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


def _injected_skills(task_dir: Path) -> dict[str, str]:
    """Map injected skill folder -> real name (skips dot-folders)."""
    skills_dir = task_dir / "agent" / "skills"
    out: dict[str, str] = {}
    if not skills_dir.exists():
        return out
    for child in skills_dir.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        md = child / "SKILL.md"
        name = (_read_skill_name(md) if md.exists() else child.name) or child.name
        out[child.name] = name
    return out


def _gt_names(gt_tasks_dir: Path, task: str) -> set[str]:
    """Ground-truth (oracle) skill folder names for a task."""
    gt = gt_tasks_dir / task / "environment" / "skills"
    if not gt.exists():
        return set()
    return {s.name for s in gt.iterdir() if s.is_dir()}


def _loaded_folders(task_dir: Path) -> set[str]:
    """Folders whose SKILL.md the agent actually opened (listing excluded)."""
    traj = task_dir / "agent" / "trajectory.json"
    if not traj.exists():
        return set()
    folders: set[str] = set()
    for finding in scan_trajectory(traj):
        folders.update(_SKILL_PATH.findall(finding["context"]))
    return folders


def _reward(task_dir: Path) -> float | None:
    """Read the verifier reward (pass = 1.0); None if absent."""
    rfile = task_dir / "verifier" / "reward.txt"
    try:
        return float(rfile.read_text().strip())
    except (OSError, ValueError):
        return None


def _run_dirs(prefix: str, eval_dir: Path = _EVAL_DIR) -> list[Path]:
    if not eval_dir.exists():
        return []
    return sorted(
        d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith(prefix)
    )


def _task_name(task_dir: Path) -> str:
    return task_dir.name.rsplit("__", 1)[0]


def _iter_tasks(run_dir: Path) -> Iterable[Path]:
    for td in sorted(run_dir.iterdir()):
        if td.is_dir() and (td / "result.json").exists():
            yield td


def collect_usage(run_dirs: list[Path], gt_tasks_dir: Path) -> list[SkillUsage]:
    """Per-skill usage records across all runs (Part A input)."""
    records: list[SkillUsage] = []
    for run in run_dirs:
        for td in _iter_tasks(run):
            task = _task_name(td)
            gt = _gt_names(gt_tasks_dir, task)
            loaded = _loaded_folders(td)
            for folder, name in _injected_skills(td).items():
                records.append(
                    SkillUsage(
                        run=run.name,
                        task=task,
                        folder=folder,
                        name=name,
                        is_oracle=folder in gt or name in gt,
                        loaded=folder in loaded,
                    )
                )
    return records


def collect_outcomes(
    sf_runs: list[Path],
    bl_runs: list[Path],
    gt_tasks_dir: Path,
) -> list[TaskOutcome]:
    """Per-task pass vectors for SkillFlow vs. baseline (Part B input)."""

    def _passes(runs: list[Path]) -> dict[str, list[bool]]:
        acc: dict[str, list[bool]] = {}
        for run in runs:
            for td in _iter_tasks(run):
                r = _reward(td)
                acc.setdefault(_task_name(td), []).append(
                    r is not None and r >= _PASS_THRESHOLD
                )
        return acc

    def _composition(runs: list[Path]) -> dict[str, tuple[int, int]]:
        # task -> (n_injected, n_oracle); injected set is fixed across runs.
        comp: dict[str, tuple[int, int]] = {}
        for run in runs:
            for td in _iter_tasks(run):
                task = _task_name(td)
                if task in comp:
                    continue
                gt = _gt_names(gt_tasks_dir, task)
                inj = _injected_skills(td)
                n_or = sum(1 for f, n in inj.items() if f in gt or n in gt)
                comp[task] = (len(inj), n_or)
        return comp

    sf, bl = _passes(sf_runs), _passes(bl_runs)
    comp = _composition(sf_runs)
    outcomes: list[TaskOutcome] = []
    for task in sorted(set(sf) & set(bl)):
        n_inj, n_or = comp.get(task, (0, 0))
        outcomes.append(
            TaskOutcome(
                task=task,
                baseline=tuple(bl[task]),
                skillflow=tuple(sf[task]),
                n_injected=n_inj,
                n_oracle=n_or,
            )
        )
    return outcomes


def _rate(num: float, den: int) -> float:
    return num / den if den else 0.0


def usage_summary(records: list[SkillUsage]) -> dict[str, float | int]:
    """Part A: injection counts and oracle vs. non-oracle load rates."""
    oracle = [r for r in records if r.is_oracle]
    nonoracle = [r for r in records if not r.is_oracle]
    n_tasks = len({(r.run, r.task) for r in records})
    return {
        "n_task_runs": n_tasks,
        "n_injected": len(records),
        "n_oracle": len(oracle),
        "n_nonoracle": len(nonoracle),
        "nonoracle_share": _rate(len(nonoracle), len(records)),
        "skills_per_task": _rate(len(records), n_tasks),
        "nonoracle_per_task": _rate(len(nonoracle), n_tasks),
        "oracle_load_rate": _rate(sum(r.loaded for r in oracle), len(oracle)),
        "nonoracle_load_rate": _rate(sum(r.loaded for r in nonoracle), len(nonoracle)),
    }


def _mean(passes: tuple[bool, ...]) -> float:
    return sum(passes) / len(passes) if passes else 0.0


def _group_of(o: TaskOutcome) -> str:
    if o.n_injected == 0:
        return "no_skills"
    return "oracle_present" if o.n_oracle > 0 else "nonoracle_only"


def _partition_stat(group: str, items: list[TaskOutcome]) -> PartitionStat:
    sf = [_mean(o.skillflow) for o in items]
    bl = [_mean(o.baseline) for o in items]
    helped = sum(s > b for s, b in zip(sf, bl, strict=True))
    hurt = sum(s < b for s, b in zip(sf, bl, strict=True))
    return PartitionStat(
        group=group,
        n=len(items),
        sf_passrate=_rate(sum(sf), len(sf)) if sf else 0.0,
        bl_passrate=_rate(sum(bl), len(bl)) if bl else 0.0,
        helped=helped,
        hurt=hurt,
        tie=len(items) - helped - hurt,
    )


def helphurt(outcomes: list[TaskOutcome]) -> HelpHurt:
    """Part B: per-injection-group SkillFlow-vs-baseline + paired bootstrap.

    The non-oracle-only group isolates whether retrieved non-oracle skills
    help or hurt. The overall paired-bootstrap test summarises the full
    SkillFlow effect (all groups), consistent with the Table 1 comparison.
    """
    order = ["all", "oracle_present", "nonoracle_only", "no_skills"]
    groups: dict[str, list[TaskOutcome]] = {"all": list(outcomes)}
    for o in outcomes:
        groups.setdefault(_group_of(o), []).append(o)

    partitions = [_partition_stat(g, groups[g]) for g in order if groups.get(g)]
    test = paired_bootstrap_test(
        [_mean(o.skillflow) for o in outcomes],
        [_mean(o.baseline) for o in outcomes],
    )
    return HelpHurt(
        partitions=partitions,
        overall_diff=test.observed_diff,
        overall_p=test.p_value,
        nonoracle_only_tasks=sorted(o.task for o in groups.get("nonoracle_only", [])),
        hurt_tasks=sorted(
            o.task for o in outcomes if _mean(o.skillflow) < _mean(o.baseline)
        ),
    )

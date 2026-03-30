"""Oracle vs. community skill quality comparison utilities.

Loads oracle and community SKILL.md files, computes structural proxy
metrics via :mod:`quality_utils`, and compares populations with
Mann-Whitney U tests (Holm-Bonferroni corrected).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel
from scipy.stats import mannwhitneyu

from analysis.results.utils.quality_utils import (
    SkillMetrics,
    compute_metrics,
    compute_metrics_text_only,
)
from analysis.stats.proportions import holm_bonferroni

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

_NO_DIR = Path("/dev/null")


def repo_root() -> Path:
    """Resolve the repository root (directory containing pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for ancestor in (current, *current.parents):
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    return current


def load_oracle_skills(tasks_dir: Path) -> dict[str, tuple[str, Path]]:
    """Load oracle skills from SkillsBench task directories.

    Returns ``{task_id/skill_name: (text, skill_dir)}``.
    """
    skills: dict[str, tuple[str, Path]] = {}
    for task_dir in sorted(d for d in tasks_dir.iterdir() if d.is_dir()):
        skills_dir = task_dir / "environment" / "skills"
        if not skills_dir.exists():
            continue
        for entry in sorted(skills_dir.iterdir()):
            skill_file = entry / "SKILL.md"
            if skill_file.is_file():
                key = f"{task_dir.name}/{entry.name}"
                text = skill_file.read_text(encoding="utf-8", errors="replace")
                skills[key] = (text, entry)
    return skills


def load_community_skills(
    corpus_dir: Path | None = None,
    contents_path: Path | None = None,
) -> dict[str, tuple[str, Path]]:
    """Load community skills from a corpus directory or JSON fallback.

    Prefers *corpus_dir* (gives both text + directory), falling back to
    *contents_path* (``skill_contents.json``, text only — bundle metrics
    will be zero).
    """
    if corpus_dir and corpus_dir.is_dir():
        skills: dict[str, tuple[str, Path]] = {}
        for skill_md in sorted(corpus_dir.rglob("SKILL.md")):
            key = str(skill_md.parent.relative_to(corpus_dir))
            text = skill_md.read_text(encoding="utf-8", errors="replace")
            skills[key] = (text, skill_md.parent)
        return skills

    if contents_path and contents_path.is_file():
        data: dict[str, str] = json.loads(
            contents_path.read_text(encoding="utf-8"),
        )
        return {k: (v, _NO_DIR) for k, v in data.items()}

    return {}


def build_metrics(
    skill_map: dict[str, tuple[str, Path]],
) -> list[SkillMetrics]:
    """Compute metrics for each skill, using text-only when dir is absent."""
    results: list[SkillMetrics] = []
    for text, skill_dir in skill_map.values():
        if skill_dir == _NO_DIR or not skill_dir.is_dir():
            results.append(compute_metrics_text_only(text))
        else:
            results.append(compute_metrics(text, skill_dir))
    return results


# ---------------------------------------------------------------------------
# Population statistics & comparison
# ---------------------------------------------------------------------------


class PopulationStats(BaseModel, frozen=True):
    """Descriptive statistics for a single metric across a population."""

    metric: str
    n: int
    median: float
    mean: float
    std: float


class ComparisonResult(BaseModel, frozen=True):
    """Mann-Whitney U test result comparing two populations on one metric."""

    metric: str
    oracle: PopulationStats
    community: PopulationStats
    u_statistic: float
    p_value: float
    p_adjusted: float


EXCLUDED = {"has_examples", "has_yaml_frontmatter", "bundled_file_count"}
_FIELDS = [f for f in SkillMetrics.model_fields if f not in EXCLUDED]


def _population_stats(
    metric_name: str,
    values: Sequence[float],
) -> PopulationStats:
    if len(values) == 0:
        return PopulationStats(
            metric=metric_name,
            n=0,
            median=0.0,
            mean=0.0,
            std=0.0,
        )
    arr = np.array(values, dtype=np.float64)
    return PopulationStats(
        metric=metric_name,
        n=len(arr),
        median=float(np.median(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    )


def compare_populations(
    oracle_metrics: list[SkillMetrics],
    community_metrics: list[SkillMetrics],
) -> list[ComparisonResult]:
    """Compare oracle and community populations on each metric.

    Raw p-values are adjusted for multiple comparisons using the
    Holm-Bonferroni method.
    """
    raw: list[tuple[str, PopulationStats, PopulationStats, float, float]] = []
    for field in _FIELDS:
        o_vals = [float(getattr(m, field)) for m in oracle_metrics]
        c_vals = [float(getattr(m, field)) for m in community_metrics]
        o_stats = _population_stats(field, o_vals)
        c_stats = _population_stats(field, c_vals)
        if len(o_vals) > 0 and len(c_vals) > 0:
            stat, pval = mannwhitneyu(
                o_vals,
                c_vals,
                alternative="two-sided",
            )
        else:
            stat, pval = 0.0, 1.0
        raw.append((field, o_stats, c_stats, float(stat), float(pval)))

    adjusted = holm_bonferroni([r[4] for r in raw])
    return [
        ComparisonResult(
            metric=field,
            oracle=o_stats,
            community=c_stats,
            u_statistic=u,
            p_value=p,
            p_adjusted=p_adj,
        )
        for (field, o_stats, c_stats, u, p), p_adj in zip(
            raw,
            adjusted,
            strict=True,
        )
    ]


def print_pvalues(results: list[ComparisonResult]) -> None:
    """Print raw and Holm-Bonferroni adjusted p-values to stdout."""
    header = f"{'Metric':<25} {'U statistic':>14} {'p (raw)':>14} {'p (adjusted)':>14}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: x.p_adjusted):
        print(
            f"{r.metric:<25} {r.u_statistic:>14.1f}"
            f" {r.p_value:>14.6e} {r.p_adjusted:>14.6e}"
        )

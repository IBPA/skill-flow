"""Adoption table row computation, formatting, and significance testing.

Extracted from ``t2_generate_adoption`` to keep file sizes manageable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from analysis.results.utils.skill_utils import compute_per_task_stats
from analysis.stats.bootstrap import bootstrap_ci, paired_bootstrap_test

if TYPE_CHECKING:
    from analysis.stats.types import ConfidenceInterval

# -- Adoption group definitions ----------------------------------------

SK_ADOPTION_GROUPS: list[dict[str, object]] = [
    {
        "label": "Oracle",
        "benchmark": "skillsbench",
        "prefix": "sk-skillsbench-inject-gpt5mini-medium-",
        "all_injected": True,
        "gt_tasks_dir": "integration/skillsbench/tasks",
    },
    {
        "label": "Vercel",
        "benchmark": "skillsbench",
        "prefix": "sk-vercel-find-skills-gpt5mini-medium-",
    },
    {
        "label": "SkillFlow top-1 (Ours)",
        "benchmark": "skillsbench",
        "prefix": "sk-skillflow-inject-top1-gpt5mini-medium-",
        "gt_tasks_dir": "integration/skillsbench/tasks",
    },
    {
        "label": "SkillFlow (Ours)",
        "benchmark": "skillsbench",
        "prefix": "sk-skillflow-inject-gpt5mini-medium-",
        "gt_tasks_dir": "integration/skillsbench/tasks",
    },
]

TB_ADOPTION_GROUPS: list[dict[str, object]] = [
    {
        "label": "Vercel",
        "benchmark": "terminal-bench",
        "prefix": "tb-vercel-find-skills-gpt5mini-medium-",
        "letta_gt": True,
    },
    {
        "label": "SkillFlow top-1 (Ours)",
        "benchmark": "terminal-bench",
        "prefix": "tb-skillflow-inject-v4.0-top1-",
        "letta_gt": True,
    },
    {
        "label": "SkillFlow (Ours)",
        "benchmark": "terminal-bench",
        "runs": [
            "tb-skillflow-inject-gpt5mini-medium-20260311-010051",
            "tb-skillflow-inject-gpt5mini-medium-20260312-192602",
            "tb-skillflow-inject-gpt5mini-medium-20260313-133825",
        ],
        "letta_gt": True,
    },
]

# Column widths for alignment padding (characters)
LABEL_W, SKILLS_W, RETR_W, GT_W, LOADED_W = 24, 4, 6, 5, 5

# Value column keys and their widths (used for bold detection)
VAL_COLS: list[tuple[str, int]] = [
    ("retr", RETR_W),
    ("gt", GT_W),
    ("loaded", LOADED_W),
]


# -- Significance result model -----------------------------------------


class SignificanceResult(BaseModel, frozen=True):
    """Raw and adjusted p-value for one condition vs. baseline."""

    label: str
    raw_p: float
    adjusted_p: float


# -- Row data model ----------------------------------------------------


class Row:
    """One row of adoption data with both formatted strings and raw floats."""

    __slots__ = ("label", "per_task", "raws", "vals")

    def __init__(
        self,
        label: str,
        vals: dict[str, str],
        raws: dict[str, float],
        per_task: dict[str, list[float]] | None = None,
    ) -> None:
        self.label = label
        self.vals = vals
        self.raws = raws
        self.per_task = per_task or {}


# -- Row data helpers --------------------------------------------------


def compute_row(
    group: dict[str, Any],
    *,
    is_tb: bool = False,
) -> Row | None:
    """Compute plain-value cells for one adoption group."""
    stats = compute_per_task_stats(group)
    if stats is None or not stats.skills_per_task:
        return None

    skills_ci = bootstrap_ci(stats.skills_per_task)
    retr_ci = bootstrap_ci(stats.retrieved)
    loaded_ci = bootstrap_ci(stats.loaded)

    def _fmt_ci(ci_obj: ConfidenceInterval, scale: float = 100.0) -> str:
        m = ci_obj.mean * scale
        lo = ci_obj.ci_lo * scale
        hi = ci_obj.ci_hi * scale
        return f"{m:.1f}" r"{\scriptsize~[" f"{lo:.1f}, {hi:.1f}" r"]}"

    skills_str = f"{skills_ci.mean:.1f}"
    retr_raw = retr_ci.mean * 100
    loaded_raw = loaded_ci.mean * 100

    gt_raw = math.nan
    if is_tb:
        gt_str = "---"
    elif stats.gt_recalls:
        gt_ci = bootstrap_ci(stats.gt_recalls)
        gt_raw = gt_ci.mean * 100
        gt_str = _fmt_ci(gt_ci)
    else:
        gt_str = "---"

    vals = {
        "skills": skills_str,
        "retr": _fmt_ci(retr_ci),
        "gt": gt_str,
        "loaded": _fmt_ci(loaded_ci),
    }
    raws = {"retr": retr_raw, "gt": gt_raw, "loaded": loaded_raw}
    per_task = {"retr": stats.retrieved, "loaded": stats.loaded}
    return Row(
        label=str(group.get("label", "")),
        vals=vals,
        raws=raws,
        per_task=per_task,
    )


def apply_bold(rows: list[Row], *, exclude: set[int]) -> None:
    """Mark the best value per column with ``\\textbf{}`` (in-place)."""
    for key, _width in VAL_COLS:
        eligible: dict[int, float] = {}
        for i, row in enumerate(rows):
            if i in exclude:
                continue
            v = row.raws.get(key)
            if v is not None and not math.isnan(v):
                eligible[i] = v
        if not eligible:
            continue
        best = max(eligible.values())
        for i, v in eligible.items():
            if v == best:
                val = rows[i].vals[key]
                ci_start = val.find(r"{\scriptsize")
                if ci_start > 0:
                    val = f"\\textbf{{{val[:ci_start]}}}{val[ci_start:]}"
                else:
                    val = f"\\textbf{{{val}}}"
                rows[i].vals[key] = val


def apply_significance(
    rows: list[Row],
    *,
    baseline_label: str,
    col: str = "loaded",
    n_comparisons: int = 1,
) -> list[SignificanceResult]:
    """Add significance markers vs. baseline (in-place) and return p-values.

    Applies Holm-Bonferroni correction for *n_comparisons* tests.
    Appends ``$^{*}$`` (p < 0.05) or ``$^{**}$`` (p < 0.01) to the
    formatted value in ``col``.

    Returns a list of ``SignificanceResult`` for every non-baseline row
    that had comparable per-task data.
    """
    baseline_idx = next(
        (i for i, r in enumerate(rows) if r.label == baseline_label),
        None,
    )
    if baseline_idx is None:
        return []
    baseline_data = rows[baseline_idx].per_task.get(col)
    if not baseline_data:
        return []

    # Collect p-values for all non-baseline rows
    results: list[tuple[int, float]] = []
    for i, row in enumerate(rows):
        if i == baseline_idx:
            continue
        row_data = row.per_task.get(col)
        if not row_data or len(row_data) != len(baseline_data):
            continue
        test = paired_bootstrap_test(row_data, baseline_data)
        results.append((i, test.p_value))

    # Holm-Bonferroni correction
    sig_results: list[SignificanceResult] = []
    sorted_results = sorted(results, key=lambda x: x[1])
    for rank, (idx, p) in enumerate(sorted_results, 1):
        adj_p = p * (n_comparisons - rank + 1)
        adj_p = min(adj_p, 1.0)
        sig_results.append(
            SignificanceResult(label=rows[idx].label, raw_p=p, adjusted_p=adj_p),
        )
        if adj_p < 0.01:
            marker = "$^{**}$"
        elif adj_p < 0.05:
            marker = "$^{*}$"
        else:
            continue
        # Insert marker after the (possibly bold-wrapped) number, before CI
        val = rows[idx].vals[col]
        ci_start = val.find(r"{\scriptsize")
        val = val[:ci_start] + marker + val[ci_start:] if ci_start > 0 else val + marker
        rows[idx].vals[col] = val

    return sig_results


def pad_bold(cell: str, width: int) -> str:
    """Pad a cell that may contain ``\\textbf{}`` around the number."""
    if cell.startswith("\\textbf{"):
        close = cell.index("}", len("\\textbf{"))
        number = cell[len("\\textbf{") : close]
        rest = cell[close + 1 :]
        visible = number + rest
        trailing = max(width - len(visible), 0) * " "
        return f"\\textbf{{{number}}}{rest}{trailing}"
    return f"{cell:<{width}}"


def format_row(row: Row, *, is_oracle: bool = False) -> str:
    """Format a ``Row`` as a LaTeX table row with column alignment."""
    label = row.label
    skills = row.vals["skills"]
    retr = row.vals["retr"]
    gt = row.vals["gt"]
    loaded = row.vals["loaded"]

    if is_oracle:
        label = f"{f'\\textit{{{label}}}':<{LABEL_W}}"
        skills = f"\\textit{{{skills}}}"
        retr = f"\\textit{{{retr}}}"
        gt = f"\\textit{{{gt}}}"
        loaded = f"\\textit{{{loaded}}}"
    else:
        label = f"{label:<{LABEL_W}}"
        skills = f"{skills:<{SKILLS_W}}"
        retr = pad_bold(retr, RETR_W)
        gt = pad_bold(gt, GT_W)
        loaded = pad_bold(loaded, LOADED_W)

    return f"  {label} & {skills} & {retr} & {gt} & {loaded} \\\\"

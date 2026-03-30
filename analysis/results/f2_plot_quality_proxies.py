"""Structural proxy metric comparison: oracle vs. community skills.

Top row: box plots for continuous metrics (sorted by significance).
Bottom row: grouped bar chart with 95% Wilson CIs for boolean metrics.

Usage::

    uv run python -m analysis.results.f2_plot_quality_proxies
    uv run python -m analysis.results.f2_plot_quality_proxies \
        -o paper/figures/2_quality_proxies.png
    uv run python -m analysis.results.f2_plot_quality_proxies --print-pvalues
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.results.utils.quality_comparison_utils import (
    EXCLUDED,
    build_metrics,
    compare_populations,
    load_community_skills,
    load_oracle_skills,
    print_pvalues,
    repo_root,
)
from analysis.results.utils.quality_plot_utils import draw_bars, draw_boxplots
from analysis.results.utils.quality_utils import SkillMetrics

_DEFAULT_TASKS_DIR = "integration/skillsbench/tasks"
_DEFAULT_CORPUS_DIR = "data/skills/skillsmp"
_DEFAULT_SKILL_CONTENTS = "outputs/indices/bge-base/skill_contents.json"

_FIELDS = [f for f in SkillMetrics.model_fields if f not in EXCLUDED]
_BOOL_FIELDS = {"has_bundled_files", "has_scripts", "has_references"}
_CONT_FIELDS = [f for f in _FIELDS if f not in _BOOL_FIELDS]


def _plot(
    oracle_vals: dict[str, list[float]],
    community_vals: dict[str, list[float]],
    p_adjusted: dict[str, float],
    output: Path | None,
) -> None:
    cont_sorted = sorted(
        _CONT_FIELDS,
        key=lambda f: p_adjusted.get(f, 1.0),
    )
    bool_sorted = sorted(
        [f for f in _BOOL_FIELDS if f in _FIELDS],
        key=lambda f: p_adjusted.get(f, 1.0),
    )
    n_cont = len(cont_sorted)
    fig, axes = plt.subplots(
        2,
        n_cont,
        figsize=(2.5 * n_cont, 7.5),
        gridspec_kw={"height_ratios": [3, 2]},
    )
    box_axes = axes[0] if n_cont > 1 else [axes[0]]
    bar_row = axes[1] if n_cont > 1 else [axes[1]]

    draw_boxplots(
        list(box_axes),
        cont_sorted,
        oracle_vals,
        community_vals,
        p_adjusted,
    )

    for a in bar_row:
        a.set_visible(False)
    bar_ax = fig.add_subplot(2, 1, 2)
    draw_bars(
        bar_ax,
        bool_sorted,
        oracle_vals,
        community_vals,
        p_adjusted,
    )

    fig.tight_layout(h_pad=4.0)
    dest = output or Path("paper/figures/2_quality_proxies.png")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    print(f"Saved to {dest}")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Generate quality proxy comparison figure."""
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Quality proxy comparison figure",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=root / _DEFAULT_TASKS_DIR,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=root / _DEFAULT_CORPUS_DIR,
    )
    parser.add_argument(
        "--skill-contents",
        type=Path,
        default=root / _DEFAULT_SKILL_CONTENTS,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("paper/figures/2_quality_proxies.png"),
        help="Output path",
    )
    parser.add_argument(
        "--print-pvalues",
        action="store_true",
        help="Print raw and Holm-Bonferroni adjusted p-values to stdout",
    )
    args = parser.parse_args(argv)

    oracle = load_oracle_skills(args.tasks_dir)
    community = load_community_skills(
        args.corpus_dir,
        args.skill_contents,
    )
    oracle_m = build_metrics(oracle)
    community_m = build_metrics(community)
    results = compare_populations(oracle_m, community_m)

    if args.print_pvalues:
        print_pvalues(results)

    p_adj = {r.metric: r.p_adjusted for r in results}
    o_vals = {f: [float(getattr(m, f)) for m in oracle_m] for f in _FIELDS}
    c_vals = {f: [float(getattr(m, f)) for m in community_m] for f in _FIELDS}
    _plot(o_vals, c_vals, p_adj, args.output)


if __name__ == "__main__":
    main()

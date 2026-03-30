"""Plotting helpers for the quality-proxy comparison figure (F3).

Extracts box-plot and bar-chart drawing logic so the main generator
stays under the 300-line project limit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.container import BarContainer

BLUE = "#4C72B0"
ORANGE = "#DD8452"

LABELS: dict[str, str] = {
    "code_fraction": "Code fraction",
    "ordered_list_count": "Ordered list\ncount",
    "code_block_count": "Code block\ncount",
    "heading_count": "Heading\ncount",
    "inline_code_density": "Inline code\ndensity",
    "word_count": "Word count",
    "has_bundled_files": "Has bundled\nfiles",
    "has_scripts": "Has scripts",
    "has_references": "Has references",
}


def p_label(p: float) -> str:
    """Format adjusted p-value for display."""
    if p < 0.001:
        return "$p_{adj}$ < 0.001"
    if p < 0.05:
        return f"$p_{{adj}}$ = {p:.3f}"
    return f"$p_{{adj}}$ = {p:.2f}"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def draw_boxplots(
    box_axes: list[plt.Axes],
    cont_sorted: list[str],
    oracle_vals: dict[str, list[float]],
    community_vals: dict[str, list[float]],
    p_adjusted: dict[str, float],
) -> None:
    """Draw box plots for continuous metrics on the top row."""
    for i, field in enumerate(cont_sorted):
        ax = box_axes[i]
        p_adj = p_adjusted.get(field, 1.0)
        significant = p_adj < 0.05
        alpha = 0.7 if significant else 0.3
        hatch = None if significant else "///"

        bp = ax.boxplot(
            [oracle_vals[field], community_vals[field]],
            tick_labels=["Oracle", "Community"],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for box in bp["boxes"]:
            box.set_alpha(alpha)
            if hatch:
                box.set_hatch(hatch)
        bp["boxes"][0].set_facecolor(BLUE)
        bp["boxes"][1].set_facecolor(ORANGE)

        label = LABELS.get(field, field.replace("_", " ").title())
        weight = "bold" if significant else "normal"
        ax.set_title(
            f"{label}\n({p_label(p_adj)})",
            fontsize=13,
            fontweight=weight,
        )
        ax.tick_params(axis="both", labelsize=11)

    # Row label on leftmost axis
    box_axes[0].set_ylabel("Continuous metrics", fontsize=13)

    # Hatching legend on rightmost axis
    solid = mpatches.Patch(facecolor="gray", alpha=0.7, label="Significant")
    hatched = mpatches.Patch(
        facecolor="gray",
        alpha=0.3,
        hatch="///",
        label="Not significant",
    )
    box_axes[-1].legend(
        handles=[solid, hatched],
        fontsize=10,
        loc="upper right",
    )


def draw_bars(
    bar_ax: plt.Axes,
    bool_sorted: list[str],
    oracle_vals: dict[str, list[float]],
    community_vals: dict[str, list[float]],
    p_adjusted: dict[str, float],
) -> None:
    """Draw grouped bar chart for boolean metrics on the bottom row."""
    x = np.arange(len(bool_sorted))
    width = 0.3

    o_ns = [len(oracle_vals[f]) for f in bool_sorted]
    c_ns = [len(community_vals[f]) for f in bool_sorted]
    o_ks = [int(sum(oracle_vals[f])) for f in bool_sorted]
    c_ks = [int(sum(community_vals[f])) for f in bool_sorted]

    def _props(ks: list[int], ns: list[int]) -> list[float]:
        return [k / n * 100 if n else 0 for k, n in zip(ks, ns, strict=True)]

    o_props, c_props = _props(o_ks, o_ns), _props(c_ks, c_ns)

    def _errs(
        props: list[float],
        ks: list[int],
        ns: list[int],
    ) -> tuple[list[float], list[float]]:
        cis = [_wilson_ci(k, n) for k, n in zip(ks, ns, strict=True)]
        lo = [p - ci_lo * 100 for p, (ci_lo, _) in zip(props, cis, strict=True)]
        hi = [h * 100 - p for p, (_, h) in zip(props, cis, strict=True)]
        return lo, hi

    o_lo, o_hi = _errs(o_props, o_ks, o_ns)
    c_lo, c_hi = _errs(c_props, c_ks, c_ns)

    bars_o: BarContainer = bar_ax.bar(
        x - width / 2,
        o_props,
        width,
        label="Oracle",
        color=BLUE,
        alpha=0.7,
        yerr=[o_lo, o_hi],
        capsize=4,
        error_kw={"linewidth": 1.2},
    )
    bars_c: BarContainer = bar_ax.bar(
        x + width / 2,
        c_props,
        width,
        label="Community",
        color=ORANGE,
        alpha=0.7,
        yerr=[c_lo, c_hi],
        capsize=4,
        error_kw={"linewidth": 1.2},
    )

    labels = []
    for f in bool_sorted:
        name = LABELS.get(f, f.replace("_", " ").title()).replace("\n", " ")
        p_adj = p_adjusted.get(f, 1.0)
        labels.append(f"{name}\n({p_label(p_adj)})")

    bar_ax.set_xticks(x)
    weights = [
        "bold" if p_adjusted.get(f, 1.0) < 0.05 else "normal" for f in bool_sorted
    ]
    bar_ax.set_xticklabels(labels, fontsize=13)
    for tick_label, w in zip(bar_ax.get_xticklabels(), weights, strict=True):
        tick_label.set_fontweight(w)
    bar_ax.set_ylabel("Binary metrics\nProportion (%)", fontsize=13)
    bar_ax.tick_params(axis="y", labelsize=11)
    bar_ax.legend(fontsize=12, loc="upper right")
    bar_ax.set_ylim(0, 65)

    _apply_significance_styling(bar_ax, bool_sorted, p_adjusted, bars_o, bars_c)


def _apply_significance_styling(
    bar_ax: plt.Axes,
    bool_sorted: list[str],
    p_adjusted: dict[str, float],
    bars_o: BarContainer,
    bars_c: BarContainer,
) -> None:
    """Apply alpha/hatch styling and percentage labels to bar containers."""
    for idx, f in enumerate(bool_sorted):
        p_adj = p_adjusted.get(f, 1.0)
        significant = p_adj < 0.05
        alpha = 0.7 if significant else 0.3
        hatch = None if significant else "///"
        for container in (bars_o, bars_c):
            container[idx].set_alpha(alpha)
            if hatch:
                container[idx].set_hatch(hatch)

    for container in (bars_o, bars_c):
        for rect in container:
            h = rect.get_height()
            if h > 0:
                bar_ax.annotate(
                    f"{h:.0f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

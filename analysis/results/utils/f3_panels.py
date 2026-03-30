"""Panel plotting functions for the multi-query impact figure (F3)."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from analysis.results.utils.f3_data import (
    M_LABEL,
    M_STYLE,
    OUTPUT_K,
    RECALL_KS,
    STAGE_ORDER,
    ReportData,
    StageData,
    bootstrap_delta_ci,
    holm_bonferroni,
    mrr,
    recall_curve,
    summ,
    task_recall_at_k,
)

logger = logging.getLogger(__name__)


def _jitter(ks: list[int], idx: int, n: int) -> list[float]:
    if n <= 1:
        return [float(k) for k in ks]
    spread = 0.08
    factor = 1.0 + spread * (2 * idx / (n - 1) - 1)
    return [k * factor for k in ks]


def _fill_spread(
    ax: plt.Axes,
    ks: list[int],
    all_curves: dict[int, list[float]],
) -> None:
    m_lo, m_hi = min(all_curves), max(all_curves)
    if m_lo == m_hi:
        return
    ys_lo, ys_hi = all_curves[m_lo], all_curves[m_hi]
    ks_f = [float(k) for k in ks]
    color_lo = M_STYLE.get(m_lo, {"color": "#999"})["color"]
    color_hi = M_STYLE.get(m_hi, {"color": "#999"})["color"]
    for j in range(len(ks_f) - 1):
        seg_x = [ks_f[j], ks_f[j + 1]]
        seg_lo = [ys_lo[j], ys_lo[j + 1]]
        seg_hi = [ys_hi[j], ys_hi[j + 1]]
        mid_lo = (seg_lo[0] + seg_lo[1]) / 2
        mid_hi = (seg_hi[0] + seg_hi[1]) / 2
        ax.fill_between(
            seg_x,
            seg_lo,
            seg_hi,
            color=color_hi if mid_hi >= mid_lo else color_lo,
            alpha=0.10,
            zorder=1,
        )


def _annotate_endpoints(
    ax: plt.Axes,
    endpoints: list[tuple[float, float, str, str]],
) -> None:
    endpoints.sort(key=lambda t: t[1])
    n_ep = len(endpoints)
    for i, (xp, yp, color, txt) in enumerate(endpoints):
        y_off = (i - n_ep / 2 + 0.5) * 16
        ax.annotate(
            txt,
            (xp, yp),
            textcoords="offset points",
            xytext=(8, y_off),
            fontsize=7,
            color=color,
            fontweight="bold",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5},
        )


def _find_crossover(
    ks: list[int],
    ys_lo_m: list[float],
    ys_hi_m: list[float],
) -> float | None:
    """Find k where ys_hi_m overtakes ys_lo_m (log-interpolated)."""
    for j in range(len(ks) - 1):
        diff_a = ys_lo_m[j] - ys_hi_m[j]
        diff_b = ys_lo_m[j + 1] - ys_hi_m[j + 1]
        if diff_a > 0 and diff_b < 0:
            log_ka = math.log10(ks[j])
            log_kb = math.log10(ks[j + 1])
            t = diff_a / (diff_a - diff_b)
            return 10 ** (log_ka + t * (log_kb - log_ka))
    return None


def plot_recall(
    ax: plt.Axes,
    stage: str,
    m_reports: dict[int, ReportData],
    label: str,
    *,
    annotate_crossover: bool = False,
) -> list[plt.Artist]:
    """Draw recall@k lines for one stage; return handles for shared legend."""
    max_k = OUTPUT_K[stage]
    ks = [k for k in RECALL_KS if k <= max_k]
    handles: list[plt.Artist] = []
    endpoints: list[tuple[float, float, str, str]] = []
    all_curves: dict[int, list[float]] = {}
    n_m = len(m_reports)

    for idx, mv in enumerate(sorted(m_reports)):
        sty = M_STYLE.get(mv, {"color": "#999", "marker": "o", "ls": "-"})
        curve = recall_curve(summ(m_reports[mv]))
        ys = [curve.get(k, float("nan")) for k in ks]
        all_curves[mv] = ys
        xs_jittered = _jitter(ks, idx, n_m)

        lines = ax.plot(
            xs_jittered,
            ys,
            color=sty["color"],
            marker=sty["marker"],
            linestyle=sty["ls"],
            linewidth=2.0,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=M_LABEL.get(mv, f"$M{{=}}{mv}$"),
            zorder=3,
        )
        handles.extend(lines)
        if ys and not math.isnan(ys[-1]):
            endpoints.append(
                (xs_jittered[-1], ys[-1], sty["color"], f"{ys[-1]:.3f}"),
            )

    _fill_spread(ax, ks, all_curves)
    _annotate_endpoints(ax, endpoints)

    if annotate_crossover and 1 in all_curves and len(all_curves) > 1:
        hi_m = max(m for m in all_curves if m != 1)
        xover = _find_crossover(ks, all_curves[1], all_curves[hi_m])
        if xover is not None:
            ax.axvline(xover, color="#666", ls=":", lw=1.0, zorder=2)
            y_mid = ax.get_ylim()[0] + 0.25 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            ax.text(
                xover * 1.15,
                y_mid,
                f"$M{{=}}{hi_m}$ overtakes\n$M{{=}}1$ at $k\\approx{round(xover)}$",
                fontsize=6.5,
                color="#444",
                ha="left",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "fc": "white",
                    "ec": "#ccc",
                    "alpha": 0.85,
                },
            )

    if len(ks) > 1:
        ax.set_xscale("log")
    ax.set_xlim(ks[0] * 0.7, ks[-1] * 2.0)
    all_ys = [y for ys in all_curves.values() for y in ys if not math.isnan(y)]
    if all_ys:
        y_min, y_max = min(all_ys), max(all_ys)
        pad = (y_max - y_min) * 0.15
        ax.set_ylim(max(0, y_min - pad), min(1.05, y_max + pad))
    else:
        ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    tick_labels = ["1K" if k == 1000 else str(k) for k in ks]
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("$k$", fontsize=9)
    ax.set_ylabel("Recall@$k$", fontsize=9)
    ax.grid(alpha=0.2, zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(f"({label})  {stage}", fontsize=9, fontweight="bold", loc="left")
    return handles


def plot_delta(ax: plt.Axes, data: StageData, label: str) -> None:
    """Horizontal lollipop: relative delta from M=1 at output cutoff, with 95% CIs."""
    present = [s for s in STAGE_ORDER if s in data and 1 in data[s]]
    y_pos = np.arange(len(present))
    m_vals = sorted(m for m in {m for s in present for m in data[s]} if m != 1)
    n_m = len(m_vals)
    offsets = np.linspace(-0.15, 0.15, n_m) if n_m > 1 else np.array([0.0])

    # Pass 1: compute all bootstrap statistics and collect raw p-values
    stats: dict[int, dict[str, tuple[float, float, float]]] = {}
    raw_pvals: dict[str, float] = {}
    for mv in m_vals:
        stats[mv] = {}
        for stage in present:
            k = OUTPUT_K[stage]
            test_report = data[stage].get(mv)
            if test_report is None:
                stats[mv][stage] = (0.0, 0.0, 0.0)
                continue
            base_vals = task_recall_at_k(data[stage][1], k)
            test_vals = task_recall_at_k(test_report, k)
            pt, lo, hi, pv = bootstrap_delta_ci(base_vals, test_vals)
            stats[mv][stage] = (pt, lo, hi)
            raw_pvals[f"{stage}|M={mv}"] = pv
    adj = holm_bonferroni(raw_pvals)
    for key in sorted(adj, key=adj.__getitem__):
        stage_s, m_s = key.split("|")
        pt, lo, hi = stats[int(m_s.split("=")[1])][stage_s]
        logger.info(
            "  %s %s: delta=%+.1f%% CI=[%.1f, %.1f] p_adj=%.4f",
            stage_s,
            m_s,
            pt,
            lo,
            hi,
            adj[key],
        )

    # Pass 2: draw
    all_extremes: list[float] = []
    for i, mv in enumerate(m_vals):
        sty = M_STYLE.get(mv, {"color": "#999", "marker": "o"})
        deltas = [stats[mv][s][0] for s in present]
        ci_los = [stats[mv][s][1] for s in present]
        ci_his = [stats[mv][s][2] for s in present]
        yy = y_pos + offsets[i]
        xerr_lo = [d - lo for d, lo in zip(deltas, ci_los, strict=True)]
        xerr_hi = [hi - d for d, hi in zip(deltas, ci_his, strict=True)]
        all_extremes.extend(ci_los)
        all_extremes.extend(ci_his)
        for yi, d in zip(yy, deltas, strict=True):
            ax.plot([0, d], [yi, yi], color=sty["color"], lw=1.5, zorder=2)
        ax.errorbar(
            deltas,
            yy,
            xerr=[xerr_lo, xerr_hi],
            fmt="none",
            ecolor=sty["color"],
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=2,
        )
        ax.scatter(
            deltas,
            yy,
            color=sty["color"],
            marker=sty["marker"],
            s=60,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
        for yi, d in zip(yy, deltas, strict=True):
            ax.text(
                d + 1.2 if d >= 0 else d - 1.2,
                yi - 0.12,
                f"{d:+.1f}%",
                fontsize=7.5,
                va="center",
                ha="left" if d >= 0 else "right",
                color=sty["color"],
                fontweight="medium",
            )

    ax.axvline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{s}\n(R@{OUTPUT_K[s]})" for s in present],
        fontsize=8,
    )
    ax.set_xlabel("Relative $\\Delta$ from $M{=}1$ (%)", fontsize=9)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.invert_yaxis()
    if all_extremes:
        pad = (max(all_extremes) - min(all_extremes)) * 0.30
        ax.set_xlim(min(all_extremes) - pad, max(all_extremes) + pad)
    ax.set_title(
        f"({label})  Output-cutoff $\\Delta$",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )


def plot_mrr(ax: plt.Axes, data: StageData, label: str) -> None:
    """Grouped bar chart: MRR by stage and M value."""
    present = [s for s in STAGE_ORDER if s in data]
    all_ms = sorted({m for s in present for m in data[s]})
    n_m = len(all_ms)
    x = np.arange(len(present))
    width = 0.7 / n_m

    for i, mv in enumerate(all_ms):
        sty = M_STYLE.get(mv, {"color": "#999"})
        vals = [mrr(summ(data[s][mv])) if mv in data[s] else 0.0 for s in present]
        offset = (i - (n_m - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.88,
            color=sty["color"],
            label=M_LABEL.get(mv, f"$M{{=}}{mv}$"),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.008,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="medium",
                color=sty["color"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in present], fontsize=8)
    ax.set_ylabel("MRR", fontsize=9)
    ax.set_ylim(
        0, min(1.0, max(mrr(summ(data[s][m])) for s in present for m in data[s]) * 1.25)
    )
    ax.grid(axis="y", alpha=0.2, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(
        f"({label})  MRR across stages",
        fontsize=9,
        fontweight="bold",
        loc="left",
    )

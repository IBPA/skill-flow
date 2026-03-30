"""Composite figure: multi-query impact across pipeline stages.

2-row layout: summary panels (a-b) on top, recall curves (c-e) on bottom.
Triple-redundant encoding (color + linestyle + marker) for accessibility.

Usage::

    uv run python -m analysis.results.f3_plot_query_impact \
        --report retriever 1 path/to/q1.json \
        --report retriever 3 path/to/q3.json \
        --report retriever 5 path/to/q5.json \
        --report reranker  1 path/to/q1.json \
        ...
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from analysis.results.utils.f3_data import (
    M_LABEL,
    M_STYLE,
    StageData,
    parse_reports,
)
from analysis.results.utils.f3_panels import (
    plot_delta,
    plot_mrr,
    plot_recall,
)

logger = logging.getLogger(__name__)


def plot_composite(data: StageData, output_path: Path) -> None:
    """Create the 5-panel composite figure.

    Layout (GridSpec):
      Row 0:  (a) delta lollipop with CIs  |  (b) MRR grouped bars
      Row 1:  (c) Retriever  |  (d) Shallow Reranker  |  (e) Deep Reranker
    """
    fig = plt.figure(figsize=(8.5, 5.8))
    gs = GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1, 1],
        hspace=0.45,
        wspace=0.55,
        top=0.93,
        bottom=0.08,
        left=0.07,
        right=0.97,
    )

    ax_delta = fig.add_subplot(gs[0, :3])
    ax_mrr = fig.add_subplot(gs[0, 3:])
    ax_ret = fig.add_subplot(gs[1, :2])
    ax_shallow = fig.add_subplot(gs[1, 2:4])
    ax_deep = fig.add_subplot(gs[1, 4:])

    plot_delta(ax_delta, data, "a")
    plot_mrr(ax_mrr, data, "b")

    panels = [
        (ax_ret, "Retriever", "c", True),
        (ax_shallow, "Shallow Reranker", "d", False),
        (ax_deep, "Deep Reranker", "e", False),
    ]
    legend_handles: list[plt.Artist] = []
    for ax, stage, lbl, crossover in panels:
        if stage not in data:
            ax.set_visible(False)
            continue
        h = plot_recall(ax, stage, data[stage], lbl, annotate_crossover=crossover)
        if not legend_handles:
            legend_handles = h

    if legend_handles:
        labels = [M_LABEL[m] for m in sorted(M_STYLE)]
        fig.legend(
            legend_handles,
            labels,
            loc="upper center",
            ncol=3,
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, 1.04),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved -> %s", output_path)


_EXP = "outputs/experiments"
_DEFAULT_REPORTS: list[list[str]] = [
    ["retriever", "1", f"{_EXP}/retriever-querygen-v2/bge-base-q1-max.json"],
    ["retriever", "3", f"{_EXP}/retriever-querygen-v2/bge-base-q3-rrf.json"],
    ["retriever", "5", f"{_EXP}/retriever-querygen-v2/bge-base-q5-rrf.json"],
    [
        "reranker",
        "1",
        f"{_EXP}/reranker-comparison/cross-encoder-ms-marco-minilm-l-6-v2-512chars-q1-max.json",
    ],
    [
        "reranker",
        "3",
        f"{_EXP}/reranker-comparison/cross-encoder-ms-marco-minilm-l-6-v2-512chars-q3-rrf.json",
    ],
    [
        "reranker",
        "5",
        f"{_EXP}/reranker-comparison/cross-encoder-ms-marco-minilm-l-6-v2-512chars-q5-rrf.json",
    ],
    [
        "deep",
        "1",
        f"{_EXP}/deep-reranker-comparison/baai-bge-reranker-v2-m3-4096chars-q1-max.json",
    ],
    [
        "deep",
        "3",
        f"{_EXP}/deep-reranker-comparison/baai-bge-reranker-v2-m3-4096chars-q3-rrf.json",
    ],
    [
        "deep",
        "5",
        f"{_EXP}/deep-reranker-comparison/baai-bge-reranker-v2-m3-4096chars-q5-rrf.json",
    ],
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Composite multi-query impact figure (5-panel).",
    )
    parser.add_argument(
        "--report",
        nargs=3,
        action="append",
        metavar=("STAGE", "M", "PATH"),
        help="stage name, M value, report JSON path (defaults built-in)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/3_query_impact.png"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = parse_reports(args.report or _DEFAULT_REPORTS)
    plot_composite(data, args.output)


if __name__ == "__main__":
    main()

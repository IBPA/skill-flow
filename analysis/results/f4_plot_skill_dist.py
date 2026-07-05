"""Plot the distribution of skills across content-length buckets.

Produces a bar chart showing how many skills fall into each character-length
bucket: <512, 512-1024, 1024-2048, 2048-4096, 4096-8192, >=8192. Lengths are
measured in characters because the reranker truncates skill content by
characters (``max_content_chars``), so the coverage curve must be expressed in
the same unit as the truncation windows.

Usage::

    uv run python -m analysis.results.f4_plot_skill_dist \
        [--corpus-path PATH | --contents-json PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skill_flow.config import load_config

logger = logging.getLogger(__name__)

_BUCKET_EDGES = [0, 512, 1024, 2048, 4096, 8192]
_BUCKET_LABELS = [
    "[0, 512)",
    "[512, 1024)",
    "[1024, 2048)",
    "[2048, 4096)",
    "[4096, 8192)",
    "[8192, ∞)",
]


def _char_count(text: str) -> int:
    """Return the character length of a skill's content."""
    return len(text)


def _bucket_index(char_count: int) -> int:
    """Return the bucket index for a given character count."""
    for i in range(len(_BUCKET_EDGES) - 1):
        if char_count < _BUCKET_EDGES[i + 1]:
            return i
    return len(_BUCKET_EDGES) - 1


def _collect_char_counts(corpus_path: Path) -> list[int]:
    index_file = corpus_path / "_metadata" / "index.json"
    data = json.loads(index_file.read_text(encoding="utf-8"))
    skills_dict: dict[str, dict[str, object]] = data["skills"]

    counts: list[int] = []
    for key in skills_dict:
        skill_file = corpus_path / key / "SKILL.md"
        try:
            content = skill_file.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            logger.warning("Missing SKILL.md for %s — counted as 0 chars", key)
            counts.append(0)
            continue
        counts.append(_char_count(content))
    return counts


def _collect_char_counts_from_contents(contents_json: Path) -> list[int]:
    """Char-count skills from a persisted ``skill_contents.json`` mapping.

    Used when the raw corpus is unavailable; the JSON maps each skill key to its
    full SKILL.md text (the same content the reranker scores).
    """
    data = json.loads(contents_json.read_text(encoding="utf-8"))
    values = data.values() if isinstance(data, dict) else data
    return [_char_count(v) for v in values if isinstance(v, str)]


def _plot_distribution(counts: list[int], output_path: Path) -> None:
    buckets = [0] * len(_BUCKET_LABELS)
    for c in counts:
        buckets[_bucket_index(c)] += 1

    cumulative = []
    running = 0
    for b in buckets:
        running += b
        cumulative.append(running / len(counts) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(_BUCKET_LABELS, buckets, edgecolor="black", linewidth=0.5)

    for bar, count in zip(bars, buckets, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(buckets) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2 = ax.twinx()
    ax2.plot(
        range(len(_BUCKET_LABELS)),
        cumulative,
        color="red",
        marker="o",
        linewidth=1.5,
        markersize=5,
        label="Cumulative %",
    )
    for i, pct in enumerate(cumulative):
        ax2.annotate(
            f"{pct:.1f}%",
            (i, pct),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="red",
        )
    ax2.set_ylabel("Cumulative coverage (%)")
    ax2.set_ylim(0, 105)

    ax.set_title(f"Skill Content Length Distribution (n={len(counts):,})")
    ax.set_xlabel("Character count")
    ax.set_ylabel("Number of skills")
    ax2.legend(loc="center right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot skill content-length distribution by character buckets.",
    )
    parser.add_argument("--corpus-path", type=Path, default=None)
    parser.add_argument(
        "--contents-json",
        type=Path,
        default=None,
        help="Path to a skill_contents.json mapping (used when the raw corpus "
        "is unavailable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/4_skill_dist.png"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.contents_json is not None:
        counts = _collect_char_counts_from_contents(args.contents_json.resolve())
    else:
        corpus_path = args.corpus_path
        if corpus_path is None:
            cfg = load_config()
            corpus_path = Path(cfg.index.input_corpus_path)
        counts = _collect_char_counts(corpus_path.resolve())
    if not counts:
        print("No skills found — check corpus / contents path.")
        return

    _plot_distribution(counts, args.output)


if __name__ == "__main__":
    main()

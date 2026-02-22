"""EDA: Skill file length distribution analysis.

Computes character, word, line, and token counts for every SKILL.md in the
corpus and produces summary statistics, a CSV export, and distribution plots.

Usage::

    uv run python -m analysis.eda.skill_length [--corpus-path PATH] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skill_flow.config import load_config
from skill_flow.corpus.loader import load_content, load_corpus

logger = logging.getLogger(__name__)

# Token count thresholds shown as vertical lines on the CDF plot.
_CDF_THRESHOLDS = [512, 1024, 2048, 4096, 8192]


def _token_count(text: str) -> int:
    """Count tokens using tiktoken cl100k_base, with char//4 fallback."""
    try:
        import tiktoken  # noqa: PLC0415

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


# ------------------------------------------------------------------
# Metrics collection
# ------------------------------------------------------------------

_COLUMNS = [
    "key",
    "name",
    "source",
    "char_count",
    "word_count",
    "line_count",
    "token_count",
]
_METRIC_KEYS = ["char_count", "word_count", "line_count", "token_count"]


def _collect_rows(
    corpus_path: Path,
) -> list[dict[str, str | int]]:
    records = load_corpus(corpus_path)
    rows: list[dict[str, str | int]] = []
    for rec in records:
        try:
            content = load_content(corpus_path, rec)
        except FileNotFoundError:
            logger.warning("Missing SKILL.md for %s — skipped", rec.key)
            continue
        rows.append(
            {
                "key": rec.key,
                "name": rec.name,
                "source": rec.source,
                "char_count": len(content),
                "word_count": len(content.split()),
                "line_count": content.count("\n") + 1,
                "token_count": _token_count(content),
            }
        )
    return rows


# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------


def _print_summary(rows: list[dict[str, str | int]]) -> None:
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n{'=' * 64}")
    print(f"Skill length summary  (n={len(rows):,})")
    print(f"{'=' * 64}")
    for metric in _METRIC_KEYS:
        vals = sorted(int(r[metric]) for r in rows)
        n = len(vals)
        print(f"\n--- {metric} ---")
        print(f"  count  : {n:>10,}")
        print(f"  mean   : {statistics.mean(vals):>10,.1f}")
        print(f"  median : {statistics.median(vals):>10,.1f}")
        print(f"  std    : {statistics.stdev(vals):>10,.1f}")
        print(f"  min    : {vals[0]:>10,}")
        print(f"  max    : {vals[-1]:>10,}")
        for p in percentiles:
            idx = int(n * p / 100)
            idx = min(idx, n - 1)
            print(f"  p{p:<3}   : {vals[idx]:>10,}")


# ------------------------------------------------------------------
# CSV export
# ------------------------------------------------------------------


def _save_csv(rows: list[dict[str, str | int]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------


def _plot_histogram(values: list[int], title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(values, bins=100, edgecolor="black", linewidth=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")


def _plot_cdf(values: list[int], path: Path) -> None:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cdf = [(i + 1) / n for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_vals, cdf, linewidth=1.2)
    for t in _CDF_THRESHOLDS:
        ax.axvline(t, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(t, 0.02, f"{t:,}", rotation=90, fontsize=8, color="red")
    ax.set_title("Token count CDF")
    ax.set_xlabel("Tokens (cl100k_base)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {path}")


def _generate_plots(rows: list[dict[str, str | int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chars = [int(r["char_count"]) for r in rows]
    tokens = [int(r["token_count"]) for r in rows]

    _plot_histogram(
        chars,
        "Character count distribution",
        "Characters",
        output_dir / "char_distribution.png",
    )
    _plot_histogram(
        tokens,
        "Token count distribution",
        "Tokens (cl100k_base)",
        output_dir / "token_distribution.png",
    )
    _plot_cdf(tokens, output_dir / "length_cdf.png")


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skill file length EDA",
    )
    parser.add_argument("--corpus-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    corpus_path = args.corpus_path
    if corpus_path is None:
        cfg = load_config()
        corpus_path = Path(cfg.index.input_corpus_path)
    corpus_path = corpus_path.resolve()

    rows = _collect_rows(corpus_path)
    if not rows:
        print("No skills found — check corpus path.")
        return

    _print_summary(rows)
    _save_csv(rows, args.output_dir / "skill_lengths.csv")
    _generate_plots(rows, args.output_dir)


if __name__ == "__main__":
    main()

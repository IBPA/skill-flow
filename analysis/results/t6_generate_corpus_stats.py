"""Generate the skill corpus collection statistics table (tab:corpus_stats).

Reads crawler metadata (index.json + sync_state.json) and renders a LaTeX
table with total-processed, excluded, failed, and indexed counts.

Usage::

    uv run python -m analysis.results.t6_generate_corpus_stats

    uv run python -m analysis.results.t6_generate_corpus_stats \
        --crawler-dir /path/to/skill-crawler \
        --output paper/tables/6_corpus_stats.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analysis.results.utils.latex_utils import write_or_print

# Crawler metadata now lives in this repo (data/skills/_metadata) after the
# migration away from the standalone skill-crawler checkout.
_DEFAULT_CRAWLER_DIR = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def _load_metadata(crawler_dir: Path) -> tuple[dict, dict]:
    """Load index.json and sync_state.json from the crawler metadata."""
    meta_dir = crawler_dir / "data" / "skills" / "_metadata"
    index = json.loads((meta_dir / "index.json").read_text())
    sync_state = json.loads((meta_dir / "sync_state.json").read_text())
    return index, sync_state


def _count_indexed(corpus_dir: Path) -> int:
    """Return the number of downloaded skills that carry a top-level SKILL.md.

    Downloading a skill's repository does not guarantee it contains a
    ``SKILL.md``; only those that do are retained and embedded into the
    retrieval index, so this is smaller than the downloaded count.
    """
    if not corpus_dir.is_dir():
        return 0
    return sum(
        1 for d in corpus_dir.iterdir() if d.is_dir() and (d / "SKILL.md").is_file()
    )


def _compute_counts(index: dict, sync_state: dict, corpus_dir: Path) -> dict[str, int]:
    """Return total-processed, skipped, failed, downloaded, indexed counts."""
    src = sync_state["sources"]["skillsmp"]
    skipped = len(src.get("skipped_skills", []))
    failed = len(src.get("failed_skills", []))
    downloaded = len(index["skills"])
    return {
        "total_processed": downloaded + skipped + failed,
        "skipped": skipped,
        "failed": failed,
        "downloaded": downloaded,
        "indexed": _count_indexed(corpus_dir),
    }


# ------------------------------------------------------------------
# Table rendering
# ------------------------------------------------------------------


def render_table(counts: dict[str, int]) -> list[str]:
    """Return LaTeX tabular content for tab:corpus_stats (no table wrapper)."""
    rows = [
        ("Skills processed", counts["total_processed"]),
        ("Excluded (repo $>$ 50\\,MB)", counts["skipped"]),
        ("Failed (deleted/inaccessible)", counts["failed"]),
        ("Downloaded", counts["downloaded"]),
        ("Indexed (valid SKILL.md)", counts["indexed"]),
    ]

    lines: list[str] = [
        r"\begin{tabular}{lr}",
        r"  \toprule",
        r"  \textbf{Metric} & \textbf{Count} \\",
        r"  \midrule",
    ]
    for label, count in rows:
        lines.append(f"  {label:<29} & {count:,} \\\\")
    lines.extend(
        [
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )
    return lines


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate tab:corpus_stats skill corpus statistics table",
    )
    parser.add_argument(
        "--crawler-dir",
        type=Path,
        default=_DEFAULT_CRAWLER_DIR,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/skills/skillsmp"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/tables/6_corpus_stats.tex"),
    )
    args = parser.parse_args()

    index, sync_state = _load_metadata(args.crawler_dir)
    counts = _compute_counts(index, sync_state, args.corpus_dir)
    write_or_print(render_table(counts), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

_DEFAULT_CRAWLER_DIR = Path(__file__).resolve().parents[2] / ".." / "skill-crawler"


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def _load_metadata(crawler_dir: Path) -> tuple[dict, dict]:
    """Load index.json and sync_state.json from the crawler metadata."""
    meta_dir = crawler_dir / "data" / "skills" / "_metadata"
    index = json.loads((meta_dir / "index.json").read_text())
    sync_state = json.loads((meta_dir / "sync_state.json").read_text())
    return index, sync_state


def _compute_counts(index: dict, sync_state: dict) -> dict[str, int]:
    """Return total-processed, skipped, failed, indexed counts."""
    src = sync_state["sources"]["skillsmp"]
    skipped = len(src.get("skipped_skills", []))
    failed = len(src.get("failed_skills", []))
    indexed = len(index["skills"])
    return {
        "total_processed": indexed + skipped + failed,
        "skipped": skipped,
        "failed": failed,
        "indexed": indexed,
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
        ("Downloaded \\& indexed", counts["indexed"]),
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
        "--output",
        type=Path,
        default=Path("paper/tables/6_corpus_stats.tex"),
    )
    args = parser.parse_args()

    index, sync_state = _load_metadata(args.crawler_dir)
    counts = _compute_counts(index, sync_state)
    write_or_print(render_table(counts), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

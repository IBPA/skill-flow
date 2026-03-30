"""Generate the query generation examples table (tab:query_examples).

Loads cached query generation outputs for a specific task and renders a
LaTeX table showing M=1 and M=5 query variants.  The table references
the ``\\courtinstruction`` macro defined in ``main.tex``.

Usage::

    uv run python -m analysis.results.t7_generate_query_examples

    uv run python -m analysis.results.t7_generate_query_examples \
        --task-id court-form-filling \
        --cache path/to/retriever_query_gen_cache.json \
        --output paper/tables/7_query_examples.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analysis.results.utils.latex_utils import write_or_print

_DEFAULT_M1_CACHE = Path(
    "outputs/pipeline/skillsbench/specificity-v2.0/query_gen_cache.json",
)
_DEFAULT_M5_CACHE = Path(
    "outputs/experiments/querygen-v2-cache/query_gen_cache_5.json",
)
_DEFAULT_TASK = "court-form-filling"
_DEFAULT_OUTPUT = Path("paper/tables/7_query_examples.tex")


def _escape_latex(text: str) -> str:
    """Escape common LaTeX special characters in query text."""
    replacements = [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def load_single_query(cache_path: Path, task_id: str) -> str:
    """Load a single generated query for a task from the M=1 cache."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    value = data[task_id]
    if isinstance(value, str):
        # Strip stray outer quotes from cached LLM output
        return value.strip('"')
    return str(value[0]).strip('"')


def load_multi_queries(cache_path: Path, task_id: str) -> list[str]:
    """Load multiple generated queries for a task from the M=5 cache."""
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    value = data[task_id]
    if isinstance(value, list):
        return [str(q).strip('"') for q in value]
    return [str(value).strip('"')]


def render_table(
    m1_query: str,
    m5_queries: list[str],
    *,
    macro: str = r"\courtinstruction",
) -> list[str]:
    """Return LaTeX tabular content for tab:query_examples (no table wrapper).

    Parameters
    ----------
    m1_query:
        Single generated query for the M=1 row.
    m5_queries:
        List of generated queries for the M=5 row.
    macro:
        LaTeX macro name for the task instruction (displayed in the
        header row).
    """
    m1_escaped = _escape_latex(m1_query)

    # Format M=5 row with numbered queries separated by \newline
    m5_parts: list[str] = []
    for i, q in enumerate(m5_queries, 1):
        m5_parts.append(f"{i}.~{_escape_latex(q)}")
    m5_text = r" \newline ".join(m5_parts)

    lines: list[str] = [
        r"\begin{tabular}{p{1.5cm}p{11cm}}",
        r"  \toprule",
        f"  \\multicolumn{{2}}{{p{{12.5cm}}}}{{{macro}}} \\\\",
        r"  \midrule",
        f"  $M = 1$ & {m1_escaped} \\\\",
        r"  \midrule",
        f"  $M = 5$ & {m5_text} \\\\",
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    return lines


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate tab:query_examples query generation table",
    )
    parser.add_argument(
        "--task-id",
        default=_DEFAULT_TASK,
        help="Task ID to show queries for (default: %(default)s)",
    )
    parser.add_argument(
        "--m1-cache",
        type=Path,
        default=_DEFAULT_M1_CACHE,
        help="Path to M=1 (single-query) cache JSON",
    )
    parser.add_argument(
        "--m5-cache",
        type=Path,
        default=_DEFAULT_M5_CACHE,
        help="Path to M=5 (multi-query) cache JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    m1 = load_single_query(args.m1_cache, args.task_id)
    m5 = load_multi_queries(args.m5_cache, args.task_id)
    table_lines = render_table(m1, m5)
    write_or_print(table_lines, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

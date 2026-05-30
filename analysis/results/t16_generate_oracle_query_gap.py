"""Classify each SkillsBench oracle as in-query / out-of-query and bucket by rank.

For each task in a reranker report, every oracle skill is either:

* **in-query**: its description shares at least one content word with the
  rerank query (lowercase alphanumeric token, length >= 3, not a stopword).
* **out-of-query**: it does not.

Out-of-query oracles are typically utility / implementation skills the user's
task description never names (e.g. ``civ6-adjacency-optimizer`` needs
``sqlite-map-parser``).

Usage::

    uv run python -m analysis.results.t16_generate_oracle_query_gap

    uv run python -m analysis.results.t16_generate_oracle_query_gap \\
        --report outputs/experiments/reranker-comparison/\\
baai-bge-reranker-v2-m3-512chars-q1-max.json \\
        --tasks-dir integration/skillsbench/tasks \\
        --out outputs/analysis/oracle_query_gap.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skill_flow.eval.utils.ground_truth import load_ground_truth

if TYPE_CHECKING:
    from collections.abc import Callable

    from skill_flow.eval.models import InjectedSkill

logger = logging.getLogger(__name__)

# Conservative English stopword list — kept inline so this script has no extra
# dependency. Tokens are lowercased and length >= 3 already, which removes
# most short function words; this set covers the residue that survives.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "has",
        "are",
        "was",
        "were",
        "but",
        "not",
        "use",
        "using",
        "used",
        "any",
        "all",
        "can",
        "will",
        "would",
        "should",
        "may",
        "such",
        "you",
        "your",
        "into",
        "onto",
        "out",
        "via",
        "per",
        "based",
        "between",
        "across",
        "each",
        "their",
        "them",
        "they",
        "its",
        "where",
        "when",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "than",
        "then",
        "also",
        "more",
        "most",
        "some",
        "other",
        "another",
        "given",
        "either",
        "both",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_MIN_TOKEN_LEN = 3


@dataclass(frozen=True)
class OracleRow:
    task_id: str
    skill_name: str
    rank: int | None  # 1-indexed; None means not in top-K returned
    in_query: bool
    description: str


def _content_words(text: str) -> set[str]:
    """Return the set of content tokens in *text*."""
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text)
        if len(t) >= _MIN_TOKEN_LEN and t.lower() not in _STOPWORDS
    }


def _classify_task(
    task_id: str,
    rerank_query: str,
    retrieved: list[dict[str, object]],
    expected_keys: list[str],
    injected_by_key: dict[str, InjectedSkill],
) -> list[OracleRow]:
    """For one task, return one OracleRow per expected oracle."""
    rank_by_key: dict[str, int] = {}
    desc_by_key: dict[str, str] = {}
    for i, r in enumerate(retrieved, start=1):
        key = str(r["key"])
        rank_by_key[key] = i
        desc_by_key[key] = str(r.get("description", ""))
    query_words = _content_words(rerank_query)
    rows: list[OracleRow] = []
    for key in expected_keys:
        description = desc_by_key.get(key) or injected_by_key[key].description
        in_query = bool(_content_words(description) & query_words)
        rows.append(
            OracleRow(
                task_id=task_id,
                skill_name=key.split("/", 2)[-1],
                rank=rank_by_key.get(key),
                in_query=in_query,
                description=description,
            )
        )
    return rows


def collect(
    report_path: Path,
    tasks_dir: Path,
) -> list[OracleRow]:
    """Compute one row per (task, oracle) pair across the whole report."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tasks, injected, _ = load_ground_truth(tasks_dir)
    injected_by_key: dict[str, InjectedSkill] = {s.key: s for s in injected}
    expected_by_task: dict[str, list[str]] = {
        t.task_id: list(t.ground_truth_keys) for t in tasks
    }

    rows: list[OracleRow] = []
    task_results = report["task_results"]
    iterable = task_results if isinstance(task_results, list) else task_results.values()
    for tr in iterable:
        task_id = str(tr["task_id"])
        if task_id not in expected_by_task:
            continue
        rerank_query = str(tr.get("rerank_query") or tr.get("retrieval_query") or "")
        rows.extend(
            _classify_task(
                task_id,
                rerank_query,
                list(tr["retrieved_skills"]),
                expected_by_task[task_id],
                injected_by_key,
            )
        )
    return rows


# Mutually exclusive rank buckets that partition every oracle exactly once.
_PARTITION_BUCKETS: list[tuple[str, Callable[[OracleRow], bool]]] = [
    ("Top-5 (rank 1-5)", lambda r: r.rank is not None and r.rank <= 5),
    ("Top-100 (rank 6-100)", lambda r: r.rank is not None and 5 < r.rank <= 100),
    (
        "Worst-ranked (rank > 100 or missing)",
        lambda r: r.rank is None or r.rank > 100,
    ),
]

# Reported separately as a subset of the worst-ranked bucket, not a partition
# cell -- listing it as a fourth row would make the bucket counts appear to sum
# to more than the oracle total.
_SUBSET_BUCKET: tuple[str, Callable[[OracleRow], bool]] = (
    "Missing from top-1000",
    lambda r: r.rank is None or r.rank > 1000,
)


def _bucket_stat(
    name: str, pred: Callable[[OracleRow], bool], rows: list[OracleRow]
) -> dict[str, object]:
    """Out-of-query rate for the oracles matching *pred*."""
    bucket = [r for r in rows if pred(r)]
    n = len(bucket)
    if n == 0:
        return {"bucket": name, "n_oracles": 0, "pct_out_of_query": 0.0}
    n_out = sum(1 for r in bucket if not r.in_query)
    return {
        "bucket": name,
        "n_oracles": n,
        "pct_out_of_query": round(100.0 * n_out / n, 1),
    }


def bucket_summary(rows: list[OracleRow]) -> list[dict[str, object]]:
    """The three mutually exclusive rank buckets reported in the paper."""
    return [_bucket_stat(name, pred, rows) for name, pred in _PARTITION_BUCKETS]


def subset_summary(rows: list[OracleRow]) -> dict[str, object]:
    """Missing-from-top-1000 subset of the worst-ranked bucket."""
    name, pred = _SUBSET_BUCKET
    return _bucket_stat(name, pred, rows)


def example_oracles(rows: list[OracleRow], k: int = 5) -> list[dict[str, object]]:
    """Pick *k* worst-ranked out-of-query oracles as illustrative examples."""
    out_of_query = [r for r in rows if not r.in_query]
    out_of_query.sort(
        key=lambda r: (r.rank is None, r.rank if r.rank is not None else 10**9),
        reverse=True,
    )
    return [
        {
            "task_id": r.task_id,
            "oracle": r.skill_name,
            "rank": r.rank,
            "description": r.description,
        }
        for r in out_of_query[:k]
    ]


def render_markdown_table(summary: list[dict[str, object]]) -> str:
    lines = [
        "| Rank bucket | # oracles | Out-of-query |",
        "| --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['bucket']} | {row['n_oracles']} | {row['pct_out_of_query']}% |"
        )
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report",
        type=Path,
        default=Path(
            "outputs/experiments/reranker-comparison/"
            "baai-bge-reranker-v2-m3-512chars-q1-max.json"
        ),
    )
    ap.add_argument(
        "--tasks-dir", type=Path, default=Path("integration/skillsbench/tasks")
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/oracle_query_gap.json"),
    )
    ap.add_argument("--n-examples", type=int, default=5)
    args = ap.parse_args()

    rows = collect(args.report, args.tasks_dir)
    summary = bucket_summary(rows)
    subset = subset_summary(rows)
    examples = example_oracles(rows, k=args.n_examples)

    total = len(rows)  # the three buckets partition every oracle exactly once
    print(f"Report: {args.report}")
    print(f"Total oracle-task pairs analyzed: {len(rows)}")
    print()
    print(render_markdown_table(summary))
    print()
    print(
        f"The three buckets partition all {total} oracles. "
        f"Within the worst-ranked bucket, the {subset['n_oracles']} oracles "
        f"missing from the top-1000 are {subset['pct_out_of_query']}% out-of-query."
    )
    print()
    print(f"Top-{args.n_examples} worst-ranked out-of-query oracles:")
    for ex in examples:
        rank_str = "missing" if ex["rank"] is None else f"rank {ex['rank']}"
        print(f"  - {ex['task_id']:35s}  needs {ex['oracle']:35s}  ({rank_str})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "report": str(args.report),
                "n_rows": len(rows),
                "summary": summary,
                "subset": subset,
                "examples": examples,
                "rows": [r.__dict__ for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

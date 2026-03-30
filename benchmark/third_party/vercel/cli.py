"""CLI for searching skills.sh and downloading skills for terminal-bench tasks.

Usage::

    uv run python -m analysis.vercel_skills.cli \\
      --query-source cache \\
      --query-cache outputs/pipeline/.../query_gen_cache.json \\
      --output-dir outputs/vercel-skills/vercel-top1 \\
      [--max-tasks 3] [--top-k 1] [--delay 0.5]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from benchmark.third_party.vercel.download import download_skills_batch
from benchmark.third_party.vercel.search import search_skills_sh

logger = logging.getLogger(__name__)


def _load_queries(args: argparse.Namespace) -> dict[str, str]:
    """Return ``{task_name: query}`` based on the chosen query source."""
    if args.query_source == "cache":
        cache_path = Path(args.query_cache)
        raw: dict[str, Any] = json.loads(cache_path.read_text("utf-8"))
        # The cache maps task_id -> query string (or list of queries).
        # Use the first query if it's a list.
        queries: dict[str, str] = {}
        for task_id, value in raw.items():
            if isinstance(value, list):
                queries[task_id] = value[0] if value else task_id
            else:
                queries[task_id] = str(value)
        return queries

    if args.query_source == "name":
        # Use cache keys but convert slug to words
        cache_path = Path(args.query_cache)
        raw = json.loads(cache_path.read_text("utf-8"))
        return {tid: tid.replace("-", " ") for tid in raw}

    if args.query_source == "instruction":
        from skill_flow.pipeline.models import load_task_instructions  # noqa: PLC0415

        tasks_dir = Path(args.tasks_dir)
        instructions = load_task_instructions(
            tasks_dir, max_query_chars=args.max_query_chars
        )
        return {t.task_id: t.query for t in instructions}

    msg = f"Unknown query source: {args.query_source}"
    raise ValueError(msg)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Search skills.sh and download skills for terminal-bench"
    )
    parser.add_argument(
        "--query-source",
        choices=["cache", "name", "instruction"],
        default="cache",
        help="How to generate search queries (default: cache)",
    )
    parser.add_argument(
        "--query-cache",
        default="outputs/pipeline/terminal-bench/20260315-selector-v4.0-top1/query_gen_cache.json",
        help="Path to query_gen_cache.json (for cache/name sources)",
    )
    parser.add_argument(
        "--tasks-dir",
        help="Path to tasks directory (for instruction source)",
    )
    parser.add_argument(
        "--max-query-chars",
        type=int,
        default=200,
        help="Max chars for instruction queries (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save downloaded skills",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="Limit tasks (0=all)")
    parser.add_argument("--top-k", type=int, default=1, help="Skills per task")
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Delay between API calls"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    queries = _load_queries(args)
    if args.max_tasks > 0:
        queries = dict(list(queries.items())[: args.max_tasks])

    logger.info("Searching skills.sh for %d tasks", len(queries))

    report: dict[str, Any] = {}
    skills_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for task_name, query in queries.items():
        results = search_skills_sh(query, top_k=args.top_k)
        if not results:
            logger.warning("No results for task %s (query: %s)", task_name, query)
            report[task_name] = {"query": query, "status": "no_results"}
            continue

        skill = results[0]
        source = skill.get("source", "")
        skill_id = skill.get("skillId", "")
        report[task_name] = {
            "query": query,
            "skill_id": skill_id,
            "source": source,
            "name": skill.get("name", ""),
            "installs": skill.get("installs", 0),
            "status": "pending",
        }
        skills_by_source[source].append((skill_id, task_name))

        if args.delay > 0:
            time.sleep(args.delay)

    # Download all skills
    download_results = download_skills_batch(skills_by_source, output_dir)
    for task_name, ok in download_results.items():
        if task_name in report:
            report[task_name]["status"] = "ok" if ok else "download_failed"

    # Write report
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Summary
    statuses = [r.get("status", "unknown") for r in report.values()]
    logger.info(
        "Done: %d ok, %d no_results, %d download_failed, %d total",
        statuses.count("ok"),
        statuses.count("no_results"),
        statuses.count("download_failed"),
        len(report),
    )
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    main()

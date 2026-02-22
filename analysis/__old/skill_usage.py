#!/usr/bin/env python3
"""Analyze skill usage in Harbor evaluation rollout files.

This script scans rollout JSONL files to detect whether skills were actually
used (read/accessed) by the agent during task execution, beyond just being
listed in the initial instructions.

Usage:
    poetry run python -m analysis.skill_usage <evaluation_dir>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from analysis.shared import (
    SkillReadRecord,
    analyze_rollout,
    find_rollout_files,
)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze skill usage in Harbor evaluation rollout files"
    )
    parser.add_argument(
        "evaluation_dir",
        type=Path,
        help="Path to evaluation directory (e.g., outputs/evaluation/tb-skills)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed skill usage for each task",
    )
    parser.add_argument(
        "--output-tsv",
        "-o",
        type=Path,
        help="Output matched skill content to TSV file",
    )
    parser.add_argument(
        "--content-limit",
        type=int,
        default=500,
        help="Max characters of skill content to capture (default: 500)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    return parser.parse_args()


def _print_verbose_result(result: dict) -> None:
    """Print verbose output for a single task result."""
    task = result["task"].split("__")[0]
    if result.get("skills_used"):
        print(f"+ {task}: {len(result['skills_used'])} skill(s) used")
        for skill, accesses in result["skills_used"].items():
            print(f"    - {skill}: {', '.join(accesses)}")
    else:
        print(f"- {task}: no skills used")


def _print_summary(
    results: list[dict], tasks_with_skill_use: int, all_skills_used: dict[str, int]
) -> None:
    """Print summary of skill usage analysis."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tasks analyzed: {len(results)}")
    pct = 100 * tasks_with_skill_use / len(results) if results else 0
    print(f"Tasks with skill usage: {tasks_with_skill_use} ({pct:.1f}%)")
    print(f"Tasks without skill usage: {len(results) - tasks_with_skill_use}")

    if all_skills_used:
        print("\nSkills used (by frequency):")
        for skill, count in sorted(all_skills_used.items(), key=lambda x: -x[1]):
            print(f"  {skill}: {count} task(s)")
    else:
        print("\nNo skills were used in any task.")

    _print_available_skills(results)


def _print_available_skills(results: list[dict]) -> None:
    """Print available skills from first result."""
    if not results or not results[0].get("available_skills"):
        return

    first_result = results[0]
    print(f"\nAvailable skills ({first_result['num_available']}):")
    for skill in first_result["available_skills"][:10]:
        print(f"  - {skill}")
    if first_result["num_available"] > 10:
        print(f"  ... and {first_result['num_available'] - 10} more")


def _export_skill_reads_tsv(records: list[SkillReadRecord], tsv_path: Path) -> None:
    """Export skill read records to a TSV file."""
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("task_name\tskill_name\tline_num\tcommand\tcontent_preview\n")

        for record in records:
            task = record.task_name.split("__")[0]
            command = record.command.replace("\t", " ").replace("\n", "\\n")
            content = record.content_preview.replace("\t", " ").replace("\n", "\\n")
            f.write(
                f"{task}\t{record.skill_name}\t{record.line_num}\t{command}\t{content}\n"
            )

    print(f"\nExported {len(records)} skill reads to {tsv_path}")


def main() -> int:
    args = _parse_args()

    if not args.evaluation_dir.exists():
        print(f"Error: Directory not found: {args.evaluation_dir}")
        return 1

    rollout_files = find_rollout_files(args.evaluation_dir)
    if not rollout_files:
        print(f"No rollout files found in {args.evaluation_dir}")
        return 1

    print(f"Found {len(rollout_files)} rollout files\n")

    results = []
    all_records: list[SkillReadRecord] = []
    tasks_with_skill_use = 0
    all_skills_used: dict[str, int] = defaultdict(int)

    for rollout_path in sorted(rollout_files):
        result, records = analyze_rollout(rollout_path, args.content_limit)
        results.append(result)
        all_records.extend(records)

        if result.get("skills_used"):
            tasks_with_skill_use += 1
            for skill_name in result["skills_used"]:
                all_skills_used[skill_name] += 1

        if args.verbose:
            _print_verbose_result(result)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    _print_summary(results, tasks_with_skill_use, all_skills_used)

    if args.output_tsv and all_records:
        _export_skill_reads_tsv(all_records, args.output_tsv)

    return 0


if __name__ == "__main__":
    sys.exit(main())

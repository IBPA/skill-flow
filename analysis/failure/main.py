#!/usr/bin/env python3
"""Analyze agent trajectories and generate human-readable failure summaries.

This script processes evaluation job directories and uses an LLM to generate
concise summaries of what the agent tried, why it failed/succeeded, and whether
a skill could help.

Usage:
    uv run python -m analysis.failure.main \
        --job-dir outputs/evaluation/tb-baseline-20260211-131435 \
        --output-dir outputs/analysis/tb-baseline-20260211-131435
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from analysis.failure.core import (
    TaskData,
    compress_trajectory,
    extract_amenability,
    extract_one_liner,
    format_test_results,
    list_task_dirs,
    load_task_data,
)

load_dotenv()


def generate_summary(client: OpenAI, task_data: TaskData, model: str) -> str:
    """Generate LLM summary for a task."""
    compressed_traj = compress_trajectory(task_data.trajectory_steps)
    test_summary = format_test_results(task_data.test_results)
    outcome = "SUCCESS" if task_data.reward > 0 else "FAILED"
    why_label = "Succeeded" if task_data.reward > 0 else "Failed"

    prompt = f"""Analyze this agent evaluation task and provide a concise summary.

## Task Metadata
- Name: {task_data.task_name}
- Outcome: {outcome}
- Tokens used: {task_data.n_input_tokens + task_data.n_output_tokens:,}
- Steps taken: {task_data.n_steps}

## Original Task Prompt
{task_data.task_prompt or "(not captured)"}

## Agent Trajectory (key steps)
{compressed_traj}

## Test Results
{test_summary}

Please provide your analysis in this exact format:

### Task Description
What was the agent asked to do? (1-2 sentences)

### What the Agent Did
Key actions taken (3-5 bullet points)

### Why It {why_label}
Root cause analysis (2-3 sentences)

### Skill Amenability
Would a procedural skill help? Answer HIGH, MEDIUM, or LOW with a brief reason.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1000,
    )

    return response.choices[0].message.content or ""


def write_task_summary(output_dir: Path, task_data: TaskData, summary: str) -> None:
    """Write task summary to markdown file."""
    summaries_dir = output_dir / "task-summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    outcome = "SUCCESS" if task_data.reward > 0 else "FAILED"
    total_tokens = task_data.n_input_tokens + task_data.n_output_tokens

    content = f"""# Task: {task_data.task_name}

## Outcome: {outcome}

{summary}

---
*Trial: {task_data.trial_name}*
*Tokens: {total_tokens:,}*
"""

    output_file = summaries_dir / f"{task_data.task_name}.md"
    output_file.write_text(content)


def generate_aggregate_report(
    output_dir: Path, task_summaries: list[tuple[TaskData, str]]
) -> None:
    """Generate aggregate summary report."""
    high_amenability: list[str] = []
    medium_amenability: list[str] = []
    low_amenability: list[str] = []
    successes: list[str] = []
    failures: list[str] = []

    for task_data, summary in task_summaries:
        one_liner = extract_one_liner(summary)
        amenability = extract_amenability(summary)
        entry = f"- **{task_data.task_name}**: {one_liner}"

        if task_data.reward > 0:
            successes.append(entry)
        else:
            failures.append(entry)

        if amenability == "HIGH":
            high_amenability.append(task_data.task_name)
        elif amenability == "MEDIUM":
            medium_amenability.append(task_data.task_name)
        else:
            low_amenability.append(task_data.task_name)

    n_success = len(successes)
    n_failed = len(failures)
    n_total = n_success + n_failed

    def fmt_amenability(label: str, items: list[str]) -> str:
        preview = ", ".join(items[:10])
        suffix = "..." if len(items) > 10 else ""
        return f"- **{label}** ({len(items)} tasks): {preview}{suffix}"

    high_line = fmt_amenability("HIGH", high_amenability)
    medium_line = fmt_amenability("MEDIUM", medium_amenability)
    low_line = fmt_amenability("LOW", low_amenability)
    failed_section = chr(10).join(failures) if failures else "None"
    success_section = chr(10).join(successes) if successes else "None"

    content = f"""# Trajectory Analysis Summary

## Overview
- **Total tasks**: {n_total}
- **Passed**: {n_success} ({100 * n_success / n_total:.1f}%)
- **Failed**: {n_failed} ({100 * n_failed / n_total:.1f}%)

## Skill Amenability Distribution
{high_line}
{medium_line}
{low_line}

## Failed Tasks
{failed_section}

## Successful Tasks
{success_section}
"""

    output_file = output_dir / "summary.md"
    output_file.write_text(content)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze agent trajectories and generate failure summaries"
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Path to evaluation job directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to output directory for analysis results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="LLM model to use for summarization (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tasks to process (0 = all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks that already have summaries",
    )
    args = parser.parse_args()

    if not args.job_dir.exists():
        print(f"Error: Job directory not found: {args.job_dir}", file=sys.stderr)
        return 1

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        return 1

    client = OpenAI(api_key=api_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = list_task_dirs(args.job_dir)
    if args.limit > 0:
        task_dirs = task_dirs[: args.limit]

    print(f"Processing {len(task_dirs)} tasks from {args.job_dir.name}")

    task_summaries: list[tuple[TaskData, str]] = []
    summaries_dir = args.output_dir / "task-summaries"

    for i, task_dir in enumerate(task_dirs, 1):
        task_name = task_dir.name.split("__")[0]
        existing_file = summaries_dir / f"{task_name}.md"

        if args.skip_existing and existing_file.exists():
            print(f"[{i}/{len(task_dirs)}] Skipping {task_name} (exists)")
            task_data = load_task_data(task_dir)
            if task_data:
                summary = existing_file.read_text()
                task_summaries.append((task_data, summary))
            continue

        print(f"[{i}/{len(task_dirs)}] Processing {task_name}...", end=" ", flush=True)

        task_data = load_task_data(task_dir)
        if task_data is None:
            print("SKIP (missing data)")
            continue

        summary = generate_summary(client, task_data, args.model)
        write_task_summary(args.output_dir, task_data, summary)
        task_summaries.append((task_data, summary))

        outcome = "PASS" if task_data.reward > 0 else "FAIL"
        print(outcome)

    generate_aggregate_report(args.output_dir, task_summaries)
    print(f"\nAnalysis complete. Output written to {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

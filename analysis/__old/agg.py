#!/usr/bin/env python3
"""Aggregate metrics for evaluation jobs.

Subcommands:
    job     - Aggregate metrics for a single evaluation job
    prefix  - Aggregate metrics across multiple jobs matching a prefix

Usage:
    poetry run python -m analysis.agg job <eval_dir> [--json]
    poetry run python -m analysis.agg prefix -p <prefix> [--json] [--list-jobs]
    poetry run python -m analysis.agg prefix -p <prefix> --distribution
"""

import argparse
import json
import sys
from pathlib import Path

from analysis.aggregate import aggregate_runs, find_runs_by_prefix
from analysis.shared import (
    AggregatedMetrics,
    aggregate_job_metrics,
    aggregated_metrics_to_dict,
    find_jobs_by_prefix,
    job_metrics_to_dict,
    load_job_metrics,
    print_aggregated_metrics,
    print_distribution,
    print_job_metrics,
)

# =============================================================================
# Job Subcommand
# =============================================================================


def cmd_job(args: argparse.Namespace) -> int:
    """Handle 'job' subcommand."""
    if not args.eval_dir.exists():
        print(f"Error: Directory not found: {args.eval_dir}", file=sys.stderr)
        return 1

    metrics = load_job_metrics(args.eval_dir)

    if metrics.n_tasks == 0:
        print(f"Error: No tasks found in {args.eval_dir}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(job_metrics_to_dict(metrics), indent=2))
    else:
        print_job_metrics(metrics)

    return 0


# =============================================================================
# Prefix Subcommand
# =============================================================================


def load_prefix_metrics(eval_root: Path, prefix: str) -> AggregatedMetrics | None:
    """Load and aggregate metrics for all jobs matching prefix.

    Args:
        eval_root: Root directory containing evaluation outputs
        prefix: Prefix to match job directories

    Returns:
        AggregatedMetrics or None if no matching jobs found
    """
    job_dirs = find_jobs_by_prefix(eval_root, prefix)
    if not job_dirs:
        return None

    job_metrics_list = [load_job_metrics(d) for d in job_dirs]
    # Filter out empty jobs
    job_metrics_list = [j for j in job_metrics_list if j.n_tasks > 0]

    if not job_metrics_list:
        return None

    return aggregate_job_metrics(job_metrics_list, name=prefix)


def cmd_prefix(args: argparse.Namespace) -> int:
    """Handle 'prefix' subcommand."""
    eval_root = Path(args.eval_dir)
    if not eval_root.exists():
        print(f"Error: Eval directory not found: {eval_root}", file=sys.stderr)
        return 1

    job_dirs = find_jobs_by_prefix(eval_root, args.prefix)
    if not job_dirs:
        print(
            f"Error: No jobs matching '{args.prefix}*' in {eval_root}", file=sys.stderr
        )
        return 1

    # Distribution mode: show task success distribution
    if args.distribution:
        run_dirs = find_runs_by_prefix(eval_root, args.prefix)
        if not run_dirs:
            print(f"Error: No runs for '{args.prefix}*'", file=sys.stderr)
            return 1
        task_results = aggregate_runs(run_dirs)
        print_distribution(task_results, args.prefix)
        return 0

    # Default: show aggregated metrics
    metrics = load_prefix_metrics(eval_root, args.prefix)
    if metrics is None:
        print(f"Error: No valid job data for '{args.prefix}*'", file=sys.stderr)
        return 1

    if args.json:
        output = aggregated_metrics_to_dict(metrics)
        if args.list_jobs:
            output["jobs"] = [d.name for d in job_dirs]
        print(json.dumps(output, indent=2))
    else:
        if args.list_jobs:
            print(f"Jobs ({len(job_dirs)}):")
            for d in job_dirs:
                print(f"  - {d.name}")
            print()
        print_aggregated_metrics(metrics)

    return 0


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics for evaluation jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Job subcommand
    job_parser = subparsers.add_parser(
        "job", help="Aggregate metrics for a single evaluation job"
    )
    job_parser.add_argument(
        "eval_dir", type=Path, help="Path to evaluation job directory"
    )
    job_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Prefix subcommand
    prefix_parser = subparsers.add_parser(
        "prefix", help="Aggregate metrics across jobs matching a prefix"
    )
    prefix_parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        required=True,
        help="Prefix to match job directories (e.g., 'tb-baseline')",
    )
    prefix_parser.add_argument(
        "--eval-dir",
        type=str,
        default="outputs/evaluation",
        help="Root directory for evaluation outputs (default: outputs/evaluation)",
    )
    prefix_parser.add_argument("--json", action="store_true", help="Output as JSON")
    prefix_parser.add_argument(
        "--list-jobs", action="store_true", help="Also list individual job names"
    )
    prefix_parser.add_argument(
        "--distribution",
        action="store_true",
        help="Show task success distribution instead of metrics",
    )

    args = parser.parse_args()

    if args.command == "job":
        return cmd_job(args)
    elif args.command == "prefix":
        return cmd_prefix(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

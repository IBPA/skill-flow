#!/usr/bin/env python3
"""Compare aggregate results between baseline and skill-augmented runs.

By default, performs full statistical comparison with skill analysis.
Use --simple for a basic metrics comparison table only.
Use --distribution for task success distribution comparison.

Usage:
    poetry run python -m analysis.compare -b tb-baseline -s tb-skill-v3
    poetry run python -m analysis.compare -b tb-baseline -s tb-skill-v3 --simple
    poetry run python -m analysis.compare -b tb-baseline -s tb-skill-v3 --distribution
    poetry run python -m analysis.compare -b tb-baseline -s tb-skill-v3 --json
"""

import argparse
import json
import sys
from pathlib import Path

from analysis.agg import load_prefix_metrics
from analysis.aggregate import (
    aggregate_metrics,
    aggregate_runs,
    analyze_skill_usage_aggregate,
    compare_all_trials,
    compare_paired_tasks,
    compare_win_rates,
    compute_aggregate_stats,
    compute_metrics_comparison,
    compute_skill_effectiveness,
    find_runs_by_prefix,
    print_comparison,
    print_metrics_comparison,
    print_skill_token_breakdown,
    print_skill_usage,
)
from analysis.shared import (
    analyze_skill_tokens_aggregate,
    print_distribution_comparison,
    print_simple_comparison,
    simple_comparison_to_dict,
)

# =============================================================================
# Distribution Comparison (--distribution flag)
# =============================================================================


def run_distribution_comparison(args: argparse.Namespace) -> int:
    """Run task success distribution comparison."""
    eval_root = Path(args.eval_dir)
    if not eval_root.exists():
        print(f"Error: Eval directory not found: {eval_root}", file=sys.stderr)
        return 1

    baseline_dirs = find_runs_by_prefix(eval_root, args.baseline_prefix)
    if not baseline_dirs:
        msg = f"Error: No runs matching '{args.baseline_prefix}*' in {eval_root}"
        print(msg, file=sys.stderr)
        return 1

    skills_dirs = find_runs_by_prefix(eval_root, args.skills_prefix)
    if not skills_dirs:
        msg = f"Error: No runs matching '{args.skills_prefix}*' in {eval_root}"
        print(msg, file=sys.stderr)
        return 1

    baseline_results = aggregate_runs(baseline_dirs)
    skills_results = aggregate_runs(skills_dirs)

    print_distribution_comparison(
        baseline_results, skills_results, args.baseline_prefix, args.skills_prefix
    )
    return 0


# =============================================================================
# Full Comparison (default)
# =============================================================================


def run_full_comparison(args: argparse.Namespace) -> int:
    """Run full statistical comparison with skill analysis."""
    eval_root = Path(args.eval_dir)

    # Find runs for each prefix
    baseline_dirs = find_runs_by_prefix(eval_root, args.baseline_prefix)
    skills_dirs = find_runs_by_prefix(eval_root, args.skills_prefix)

    if not baseline_dirs:
        msg = f"Error: No runs matching '{args.baseline_prefix}*' in {eval_root}"
        print(msg, file=sys.stderr)
        return 1

    if not skills_dirs:
        msg = f"Error: No runs matching '{args.skills_prefix}*' in {eval_root}"
        print(msg, file=sys.stderr)
        return 1

    # Aggregate results
    baseline_results = aggregate_runs(baseline_dirs)
    skills_results = aggregate_runs(skills_dirs)

    # Compute statistics
    baseline_stats = compute_aggregate_stats(baseline_results)
    skills_stats = compute_aggregate_stats(skills_results)

    # Statistical comparisons
    paired_comparison = compare_paired_tasks(baseline_results, skills_results)
    trial_comparison = compare_all_trials(baseline_results, skills_results)
    win_rate_comparison = compare_win_rates(baseline_results, skills_results)

    # Aggregate metrics (tokens, cost, time, steps)
    baseline_metrics = aggregate_metrics(baseline_dirs)
    skills_metrics = aggregate_metrics(skills_dirs)
    metrics_comparison = compute_metrics_comparison(baseline_metrics, skills_metrics)

    # Skill usage analysis (skills runs only)
    skill_usage = analyze_skill_usage_aggregate(skills_dirs)
    skill_effectiveness = compute_skill_effectiveness(skills_results, skill_usage)

    if args.json:
        output = {
            "baseline_prefix": args.baseline_prefix,
            "skills_prefix": args.skills_prefix,
            "baseline_runs": [d.name for d in baseline_dirs],
            "skills_runs": [d.name for d in skills_dirs],
            "baseline_stats": {
                k: v for k, v in baseline_stats.items() if k != "task_success_rates"
            },
            "skills_stats": {
                k: v for k, v in skills_stats.items() if k != "task_success_rates"
            },
            "paired_comparison": paired_comparison,
            "trial_comparison": trial_comparison,
            "win_rate_comparison": win_rate_comparison,
            "metrics_comparison": metrics_comparison,
            "skill_usage": {
                k: v
                for k, v in skill_usage.items()
                if k != "tasks_with_skills"  # set not JSON serializable
            },
            "skill_effectiveness": skill_effectiveness,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print_comparison(
            args.baseline_prefix,
            args.skills_prefix,
            baseline_dirs,
            skills_dirs,
            baseline_stats,
            skills_stats,
            paired_comparison,
            trial_comparison,
            win_rate_comparison,
        )
        print_metrics_comparison(metrics_comparison)

        # Skill token breakdown (needed for print_skill_usage)
        skill_token_analysis = analyze_skill_tokens_aggregate(skills_dirs)

        print_skill_usage(
            skill_usage,
            skill_effectiveness,
            baseline_results,
            skills_results,
            baseline_metrics,
            skills_metrics,
            skill_token_analysis,
        )
        print_skill_token_breakdown(skill_token_analysis)

    return 0


# =============================================================================
# Simple Comparison (--simple flag)
# =============================================================================


def run_simple_comparison(args: argparse.Namespace) -> int:
    """Run simple metrics comparison table."""
    eval_root = Path(args.eval_dir)
    if not eval_root.exists():
        print(f"Error: Eval directory not found: {eval_root}", file=sys.stderr)
        return 1

    baseline_metrics = load_prefix_metrics(eval_root, args.baseline_prefix)
    if baseline_metrics is None:
        msg = f"Error: No jobs matching '{args.baseline_prefix}*' in {eval_root}"
        print(msg, file=sys.stderr)
        return 1

    skills_metrics = load_prefix_metrics(eval_root, args.skills_prefix)
    if skills_metrics is None:
        msg = f"Error: No jobs matching '{args.skills_prefix}*' in {eval_root}"
        print(msg, file=sys.stderr)
        return 1

    if args.json:
        output = simple_comparison_to_dict(baseline_metrics, skills_metrics)
        print(json.dumps(output, indent=2))
    else:
        print_simple_comparison(baseline_metrics, skills_metrics)

    return 0


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs skills evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline-prefix",
        "-b",
        type=str,
        required=True,
        help="Prefix for baseline runs (e.g., 'tb-baseline')",
    )
    parser.add_argument(
        "--skills-prefix",
        "-s",
        type=str,
        required=True,
        help="Prefix for skills runs (e.g., 'tb-skill-v3')",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="outputs/evaluation",
        help="Evaluation directory to search (default: outputs/evaluation)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Show simple metrics comparison table only (no statistical analysis)",
    )
    parser.add_argument(
        "--distribution",
        action="store_true",
        help="Show task success distribution comparison",
    )
    args = parser.parse_args()

    if args.simple:
        return run_simple_comparison(args)
    elif args.distribution:
        return run_distribution_comparison(args)
    else:
        return run_full_comparison(args)


if __name__ == "__main__":
    sys.exit(main())

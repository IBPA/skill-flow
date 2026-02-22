"""Aggregate analysis modules for evaluation results."""

# Re-export from shared for backwards compatibility
from ..shared.metrics import find_jobs_by_prefix as find_runs_by_prefix
from ..shared.utils import EXCLUDED_TASKS, extract_task_name
from .core import (
    aggregate_metrics,
    aggregate_runs,
    analyze_skill_usage_aggregate,
    compute_aggregate_stats,
    compute_metrics_comparison,
    compute_skill_effectiveness,
    load_trial_results,
)
from .reporting import (
    print_comparison,
    print_metrics_comparison,
    print_skill_token_breakdown,
    print_skill_usage,
)
from .statistics import (
    compare_all_trials,
    compare_paired_tasks,
    compare_win_rates,
)

__all__ = [
    # re-exported from shared for backwards compatibility
    "EXCLUDED_TASKS",
    # metrics
    "aggregate_metrics",
    "aggregate_runs",
    "analyze_skill_usage_aggregate",
    "compare_all_trials",
    # statistics
    "compare_paired_tasks",
    "compare_win_rates",
    "compute_aggregate_stats",
    "compute_metrics_comparison",
    "compute_skill_effectiveness",
    "extract_task_name",
    "find_runs_by_prefix",
    # loaders
    "load_trial_results",
    # reporting
    "print_comparison",
    "print_metrics_comparison",
    "print_skill_token_breakdown",
    "print_skill_usage",
]

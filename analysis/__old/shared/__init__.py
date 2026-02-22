"""Shared utilities for analysis scripts."""

from .distribution import (
    compute_task_stats,
    print_distribution,
    print_distribution_comparison,
)
from .formatting import (
    aggregated_metrics_to_dict,
    fmt_num,
    fmt_pct,
    fmt_sec,
    fmt_tok,
    fmt_usd,
    job_metrics_to_dict,
    print_aggregated_metrics,
    print_job_metrics,
    print_simple_comparison,
    simple_comparison_to_dict,
)
from .metrics import (
    AggregatedMetrics,
    JobMetrics,
    Stats,
    aggregate_job_metrics,
    compute_stats,
    find_jobs_by_prefix,
    format_tokens,
    load_job_metrics,
    load_task_metrics,
)
from .models import SkillReadRecord, TaskMetrics
from .skill_detection import (
    analyze_rollout,
    detect_skill_reads,
    extract_skill_names_from_instructions,
    find_rollout_files,
)
from .token_analysis import analyze_skill_tokens_aggregate
from .utils import (
    CACHE_TOKEN_COST_PER_1K,
    EXCLUDED_TASKS,
    INPUT_TOKEN_COST_PER_1K,
    OUTPUT_TOKEN_COST_PER_1K,
    calculate_cost,
    extract_task_name,
    pct_diff,
    safe_mean,
    safe_median,
)

__all__ = [
    "CACHE_TOKEN_COST_PER_1K",
    # utils (common utilities)
    "EXCLUDED_TASKS",
    # cost
    "INPUT_TOKEN_COST_PER_1K",
    "OUTPUT_TOKEN_COST_PER_1K",
    "AggregatedMetrics",
    "JobMetrics",
    "SkillReadRecord",
    # metrics (simplified aggregation)
    "Stats",
    # models
    "TaskMetrics",
    "aggregate_job_metrics",
    "aggregated_metrics_to_dict",
    "analyze_rollout",
    # token_analysis
    "analyze_skill_tokens_aggregate",
    "calculate_cost",
    "compute_stats",
    # distribution
    "compute_task_stats",
    "detect_skill_reads",
    "extract_skill_names_from_instructions",
    "extract_task_name",
    "find_jobs_by_prefix",
    # skill_detection
    "find_rollout_files",
    "fmt_num",
    # formatting
    "fmt_pct",
    "fmt_sec",
    "fmt_tok",
    "fmt_usd",
    "format_tokens",
    "job_metrics_to_dict",
    "load_job_metrics",
    # task_metrics
    "load_task_metrics",
    "pct_diff",
    "print_aggregated_metrics",
    "print_distribution",
    "print_distribution_comparison",
    "print_job_metrics",
    "print_simple_comparison",
    "safe_mean",
    "safe_median",
    "simple_comparison_to_dict",
]

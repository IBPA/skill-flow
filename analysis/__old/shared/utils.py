"""Common utility functions for analysis modules.

This module consolidates frequently-used utilities to avoid duplication across
analysis scripts. Import these functions instead of re-implementing them.
"""

from collections.abc import Sequence

# Tasks to exclude from analysis (known problematic)
EXCLUDED_TASKS = {"train-fasttext", "fix-ocaml-gc"}

# Token pricing constants (per 1K tokens)
INPUT_TOKEN_COST_PER_1K = 0.00025  # $0.25/1M input
OUTPUT_TOKEN_COST_PER_1K = 0.002  # $2/1M output
CACHE_TOKEN_COST_PER_1K = 0.000025  # 90% discount for cached


def calculate_cost(input_tokens: int, output_tokens: int, cache_tokens: int) -> float:
    """Calculate cost in USD based on token counts."""
    non_cached_input = input_tokens - cache_tokens
    cost = (
        (non_cached_input / 1000) * INPUT_TOKEN_COST_PER_1K
        + (cache_tokens / 1000) * CACHE_TOKEN_COST_PER_1K
        + (output_tokens / 1000) * OUTPUT_TOKEN_COST_PER_1K
    )
    return cost


def extract_task_name(task_id: str) -> str:
    """Extract base task name without hash suffix.

    Example: "my-task__abc123" -> "my-task"
    """
    return task_id.split("__")[0] if "__" in task_id else task_id


def pct_diff(new: float, old: float) -> float:
    """Calculate percentage difference between two values.

    Returns ((new - old) / old * 100) if old != 0, else 0.0
    """
    return ((new - old) / old * 100) if old else 0.0


def safe_mean(values: Sequence[float | int]) -> float:
    """Compute mean of a sequence, returning 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0


def safe_median(values: Sequence[float | int]) -> float:
    """Compute median of a sequence, returning 0.0 if empty."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return float(s[n // 2])

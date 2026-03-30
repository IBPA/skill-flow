"""Bridge between ConditionResults/TaskReward and statistical utilities.

Extracts per-task metric values from benchmark evaluation data and computes
bootstrap CIs, paired tests, and McNemar's test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from analysis.stats.bootstrap import bootstrap_ci, paired_bootstrap_test
from analysis.stats.proportions import mcnemar_test

if TYPE_CHECKING:
    from analysis.comparison.utils.loader import ConditionResults
    from analysis.stats.types import ConfidenceInterval, McNemarResult, PairedTestResult

_DEFAULT_THRESHOLD = 1.0


def _extract_pass_at_k(
    cond: ConditionResults, k: int, threshold: float = _DEFAULT_THRESHOLD
) -> list[float]:
    """Extract per-task Pass@k values from a condition."""
    return [tr.pass_at_k(threshold, k) for tr in cond.task_rewards.values()]


def _extract_pass_pow_k(
    cond: ConditionResults, k: int, threshold: float = _DEFAULT_THRESHOLD
) -> list[float]:
    """Extract per-task Pass^k values from a condition."""
    return [tr.pass_pow_k(threshold, k) for tr in cond.task_rewards.values()]


def _extract_mean_reward(cond: ConditionResults) -> list[float]:
    """Extract per-task mean reward values from a condition."""
    return [tr.mean_reward for tr in cond.task_rewards.values()]


def _extract_mean_steps(cond: ConditionResults) -> list[float]:
    """Extract per-task mean steps values from a condition."""
    return [tr.mean_steps for tr in cond.task_rewards.values()]


def _extract_mean_cost(cond: ConditionResults) -> list[float]:
    """Extract per-task mean cost values from a condition."""
    return [tr.mean_cost for tr in cond.task_rewards.values()]


def _extract_values(
    cond: ConditionResults,
    metric: str,
    k: int,
    threshold: float,
) -> list[float]:
    """Extract per-task values for the given metric."""
    if metric == "pass_at":
        return _extract_pass_at_k(cond, k, threshold)
    if metric == "pass_pow":
        return _extract_pass_pow_k(cond, k, threshold)
    if metric == "mean_reward":
        return _extract_mean_reward(cond)
    if metric == "mean_steps":
        return _extract_mean_steps(cond)
    if metric == "mean_cost":
        return _extract_mean_cost(cond)
    msg = f"Unknown metric: {metric!r}"
    raise ValueError(msg)


def benchmark_ci(
    cond: ConditionResults,
    metric: str,
    k: int = 1,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> ConfidenceInterval:
    """Bootstrap CI for a benchmark metric averaged across tasks.

    Args:
        cond: Loaded condition results.
        metric: One of "pass_at", "pass_pow", "mean_reward",
                "mean_steps", "mean_cost".
        k: The k parameter for Pass@k / Pass^k.
        threshold: Reward threshold for pass/fail.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level.
        seed: RNG seed.

    Returns:
        ConfidenceInterval for the mean of the chosen metric across tasks.
    """
    values = _extract_values(cond, metric, k, threshold)
    return bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)


def benchmark_paired_test(
    cond_a: ConditionResults,
    cond_b: ConditionResults,
    metric: str,
    k: int = 1,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairedTestResult:
    """Paired bootstrap test comparing two conditions on a metric.

    Tasks are aligned by name: only tasks present in both conditions are
    compared. Task order is sorted alphabetically to ensure reproducibility.

    Args:
        cond_a: First condition (e.g. treatment).
        cond_b: Second condition (e.g. baseline).
        metric: Metric name (see ``benchmark_ci``).
        k: The k parameter for Pass@k / Pass^k.
        threshold: Reward threshold.
        n_boot: Number of bootstrap resamples.
        alpha: Significance level.
        seed: RNG seed.

    Returns:
        PairedTestResult for the mean difference (A - B).
    """
    shared = sorted(cond_a.all_tasks & cond_b.all_tasks)
    vals_a = _extract_paired(cond_a, shared, metric, k, threshold)
    vals_b = _extract_paired(cond_b, shared, metric, k, threshold)
    return paired_bootstrap_test(vals_a, vals_b, n_boot=n_boot, alpha=alpha, seed=seed)


def _extract_paired(
    cond: ConditionResults,
    tasks: list[str],
    metric: str,
    k: int,
    threshold: float,
) -> list[float]:
    """Extract values for specific tasks in order."""
    if metric == "pass_at":
        return [cond.task_rewards[t].pass_at_k(threshold, k) for t in tasks]
    if metric == "pass_pow":
        return [cond.task_rewards[t].pass_pow_k(threshold, k) for t in tasks]
    if metric == "mean_reward":
        return [cond.task_rewards[t].mean_reward for t in tasks]
    if metric == "mean_steps":
        return [cond.task_rewards[t].mean_steps for t in tasks]
    if metric == "mean_cost":
        return [cond.task_rewards[t].mean_cost for t in tasks]
    msg = f"Unknown metric: {metric!r}"
    raise ValueError(msg)


def benchmark_mcnemar(
    cond_a: ConditionResults,
    cond_b: ConditionResults,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
    alpha: float = 0.05,
) -> McNemarResult:
    """McNemar's test aggregating discordant pairs across all runs.

    For each shared task and each run, counts:
        b = task passes in A but fails in B
        c = task fails in A but passes in B

    Args:
        cond_a: First condition.
        cond_b: Second condition.
        threshold: Reward threshold.
        alpha: Significance level.

    Returns:
        McNemarResult for the aggregated discordant counts.
    """
    shared = sorted(cond_a.all_tasks & cond_b.all_tasks)
    n_runs = min(len(cond_a.runs), len(cond_b.runs))
    b_count = 0
    c_count = 0
    for task in shared:
        tr_a = cond_a.task_rewards[task]
        tr_b = cond_b.task_rewards[task]
        for run_idx in range(n_runs):
            a_pass = tr_a.rewards[run_idx] >= threshold
            b_pass = tr_b.rewards[run_idx] >= threshold
            if a_pass and not b_pass:
                b_count += 1
            elif b_pass and not a_pass:
                c_count += 1
    return mcnemar_test(b_count, c_count, alpha=alpha)


def significance_stars(p_value: float) -> str:
    """Convert a p-value to significance stars for display.

    Returns:
        "" if p >= 0.05, "*" if p < 0.05, "**" if p < 0.01,
        "***" if p < 0.001.
    """
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""

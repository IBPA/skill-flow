"""Bootstrap confidence intervals and paired hypothesis tests.

All implementations use numpy only (no scipy). The percentile method is used
for CIs, which is standard and well-accepted for sample sizes >= 30.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from analysis.stats.types import ConfidenceInterval, PairedTestResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

_DEFAULT_N_BOOT = 10_000
_DEFAULT_ALPHA = 0.05
_DEFAULT_SEED = 42


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = _DEFAULT_N_BOOT,
    alpha: float = _DEFAULT_ALPHA,
    seed: int = _DEFAULT_SEED,
) -> ConfidenceInterval:
    """Compute a percentile bootstrap CI for the mean of *values*.

    Args:
        values: Sample values (one per observation, e.g. one per task).
        n_boot: Number of bootstrap resamples.
        alpha: Significance level (0.05 for 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ConfidenceInterval with mean, lower, and upper bounds.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return ConfidenceInterval(mean=0.0, ci_lo=0.0, ci_hi=0.0, alpha=alpha)
    if n == 1:
        v = float(arr[0])
        return ConfidenceInterval(mean=v, ci_lo=v, ci_hi=v, alpha=alpha)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means: NDArray[np.floating] = arr[indices].mean(axis=1)

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return ConfidenceInterval(mean=float(arr.mean()), ci_lo=lo, ci_hi=hi, alpha=alpha)


def paired_bootstrap_test(
    values_a: list[float],
    values_b: list[float],
    *,
    n_boot: int = _DEFAULT_N_BOOT,
    alpha: float = _DEFAULT_ALPHA,
    seed: int = _DEFAULT_SEED,
) -> PairedTestResult:
    """Paired bootstrap test for H0: mean(A) == mean(B).

    Computes a CI for the mean difference (A - B) and a two-sided p-value
    as the fraction of bootstrap differences whose sign disagrees with the
    observed difference.

    Args:
        values_a: Per-observation values for condition A.
        values_b: Per-observation values for condition B (same length).
        n_boot: Number of bootstrap resamples.
        alpha: Significance level.
        seed: RNG seed for reproducibility.

    Returns:
        PairedTestResult with observed difference, CI, and p-value.

    Raises:
        ValueError: If inputs have different lengths.
    """
    if len(values_a) != len(values_b):
        msg = (
            f"Paired test requires equal-length inputs, "
            f"got {len(values_a)} and {len(values_b)}"
        )
        raise ValueError(msg)

    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)
    diffs = arr_a - arr_b
    n = len(diffs)

    if n == 0:
        ci = ConfidenceInterval(mean=0.0, ci_lo=0.0, ci_hi=0.0, alpha=alpha)
        return PairedTestResult(
            observed_diff=0.0, ci=ci, p_value=1.0, significant=False
        )

    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_diffs: NDArray[np.floating] = diffs[indices].mean(axis=1)

    lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))
    ci = ConfidenceInterval(mean=observed, ci_lo=lo, ci_hi=hi, alpha=alpha)

    # Two-sided p-value: fraction of bootstrap samples on the opposite side of 0
    if observed > 0:
        p_value = float(np.mean(boot_diffs <= 0)) * 2
    elif observed < 0:
        p_value = float(np.mean(boot_diffs >= 0)) * 2
    else:
        p_value = 1.0
    p_value = min(p_value, 1.0)

    return PairedTestResult(
        observed_diff=observed,
        ci=ci,
        p_value=p_value,
        significant=p_value < alpha,
    )

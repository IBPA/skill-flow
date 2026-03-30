"""Statistical tests for proportions: Wilson interval, McNemar, Cohen's h.

All implementations are pure Python + math (no scipy). The McNemar p-value
uses the exact relationship between chi-squared(1 df) and the error function.
"""

from __future__ import annotations

import math

from analysis.stats.types import EffectSize, McNemarResult, WilsonInterval

# z-values for common alpha levels (two-sided).
_Z_TABLE: dict[float, float] = {
    0.10: 1.6449,
    0.05: 1.9600,
    0.01: 2.5758,
    0.001: 3.2905,
}


def _z_for_alpha(alpha: float) -> float:
    """Return the z critical value for a two-sided test at *alpha*."""
    z = _Z_TABLE.get(alpha)
    if z is not None:
        return z
    # Fallback: pick the closest available alpha.
    closest = min(_Z_TABLE, key=lambda a: abs(a - alpha))
    return _Z_TABLE[closest]


def wilson_score_interval(
    successes: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> WilsonInterval:
    """Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        alpha: Significance level (0.05 for 95% CI).

    Returns:
        WilsonInterval with proportion and bounds.
    """
    if total == 0:
        return WilsonInterval(proportion=0.0, ci_lo=0.0, ci_hi=0.0, n=0)

    p_hat = successes / total
    z = _z_for_alpha(alpha)
    z2 = z * z

    denom = 1 + z2 / total
    centre = p_hat + z2 / (2 * total)
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z2 / (4 * total)) / total)

    lo = max(0.0, (centre - spread) / denom)
    hi = min(1.0, (centre + spread) / denom)

    return WilsonInterval(proportion=p_hat, ci_lo=lo, ci_hi=hi, n=total)


def _chi2_sf_1df(x: float) -> float:
    """Survival function (1 - CDF) of chi-squared with 1 degree of freedom.

    Uses the identity: chi2_sf(x, 1) = erfc(sqrt(x / 2)), which is exact.
    """
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2))


def mcnemar_test(
    b: int,
    c: int,
    *,
    alpha: float = 0.05,
) -> McNemarResult:
    """McNemar's test for paired binary outcomes.

    Given a 2x2 contingency table of paired outcomes (A vs B):
        - b = count(A passes, B fails)
        - c = count(A fails, B passes)

    Uses the chi-squared approximation with continuity correction when
    n_discordant >= 25, otherwise uses the uncorrected version.

    Args:
        b: Discordant count (A wins).
        c: Discordant count (B wins).
        alpha: Significance level.

    Returns:
        McNemarResult with test statistic, p-value, and significance.
    """
    n_disc = b + c
    if n_disc == 0:
        return McNemarResult(
            statistic=0.0, p_value=1.0, n_discordant=0, significant=False
        )

    # Use continuity correction for larger samples.
    stat = (abs(b - c) - 1) ** 2 / n_disc if n_disc >= 25 else (b - c) ** 2 / n_disc

    p_value = _chi2_sf_1df(stat)

    return McNemarResult(
        statistic=stat,
        p_value=p_value,
        n_discordant=n_disc,
        significant=p_value < alpha,
    )


def cohens_h(p1: float, p2: float) -> EffectSize:
    """Cohen's h effect size for comparing two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Interpretation (Cohen, 1988):
        |h| < 0.20  → negligible
        |h| < 0.50  → small
        |h| < 0.80  → medium
        |h| >= 0.80 → large

    Args:
        p1: First proportion (e.g. treatment pass rate).
        p2: Second proportion (e.g. baseline pass rate).

    Returns:
        EffectSize with Cohen's h value and interpretation string.
    """
    h = 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))
    abs_h = abs(h)

    if abs_h < 0.20:
        interp = "negligible"
    elif abs_h < 0.50:
        interp = "small"
    elif abs_h < 0.80:
        interp = "medium"
    else:
        interp = "large"

    return EffectSize(cohens_h=h, interpretation=interp)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Controls the family-wise error rate (FWER). Returns adjusted p-values
    in the same order as the input.

    Args:
        p_values: Raw p-values (one per comparison).
        alpha: Nominal significance level (used only for documentation;
               the returned adjusted p-values can be compared against any alpha).

    Returns:
        Adjusted p-values in the original input order.
    """
    _ = alpha  # retained for API clarity
    n = len(p_values)
    if n == 0:
        return []

    # Sort by raw p-value, keeping original indices.
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n

    # Step-down: multiply by (n - rank), enforce monotonicity.
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(p * (n - rank), 1.0)
        running_max = max(adj, running_max)
        adjusted[orig_idx] = running_max

    return adjusted

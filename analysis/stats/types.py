"""Pydantic models for statistical test results."""

from pydantic import BaseModel


class ConfidenceInterval(BaseModel, frozen=True):
    """Bootstrap confidence interval for a sample mean."""

    mean: float
    ci_lo: float
    ci_hi: float
    alpha: float = 0.05


class PairedTestResult(BaseModel, frozen=True):
    """Result of a paired bootstrap hypothesis test."""

    observed_diff: float
    ci: ConfidenceInterval
    p_value: float
    significant: bool


class WilsonInterval(BaseModel, frozen=True):
    """Wilson score confidence interval for a binomial proportion."""

    proportion: float
    ci_lo: float
    ci_hi: float
    n: int


class McNemarResult(BaseModel, frozen=True):
    """Result of McNemar's test for paired binary outcomes."""

    statistic: float
    p_value: float
    n_discordant: int
    significant: bool


class EffectSize(BaseModel, frozen=True):
    """Cohen's h effect size for comparing two proportions."""

    cohens_h: float
    interpretation: str  # "negligible", "small", "medium", "large"

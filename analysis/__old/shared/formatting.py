"""Formatting functions for metrics display and JSON conversion."""

from analysis.shared.metrics import (
    AggregatedMetrics,
    JobMetrics,
    Stats,
    format_tokens,
)
from analysis.shared.utils import pct_diff

# =============================================================================
# Stat Formatters (reusable)
# =============================================================================


def fmt_pct(s: Stats) -> tuple[str, str]:
    """Format percentage stat as (mean, std) strings."""
    return f"{s.mean:.1%}", f"+/- {s.std:.1%}"


def fmt_usd(s: Stats) -> tuple[str, str]:
    """Format USD stat as (mean, std) strings."""
    return f"${s.mean:.4f}", f"+/- ${s.std:.4f}"


def fmt_sec(s: Stats) -> tuple[str, str]:
    """Format seconds stat as (mean, std) strings."""
    return f"{s.mean:.1f}s", f"+/- {s.std:.1f}s"


def fmt_tok(s: Stats) -> tuple[str, str]:
    """Format token stat as (mean, std) strings."""
    return format_tokens(s.mean), f"+/- {format_tokens(s.std)}"


def fmt_num(s: Stats) -> tuple[str, str]:
    """Format numeric stat as (mean, std) strings."""
    return f"{s.mean:.1f}", f"+/- {s.std:.1f}"


# =============================================================================
# Job Metrics Display
# =============================================================================


def print_job_metrics(metrics: JobMetrics) -> None:
    """Print formatted job metrics to stdout."""
    print(f"Job: {metrics.job_name}")
    print(f"Tasks: {metrics.n_tasks}")
    print("-" * 50)
    print(f"{'Metric':<20s} {'Value':>20s}")
    print("-" * 50)
    print(f"{'Success rate':<20s} {metrics.success_rate:>19.1%}")
    print(f"{'Cost (USD)':<20s} ${metrics.cost_usd:>18.4f}")
    print(f"{'Time (sec)':<20s} {metrics.time_sec:>19.1f}")
    print(f"{'Input tokens':<20s} {format_tokens(metrics.input_tokens):>20s}")
    print(f"{'Cached tokens':<20s} {format_tokens(metrics.cached_tokens):>20s}")
    print(f"{'Output tokens':<20s} {format_tokens(metrics.output_tokens):>20s}")
    print(f"{'Steps':<20s} {metrics.steps:>20.1f}")


def job_metrics_to_dict(metrics: JobMetrics) -> dict:
    """Convert JobMetrics to JSON-serializable dict."""
    return {
        "job_name": metrics.job_name,
        "n_tasks": metrics.n_tasks,
        "success_rate": metrics.success_rate,
        "cost_usd": metrics.cost_usd,
        "time_sec": metrics.time_sec,
        "input_tokens": metrics.input_tokens,
        "cached_tokens": metrics.cached_tokens,
        "output_tokens": metrics.output_tokens,
        "steps": metrics.steps,
    }


# =============================================================================
# Aggregated Metrics Display
# =============================================================================


def print_aggregated_metrics(metrics: AggregatedMetrics) -> None:
    """Print formatted aggregated metrics with mean +/- std."""
    print(f"Prefix: {metrics.name}*")
    print(f"Jobs: {metrics.n_jobs}, Total tasks: {metrics.n_tasks}")
    print("-" * 60)
    print(f"{'Metric':<20s} {'Mean':>15s} {'Std':>15s}")
    print("-" * 60)

    rows = [
        ("Success rate", fmt_pct(metrics.success_rate)),
        ("Cost (USD)", fmt_usd(metrics.cost_usd)),
        ("Time (sec)", fmt_sec(metrics.time_sec)),
        ("Input tokens", fmt_tok(metrics.input_tokens)),
        ("Cached tokens", fmt_tok(metrics.cached_tokens)),
        ("Output tokens", fmt_tok(metrics.output_tokens)),
        ("Steps", fmt_num(metrics.steps)),
    ]

    for name, (mean_str, std_str) in rows:
        print(f"{name:<20s} {mean_str:>15s} {std_str:>15s}")


def aggregated_metrics_to_dict(metrics: AggregatedMetrics) -> dict:
    """Convert AggregatedMetrics to JSON-serializable dict."""
    return {
        "name": metrics.name,
        "n_jobs": metrics.n_jobs,
        "n_tasks": metrics.n_tasks,
        "success_rate": {
            "mean": metrics.success_rate.mean,
            "std": metrics.success_rate.std,
        },
        "cost_usd": {"mean": metrics.cost_usd.mean, "std": metrics.cost_usd.std},
        "time_sec": {"mean": metrics.time_sec.mean, "std": metrics.time_sec.std},
        "input_tokens": {
            "mean": metrics.input_tokens.mean,
            "std": metrics.input_tokens.std,
        },
        "cached_tokens": {
            "mean": metrics.cached_tokens.mean,
            "std": metrics.cached_tokens.std,
        },
        "output_tokens": {
            "mean": metrics.output_tokens.mean,
            "std": metrics.output_tokens.std,
        },
        "steps": {"mean": metrics.steps.mean, "std": metrics.steps.std},
    }


# =============================================================================
# Simple Comparison Display
# =============================================================================


def print_simple_comparison(a: AggregatedMetrics, b: AggregatedMetrics) -> None:
    """Print side-by-side comparison of two aggregated metrics."""
    print("=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"  Baseline: {a.name}* ({a.n_jobs} jobs, {a.n_tasks} tasks)")
    print(f"  Skills:   {b.name}* ({b.n_jobs} jobs, {b.n_tasks} tasks)")
    print()
    print("-" * 80)
    header = f"{'Metric':<18s} {'Baseline':>22s} {'Skills':>22s} {'Diff':>12s}"
    print(header)
    print("-" * 80)

    def fmt_pct_inline(s: Stats) -> str:
        return f"{s.mean:.1%} +/- {s.std:.1%}"

    def fmt_usd_inline(s: Stats) -> str:
        return f"${s.mean:.4f} +/- ${s.std:.4f}"

    def fmt_sec_inline(s: Stats) -> str:
        return f"{s.mean:.1f} +/- {s.std:.1f}s"

    def fmt_tok_inline(s: Stats) -> str:
        return f"{format_tokens(s.mean)} +/- {format_tokens(s.std)}"

    def fmt_num_inline(s: Stats) -> str:
        return f"{s.mean:.1f} +/- {s.std:.1f}"

    rows = [
        (
            "Success rate",
            fmt_pct_inline(a.success_rate),
            fmt_pct_inline(b.success_rate),
            pct_diff(b.success_rate.mean, a.success_rate.mean),
        ),
        (
            "Cost (USD)",
            fmt_usd_inline(a.cost_usd),
            fmt_usd_inline(b.cost_usd),
            pct_diff(b.cost_usd.mean, a.cost_usd.mean),
        ),
        (
            "Time (sec)",
            fmt_sec_inline(a.time_sec),
            fmt_sec_inline(b.time_sec),
            pct_diff(b.time_sec.mean, a.time_sec.mean),
        ),
        (
            "Input tokens",
            fmt_tok_inline(a.input_tokens),
            fmt_tok_inline(b.input_tokens),
            pct_diff(b.input_tokens.mean, a.input_tokens.mean),
        ),
        (
            "Cached tokens",
            fmt_tok_inline(a.cached_tokens),
            fmt_tok_inline(b.cached_tokens),
            pct_diff(b.cached_tokens.mean, a.cached_tokens.mean),
        ),
        (
            "Output tokens",
            fmt_tok_inline(a.output_tokens),
            fmt_tok_inline(b.output_tokens),
            pct_diff(b.output_tokens.mean, a.output_tokens.mean),
        ),
        (
            "Steps",
            fmt_num_inline(a.steps),
            fmt_num_inline(b.steps),
            pct_diff(b.steps.mean, a.steps.mean),
        ),
    ]

    for name, a_str, b_str, diff in rows:
        diff_str = f"{diff:+.1f}%"
        print(f"{name:<18s} {a_str:>22s} {b_str:>22s} {diff_str:>12s}")

    print("-" * 80)


def simple_comparison_to_dict(a: AggregatedMetrics, b: AggregatedMetrics) -> dict:
    """Convert comparison to JSON-serializable dict."""
    a_dict = aggregated_metrics_to_dict(a)
    b_dict = aggregated_metrics_to_dict(b)

    diffs = {
        "success_rate_pct": pct_diff(b.success_rate.mean, a.success_rate.mean),
        "cost_usd_pct": pct_diff(b.cost_usd.mean, a.cost_usd.mean),
        "time_sec_pct": pct_diff(b.time_sec.mean, a.time_sec.mean),
        "input_tokens_pct": pct_diff(b.input_tokens.mean, a.input_tokens.mean),
        "cached_tokens_pct": pct_diff(b.cached_tokens.mean, a.cached_tokens.mean),
        "output_tokens_pct": pct_diff(b.output_tokens.mean, a.output_tokens.mean),
        "steps_pct": pct_diff(b.steps.mean, a.steps.mean),
    }

    return {
        "baseline": a_dict,
        "skills": b_dict,
        "diff_pct": diffs,
    }

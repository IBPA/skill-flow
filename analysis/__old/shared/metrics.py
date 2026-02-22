"""Core metrics aggregation utilities.

This module provides the foundation for computing and displaying metrics
across evaluation jobs. It defines the standard metrics structure and
utility functions for computing mean ± std statistics.
"""

import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .models import TaskMetrics
from .utils import EXCLUDED_TASKS, calculate_cost

# =============================================================================
# Task Metrics Loading
# =============================================================================


def _count_steps_in_rollout(task_dir: Path) -> int:
    """Count function_call entries in rollout JSONL files."""
    sessions_dir = task_dir / "agent" / "sessions"
    if not sessions_dir.exists():
        return 0

    n_steps = 0
    for root, _, files in os.walk(sessions_dir):
        for f in files:
            if not f.endswith(".jsonl"):
                continue
            rollout_path = Path(root) / f
            with rollout_path.open(encoding="utf-8") as fp:
                for line in fp:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Count function_call in payload (OpenAI format)
                    payload = entry.get("payload", {})
                    if payload.get("type") == "function_call":
                        n_steps += 1
                    # Count command_execution in item (alternative format)
                    item = entry.get("item", {})
                    if item.get("type") == "command_execution":
                        n_steps += 1
    return n_steps


def load_task_metrics(eval_dir: Path) -> list[TaskMetrics]:
    """Load detailed metrics from individual task result.json files."""
    metrics = []

    for task_dir in eval_dir.iterdir():
        if not task_dir.is_dir() or task_dir.name.startswith("."):
            continue

        result_file = task_dir / "result.json"
        if not result_file.exists():
            continue

        with result_file.open() as f:
            data = json.load(f)

        # Extract task name (without hash suffix)
        task_name = data.get("task_name", task_dir.name.split("__")[0])

        # Get reward
        verifier = data.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        reward = rewards.get("reward", 0.0)

        # Get token usage
        agent_result = data.get("agent_result") or {}
        input_tokens = agent_result.get("n_input_tokens", 0) or 0
        output_tokens = agent_result.get("n_output_tokens", 0) or 0
        cache_tokens = agent_result.get("n_cache_tokens", 0) or 0

        # Get execution time
        agent_exec = data.get("agent_execution") or {}
        exec_time = 0.0
        if agent_exec.get("started_at") and agent_exec.get("finished_at"):
            start = datetime.fromisoformat(agent_exec["started_at"])
            end = datetime.fromisoformat(agent_exec["finished_at"])
            exec_time = (end - start).total_seconds()

        # Count steps from rollout
        n_steps = _count_steps_in_rollout(task_dir)

        metrics.append(
            TaskMetrics(
                task_name=task_name,
                reward=reward,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_tokens=cache_tokens,
                execution_time_sec=exec_time,
                n_steps=n_steps,
            )
        )

    return metrics


# =============================================================================
# Metrics Dataclasses
# =============================================================================


@dataclass
class Stats:
    """Mean and standard deviation pair."""

    mean: float = 0.0
    std: float = 0.0

    def __str__(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f}"

    def format(self, fmt: str = ".4f", unit: str = "") -> str:
        """Format with custom precision and optional unit."""
        return f"{self.mean:{fmt}}{unit} ± {self.std:{fmt}}{unit}"


@dataclass
class JobMetrics:
    """Aggregated metrics for a single evaluation job.

    All numeric fields represent mean across tasks in the job.
    """

    job_name: str
    n_tasks: int = 0
    success_rate: float = 0.0
    cost_usd: float = 0.0
    time_sec: float = 0.0
    input_tokens: float = 0.0
    cached_tokens: float = 0.0
    output_tokens: float = 0.0
    steps: float = 0.0


@dataclass
class AggregatedMetrics:
    """Metrics aggregated across multiple jobs with mean ± std.

    Use compute_stats() to aggregate multiple JobMetrics into this.
    """

    name: str
    n_jobs: int = 0
    n_tasks: int = 0
    success_rate: Stats = field(default_factory=Stats)
    cost_usd: Stats = field(default_factory=Stats)
    time_sec: Stats = field(default_factory=Stats)
    input_tokens: Stats = field(default_factory=Stats)
    cached_tokens: Stats = field(default_factory=Stats)
    output_tokens: Stats = field(default_factory=Stats)
    steps: Stats = field(default_factory=Stats)


def compute_stats(values: list[float]) -> Stats:
    """Compute mean and standard deviation from a list of values."""
    if not values:
        return Stats(0.0, 0.0)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return Stats(mean, std)


def load_job_metrics(eval_dir: Path) -> JobMetrics:
    """Load and aggregate metrics for a single evaluation job.

    Args:
        eval_dir: Path to evaluation output directory
            (e.g., outputs/evaluation/tb-baseline-001)

    Returns:
        JobMetrics with averaged values across all tasks in the job
    """
    task_metrics = load_task_metrics(eval_dir)

    # Filter excluded tasks
    task_metrics = [m for m in task_metrics if m.task_name not in EXCLUDED_TASKS]

    if not task_metrics:
        return JobMetrics(job_name=eval_dir.name, n_tasks=0)

    n = len(task_metrics)
    successes = sum(1 for m in task_metrics if m.reward == 1.0)

    return JobMetrics(
        job_name=eval_dir.name,
        n_tasks=n,
        success_rate=successes / n,
        cost_usd=sum(
            calculate_cost(m.input_tokens, m.output_tokens, m.cache_tokens)
            for m in task_metrics
        )
        / n,
        time_sec=sum(m.execution_time_sec for m in task_metrics) / n,
        input_tokens=sum(m.input_tokens for m in task_metrics) / n,
        cached_tokens=sum(m.cache_tokens for m in task_metrics) / n,
        output_tokens=sum(m.output_tokens for m in task_metrics) / n,
        steps=sum(m.n_steps for m in task_metrics) / n,
    )


def aggregate_job_metrics(jobs: list[JobMetrics], name: str = "") -> AggregatedMetrics:
    """Aggregate multiple JobMetrics into AggregatedMetrics with mean ± std.

    Args:
        jobs: List of JobMetrics from individual evaluation runs
        name: Name for the aggregated result (e.g., prefix name)

    Returns:
        AggregatedMetrics with mean ± std for each metric
    """
    if not jobs:
        return AggregatedMetrics(name=name)

    return AggregatedMetrics(
        name=name,
        n_jobs=len(jobs),
        n_tasks=sum(j.n_tasks for j in jobs),
        success_rate=compute_stats([j.success_rate for j in jobs]),
        cost_usd=compute_stats([j.cost_usd for j in jobs]),
        time_sec=compute_stats([j.time_sec for j in jobs]),
        input_tokens=compute_stats([j.input_tokens for j in jobs]),
        cached_tokens=compute_stats([j.cached_tokens for j in jobs]),
        output_tokens=compute_stats([j.output_tokens for j in jobs]),
        steps=compute_stats([j.steps for j in jobs]),
    )


def find_jobs_by_prefix(eval_root: Path, prefix: str) -> list[Path]:
    """Find all evaluation job directories matching the given prefix.

    Args:
        eval_root: Root directory containing evaluation outputs
        prefix: Prefix to match (e.g., "tb-baseline")

    Returns:
        Sorted list of matching directory paths
    """
    if not eval_root.exists():
        return []
    return sorted(d for d in eval_root.glob(f"{prefix}*") if d.is_dir())


def format_tokens(n: float) -> str:
    """Format token count compactly (e.g., '123k', '1.5M')."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return f"{n:.0f}"

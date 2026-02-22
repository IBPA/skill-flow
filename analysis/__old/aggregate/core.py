"""Core aggregate analysis: result loading and metrics aggregation."""

import json
from collections import defaultdict
from pathlib import Path

from ..shared import (
    EXCLUDED_TASKS,
    TaskMetrics,
    analyze_rollout,
    calculate_cost,
    extract_task_name,
    find_rollout_files,
    load_task_metrics,
    pct_diff,
    safe_mean,
)

# =============================================================================
# Result Loading
# =============================================================================


def load_trial_results(eval_dir: Path) -> dict[str, float]:
    """Load Harbor evaluation results from task directories.

    Returns dict of {task_id: reward}
    """
    results = {}

    for task_dir in eval_dir.iterdir():
        if not task_dir.is_dir():
            continue

        result_file = task_dir / "result.json"
        if not result_file.exists():
            continue

        with result_file.open() as f:
            task_result = json.load(f)

        task_id = task_dir.name
        verifier_result = task_result.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") or {}
        reward = rewards.get("reward", 0.0)
        results[task_id] = reward

    return results


def aggregate_runs(eval_dirs: list[Path]) -> dict[str, list[float]]:
    """Aggregate results across multiple evaluation runs.

    Returns dict of {task_name: [reward1, reward2, ...]}
    """
    task_results: dict[str, list[float]] = defaultdict(list)

    for eval_dir in eval_dirs:
        results = load_trial_results(eval_dir)

        for task_id, reward in results.items():
            task_name = extract_task_name(task_id)
            if task_name in EXCLUDED_TASKS:
                continue
            task_results[task_name].append(reward)

    return dict(task_results)


def compute_aggregate_stats(task_results: dict[str, list[float]]) -> dict:
    """Compute aggregate statistics for a set of runs."""
    total_trials = sum(len(rewards) for rewards in task_results.values())
    total_successes = sum(
        sum(1 for r in rewards if r == 1.0) for rewards in task_results.values()
    )

    # Per-task success rates
    task_success_rates = {}
    for task_name, rewards in task_results.items():
        successes = sum(1 for r in rewards if r == 1.0)
        task_success_rates[task_name] = successes / len(rewards) if rewards else 0.0

    return {
        "n_tasks": len(task_results),
        "n_trials": total_trials,
        "n_successes": total_successes,
        "overall_success_rate": total_successes / total_trials if total_trials else 0.0,
        "task_success_rates": task_success_rates,
    }


# =============================================================================
# Metrics Aggregation
# =============================================================================


def aggregate_metrics(
    eval_dirs: list[Path],
) -> dict[str, list[TaskMetrics]]:
    """Aggregate TaskMetrics across multiple runs, grouped by task name.

    Returns dict of {task_name: [TaskMetrics from run1, run2, ...]}
    """
    task_metrics: dict[str, list[TaskMetrics]] = defaultdict(list)

    for eval_dir in eval_dirs:
        metrics = load_task_metrics(eval_dir)
        for m in metrics:
            if m.task_name in EXCLUDED_TASKS:
                continue
            task_metrics[m.task_name].append(m)

    return dict(task_metrics)


def compute_metrics_comparison(
    baseline_metrics: dict[str, list[TaskMetrics]],
    skills_metrics: dict[str, list[TaskMetrics]],
) -> dict:
    """Compare aggregated metrics between baseline and skills."""
    # Flatten all metrics
    baseline_all = [m for metrics in baseline_metrics.values() for m in metrics]
    skills_all = [m for metrics in skills_metrics.values() for m in metrics]

    if not baseline_all or not skills_all:
        return {"error": "No metrics data available"}

    # Compute aggregates
    baseline_input = [m.input_tokens for m in baseline_all]
    skills_input = [m.input_tokens for m in skills_all]

    baseline_output = [m.output_tokens for m in baseline_all]
    skills_output = [m.output_tokens for m in skills_all]

    baseline_total = [m.input_tokens + m.output_tokens for m in baseline_all]
    skills_total = [m.input_tokens + m.output_tokens for m in skills_all]

    baseline_cost = [
        calculate_cost(m.input_tokens, m.output_tokens, m.cache_tokens)
        for m in baseline_all
    ]
    skills_cost = [
        calculate_cost(m.input_tokens, m.output_tokens, m.cache_tokens)
        for m in skills_all
    ]

    baseline_time = [m.execution_time_sec for m in baseline_all]
    skills_time = [m.execution_time_sec for m in skills_all]

    baseline_steps = [m.n_steps for m in baseline_all]
    skills_steps = [m.n_steps for m in skills_all]

    return {
        "n_baseline_trials": len(baseline_all),
        "n_skills_trials": len(skills_all),
        "input_tokens": {
            "baseline_mean": safe_mean(baseline_input),
            "skills_mean": safe_mean(skills_input),
            "diff_pct": pct_diff(safe_mean(skills_input), safe_mean(baseline_input)),
        },
        "output_tokens": {
            "baseline_mean": safe_mean(baseline_output),
            "skills_mean": safe_mean(skills_output),
            "diff_pct": pct_diff(safe_mean(skills_output), safe_mean(baseline_output)),
        },
        "total_tokens": {
            "baseline_mean": safe_mean(baseline_total),
            "skills_mean": safe_mean(skills_total),
            "baseline_sum": sum(baseline_total),
            "skills_sum": sum(skills_total),
            "diff_pct": pct_diff(safe_mean(skills_total), safe_mean(baseline_total)),
        },
        "cost_usd": {
            "baseline_mean": safe_mean(baseline_cost),
            "skills_mean": safe_mean(skills_cost),
            "baseline_total": sum(baseline_cost),
            "skills_total": sum(skills_cost),
            "diff_pct": pct_diff(safe_mean(skills_cost), safe_mean(baseline_cost)),
        },
        "execution_time_sec": {
            "baseline_mean": safe_mean(baseline_time),
            "skills_mean": safe_mean(skills_time),
            "baseline_total": sum(baseline_time),
            "skills_total": sum(skills_time),
            "diff_pct": pct_diff(safe_mean(skills_time), safe_mean(baseline_time)),
        },
        "n_steps": {
            "baseline_mean": safe_mean(baseline_steps),
            "skills_mean": safe_mean(skills_steps),
            "baseline_total": sum(baseline_steps),
            "skills_total": sum(skills_steps),
            "diff_pct": pct_diff(safe_mean(skills_steps), safe_mean(baseline_steps)),
        },
    }


# =============================================================================
# Skill Usage Analysis
# =============================================================================


def analyze_skill_usage_aggregate(
    eval_dirs: list[Path],
) -> dict:
    """Analyze skill usage patterns across multiple runs.

    Returns:
        - tasks_with_skills: set of task names that used skills
        - skill_usage_by_task: {task_name: [(run_name, [skill_names])]}
        - skill_frequency: {skill_name: count}
        - total_skill_reads: int
        - n_tasks_analyzed: int
    """
    skill_usage_by_task: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    skill_frequency: dict[str, int] = defaultdict(int)
    total_skill_reads = 0
    n_tasks_analyzed = 0

    for eval_dir in eval_dirs:
        run_name = eval_dir.name
        rollout_files = find_rollout_files(eval_dir)

        for rollout_path in rollout_files:
            result, _ = analyze_rollout(rollout_path)

            task_name = extract_task_name(result.get("task", "unknown"))

            if task_name in EXCLUDED_TASKS:
                continue

            n_tasks_analyzed += 1
            skills_used = result.get("skills_used", {})

            if skills_used:
                skill_names = list(skills_used.keys())
                skill_usage_by_task[task_name].append((run_name, skill_names))
                total_skill_reads += len(skill_names)
                for skill in skill_names:
                    skill_frequency[skill] += 1

    tasks_with_skills = set(skill_usage_by_task.keys())

    return {
        "tasks_with_skills": tasks_with_skills,
        "skill_usage_by_task": dict(skill_usage_by_task),
        "skill_frequency": dict(skill_frequency),
        "total_skill_reads": total_skill_reads,
        "n_tasks_analyzed": n_tasks_analyzed,
    }


def compute_skill_effectiveness(
    task_results: dict[str, list[float]],
    skill_usage: dict,
) -> dict:
    """Compare success rates for tasks with vs without skill usage."""
    tasks_with_skills = skill_usage.get("tasks_with_skills", set())
    skill_usage_by_task = skill_usage.get("skill_usage_by_task", {})

    # Calculate success for tasks WITH skill usage
    with_skill_successes = 0
    with_skill_total = 0
    for task_name in tasks_with_skills:
        if task_name in task_results:
            rewards = task_results[task_name]
            with_skill_successes += sum(1 for r in rewards if r == 1.0)
            with_skill_total += len(rewards)

    # Calculate success for tasks WITHOUT skill usage
    without_skill_successes = 0
    without_skill_total = 0
    for task_name, rewards in task_results.items():
        if task_name not in tasks_with_skills:
            without_skill_successes += sum(1 for r in rewards if r == 1.0)
            without_skill_total += len(rewards)

    with_skill_rate = with_skill_successes / with_skill_total if with_skill_total else 0
    without_skill_rate = (
        without_skill_successes / without_skill_total if without_skill_total else 0
    )

    return {
        "with_skill": {
            "n_tasks": len(tasks_with_skills),
            "n_trials": with_skill_total,
            "n_successes": with_skill_successes,
            "success_rate": with_skill_rate,
        },
        "without_skill": {
            "n_tasks": len(task_results) - len(tasks_with_skills),
            "n_trials": without_skill_total,
            "n_successes": without_skill_successes,
            "success_rate": without_skill_rate,
        },
        "skill_usage_by_task": skill_usage_by_task,
    }

"""Loading, parsing, and discovery helpers for condition comparison."""

import json
import logging
import re
import statistics
from math import comb
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Tasks excluded from calculations (e.g. persistent infra errors).
EXCLUDED_TASKS: set[str] = {"train-fasttext"}

# gpt-5-mini pricing (USD per token).
_INPUT_PRICE = 0.250 / 1_000_000
_CACHED_PRICE = 0.025 / 1_000_000
_OUTPUT_PRICE = 2.000 / 1_000_000


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------


class TaskReward(BaseModel, frozen=True):
    """Per-task reward data across repeated runs."""

    task: str
    rewards: list[float]  # one per run
    errors: list[bool]  # one per run
    steps: list[int]  # one per run (from trajectory)
    costs: list[float]  # one per run (USD, computed from tokens)

    @property
    def mean_reward(self) -> float:
        return statistics.mean(self.rewards) if self.rewards else 0.0

    @property
    def mean_steps(self) -> float:
        return statistics.mean(self.steps) if self.steps else 0.0

    @property
    def mean_cost(self) -> float:
        return statistics.mean(self.costs) if self.costs else 0.0

    def pass_count(self, threshold: float) -> int:
        return sum(1 for r in self.rewards if r >= threshold)

    def pass_at_k(self, threshold: float, k: int) -> float:
        """Unbiased Pass@k: P(at least 1 of k sampled runs passes)."""
        n = len(self.rewards)
        c = self.pass_count(threshold)
        if n < k:
            return float(c > 0)
        return 1.0 - comb(n - c, k) / comb(n, k)

    def pass_pow_k(self, threshold: float, k: int) -> float:
        """Strict Pass^k: P(all k sampled runs pass)."""
        n = len(self.rewards)
        c = self.pass_count(threshold)
        if n < k:
            return float(c == n)
        return comb(c, k) / comb(n, k)


class ConditionResults(BaseModel, frozen=True):
    """Aggregated results for one experimental condition."""

    label: str
    runs: list[Path]
    task_rewards: dict[str, TaskReward]

    @property
    def all_tasks(self) -> set[str]:
        return set(self.task_rewards.keys())

    def per_run_pass_rates(self, threshold: float) -> list[float]:
        """Pass rate for each individual run."""
        rates: list[float] = []
        for run_idx in range(len(self.runs)):
            scored = 0
            passed = 0
            for tr in self.task_rewards.values():
                if run_idx < len(tr.errors) and not tr.errors[run_idx]:
                    scored += 1
                    if tr.rewards[run_idx] >= threshold:
                        passed += 1
            rates.append(passed / scored if scored else 0.0)
        return rates

    def mean_pass_rate(self, threshold: float) -> float:
        rates = self.per_run_pass_rates(threshold)
        return statistics.mean(rates) if rates else 0.0

    def std_pass_rate(self, threshold: float) -> float:
        rates = self.per_run_pass_rates(threshold)
        return statistics.stdev(rates) if len(rates) > 1 else 0.0

    def mean_reward(self) -> float:
        means = [tr.mean_reward for tr in self.task_rewards.values()]
        return statistics.mean(means) if means else 0.0

    def avg_passed(self, threshold: float) -> float:
        counts = [
            sum(
                1
                for tr in self.task_rewards.values()
                if run_idx < len(tr.rewards) and tr.rewards[run_idx] >= threshold
            )
            for run_idx in range(len(self.runs))
        ]
        return statistics.mean(counts) if counts else 0.0

    def avg_errored(self) -> float:
        counts = [
            sum(
                1
                for tr in self.task_rewards.values()
                if run_idx < len(tr.errors) and tr.errors[run_idx]
            )
            for run_idx in range(len(self.runs))
        ]
        return statistics.mean(counts) if counts else 0.0

    def mean_pass_at_k(self, threshold: float, k: int) -> float:
        """Mean Pass@k across all tasks."""
        values = [tr.pass_at_k(threshold, k) for tr in self.task_rewards.values()]
        return statistics.mean(values) if values else 0.0

    def mean_pass_pow_k(self, threshold: float, k: int) -> float:
        """Mean Pass^k across all tasks."""
        values = [tr.pass_pow_k(threshold, k) for tr in self.task_rewards.values()]
        return statistics.mean(values) if values else 0.0

    def avg_steps_per_task(self) -> float:
        """Mean steps/task across all runs."""
        means = [tr.mean_steps for tr in self.task_rewards.values()]
        return statistics.mean(means) if means else 0.0

    def avg_cost_per_task(self) -> float:
        """Mean cost (USD) per task across all runs."""
        means = [tr.mean_cost for tr in self.task_rewards.values()]
        return statistics.mean(means) if means else 0.0


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------


def _extract_task_name(trial_key: str) -> str:
    """Extract task name from trial key like 'chess-best-move__WxpWhHJ'."""
    return trial_key.rsplit("__", 1)[0]


def _tok(fm: dict[str, object], key: str) -> int:
    """Safely extract an integer token count from metrics."""
    v = fm.get(key)
    return int(v) if isinstance(v, (int, float)) else 0


def _compute_cost(final_metrics: dict[str, object]) -> float:
    """Compute USD cost from trajectory token counts."""
    prompt = _tok(final_metrics, "total_prompt_tokens")
    cached = _tok(final_metrics, "total_cached_tokens")
    completion = _tok(final_metrics, "total_completion_tokens")
    uncached = prompt - cached
    return uncached * _INPUT_PRICE + cached * _CACHED_PRICE + completion * _OUTPUT_PRICE


def _parse_run(result_path: Path) -> dict[str, tuple[float, bool]]:
    """Parse a single result.json → {task: (reward, is_error)}."""
    data = json.loads(result_path.read_text(encoding="utf-8"))
    tasks: dict[str, tuple[float, bool]] = {}

    for _eval_name, eval_data in data["stats"]["evals"].items():
        reward_stats = eval_data.get("reward_stats", {}).get("reward", {})
        for reward_str, trial_keys in reward_stats.items():
            reward = float(reward_str)
            for trial_key in trial_keys:
                name = _extract_task_name(trial_key)
                if name not in EXCLUDED_TASKS:
                    tasks[name] = (reward, False)

        exception_stats = eval_data.get("exception_stats", {})
        for _err_type, trial_keys in exception_stats.items():
            for trial_key in trial_keys:
                name = _extract_task_name(trial_key)
                if name not in EXCLUDED_TASKS and name not in tasks:
                    tasks[name] = (0.0, True)

    return tasks


def _parse_trajectories(run_dir: Path) -> dict[str, tuple[int, float]]:
    """Parse trajectory data from run directory → {task: (steps, cost_usd)}."""
    tasks: dict[str, tuple[int, float]] = {}
    for task_dir in run_dir.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = _extract_task_name(task_dir.name)
        if task_name in EXCLUDED_TASKS:
            continue
        traj = task_dir / "agent" / "trajectory.json"
        if not traj.exists():
            continue
        data = json.loads(traj.read_text(encoding="utf-8"))
        fm = data.get("final_metrics") or {}
        steps = int(fm.get("total_steps") or 0)
        cost = _compute_cost(fm)
        tasks[task_name] = (steps, cost)
    return tasks


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------


def discover_runs(eval_dir: Path, condition: str, *, prefix: str = "sk") -> list[Path]:
    """Find run directories whose extracted condition exactly matches."""
    pattern = f"{prefix}-{condition}-*"
    return sorted(
        p
        for p in eval_dir.glob(pattern)
        if p.is_dir()
        and (p / "result.json").exists()
        and _extract_condition(p.name, prefix) == condition
    )


_TIMESTAMP_RE = re.compile(r"-\d{8}-\d{6}(?:-.+)?$")
_EFFORT_LEVELS = frozenset({"high", "medium", "low", "mini"})


def _extract_condition(dirname: str, prefix: str) -> str | None:
    """Extract condition from ``{prefix}-{cond}-{model}[-{effort}]-{ts}``."""
    if not dirname.startswith(f"{prefix}-"):
        return None
    rest = _TIMESTAMP_RE.sub("", dirname[len(prefix) + 1 :])
    parts = rest.split("-")
    n_strip = 2 if len(parts) >= 3 and parts[-1] in _EFFORT_LEVELS else 1
    return "-".join(parts[:-n_strip]) if len(parts) > n_strip else None


def discover_all_conditions(eval_dir: Path, *, prefix: str = "sk") -> list[str]:
    """Auto-discover condition names from run directories."""
    conditions: set[str] = set()
    for p in sorted(eval_dir.iterdir()):
        if not p.is_dir() or not (p / "result.json").exists():
            continue
        cond = _extract_condition(p.name, prefix)
        if cond:
            conditions.add(cond)
    return sorted(conditions)


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def load_condition(
    eval_dir: Path,
    condition: str,
    label: str,
    *,
    prefix: str = "sk",
) -> ConditionResults:
    """Load all runs for a condition and aggregate."""
    runs = discover_runs(eval_dir, condition, prefix=prefix)
    if not runs:
        logger.warning("No runs found for condition '%s' in %s", condition, eval_dir)

    all_task_names: set[str] = set()
    run_data: list[dict[str, tuple[float, bool]]] = []
    traj_data: list[dict[str, tuple[int, float]]] = []
    for run_dir in runs:
        parsed = _parse_run(run_dir / "result.json")
        run_data.append(parsed)
        traj_data.append(_parse_trajectories(run_dir))
        all_task_names.update(parsed.keys())

    task_rewards: dict[str, TaskReward] = {}
    for task in sorted(all_task_names):
        rewards: list[float] = []
        errors: list[bool] = []
        steps_list: list[int] = []
        costs_list: list[float] = []
        for rd, td in zip(run_data, traj_data, strict=True):
            if task in rd:
                reward, is_error = rd[task]
                rewards.append(reward)
                errors.append(is_error)
            else:
                rewards.append(0.0)
                errors.append(True)
            if task in td:
                s, c = td[task]
                steps_list.append(s)
                costs_list.append(c)
            else:
                steps_list.append(0)
                costs_list.append(0.0)
        task_rewards[task] = TaskReward(
            task=task,
            rewards=rewards,
            errors=errors,
            steps=steps_list,
            costs=costs_list,
        )

    return ConditionResults(label=label, runs=runs, task_rewards=task_rewards)

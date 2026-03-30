"""Compare task-level results between two Harbor evaluation runs.

Loads two Harbor result.json files and prints:
  1. Aggregate pass rates side-by-side
  2. Per-task win/loss/tie breakdown
  3. Specific tasks that improved, regressed, or errored

Usage::

    uv run python -m analysis.ablation.compare_runs \\
        outputs/evaluation/tb-baseline-gpt5mini-high-20260225-211854 \\
        outputs/evaluation/tb-skillflow-injection-gpt5mini-high-20260225-205641

    # With short labels
    uv run python -m analysis.ablation.compare_runs \\
        outputs/evaluation/tb-baseline-... \\
        outputs/evaluation/tb-skillflow-... \\
        --label-a baseline --label-b skillflow
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Tasks excluded from pass-rate calculations (e.g. persistent infra errors).
EXCLUDED_TASKS: set[str] = {"train-fasttext"}


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------


class RunResults(BaseModel, frozen=True):
    """Parsed Harbor result.json for comparison."""

    label: str
    passed: set[str]
    failed: set[str]
    errored: set[str]

    @property
    def all_tasks(self) -> set[str]:
        return self.passed | self.failed | self.errored

    @property
    def n_scored(self) -> int:
        return len(self.passed) + len(self.failed)

    @property
    def pass_rate(self) -> float:
        return len(self.passed) / self.n_scored if self.n_scored else 0.0


class TaskDelta(BaseModel, frozen=True):
    """Per-task change between two runs."""

    task: str
    status_a: str  # "pass", "fail", "error", "missing"
    status_b: str


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------


def _extract_task_name(trial_key: str) -> str:
    """Extract task name from trial key like 'chess-best-move__WxpWhHJ'."""
    return trial_key.rsplit("__", 1)[0]


def load_run(path: Path, label: str) -> RunResults:
    """Load a Harbor result.json into RunResults."""
    result_file = path / "result.json" if path.is_dir() else path
    data = json.loads(result_file.read_text(encoding="utf-8"))

    passed: set[str] = set()
    failed: set[str] = set()
    errored: set[str] = set()

    for _eval_name, eval_data in data["stats"]["evals"].items():
        reward_stats = eval_data.get("reward_stats", {}).get("reward", {})
        for trial_key in reward_stats.get("1.0", []):
            passed.add(_extract_task_name(trial_key))
        for trial_key in reward_stats.get("0.0", []):
            failed.add(_extract_task_name(trial_key))

        exception_stats = eval_data.get("exception_stats", {})
        for _err_type, trial_keys in exception_stats.items():
            for trial_key in trial_keys:
                errored.add(_extract_task_name(trial_key))

    return RunResults(
        label=label,
        passed=passed - EXCLUDED_TASKS,
        failed=failed - EXCLUDED_TASKS,
        errored=errored - EXCLUDED_TASKS,
    )


# ------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------


def _task_status(task: str, run: RunResults) -> str:
    if task in run.passed:
        return "passed"
    if task in run.failed:
        return "fail"
    if task in run.errored:
        return "error"
    return "missing"


def build_deltas(a: RunResults, b: RunResults) -> list[TaskDelta]:
    all_tasks = sorted(a.all_tasks | b.all_tasks)
    return [
        TaskDelta(
            task=t,
            status_a=_task_status(t, a),
            status_b=_task_status(t, b),
        )
        for t in all_tasks
    ]


# ------------------------------------------------------------------
# Printing
# ------------------------------------------------------------------

_STATUS_SYMBOL = {"passed": "+", "fail": "-", "error": "!", "missing": "?"}


def print_summary(a: RunResults, b: RunResults) -> None:
    w = max(len(a.label), len(b.label), 10)
    print(f"\n{'=' * (w + 40)}")
    print(f"{'':15} {a.label:>{w}}  {b.label:>{w}}")
    print(f"{'-' * (w + 40)}")
    print(f"{'Passed':15} {len(a.passed):>{w}}  {len(b.passed):>{w}}")
    print(f"{'Failed':15} {len(a.failed):>{w}}  {len(b.failed):>{w}}")
    print(f"{'Errored':15} {len(a.errored):>{w}}  {len(b.errored):>{w}}")
    print(f"{'Total scored':15} {a.n_scored:>{w}}  {b.n_scored:>{w}}")
    print(f"{'Pass rate':15} {a.pass_rate:>{w}.1%}  {b.pass_rate:>{w}.1%}")
    delta = b.pass_rate - a.pass_rate
    sign = "+" if delta > 0 else ""
    print(f"{'Delta':15} {'':>{w}}  {sign}{delta:>{w}.1%}")
    print(f"{'=' * (w + 40)}")


def print_changes(deltas: list[TaskDelta], a: RunResults, b: RunResults) -> None:
    improved = [d for d in deltas if d.status_a != "passed" and d.status_b == "passed"]
    regressed = [d for d in deltas if d.status_a == "passed" and d.status_b != "passed"]
    same_pass = [d for d in deltas if d.status_a == "passed" and d.status_b == "passed"]
    same_fail = [
        d for d in deltas if d.status_a == d.status_b and d.status_a != "passed"
    ]

    print(f"\nWin/Loss/Tie ({a.label} → {b.label}):")
    print(f"  Improved (fail→pass): {len(improved)}")
    print(f"  Regressed (pass→fail/err): {len(regressed)}")
    print(f"  Same (both pass):     {len(same_pass)}")
    print(f"  Same (both fail/err): {len(same_fail)}")

    if improved:
        print(f"\n  Improved tasks ({a.label}→{b.label}):")
        for d in improved:
            sym_a = _STATUS_SYMBOL[d.status_a]
            print(f"    [{sym_a}→+] {d.task}")

    if regressed:
        print(f"\n  Regressed tasks ({a.label}→{b.label}):")
        for d in regressed:
            sym_b = _STATUS_SYMBOL[d.status_b]
            print(f"    [+→{sym_b}] {d.task}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare task-level results between two Harbor runs",
    )
    parser.add_argument("run_a", type=Path, help="Path to first run directory")
    parser.add_argument("run_b", type=Path, help="Path to second run directory")
    parser.add_argument("--label-a", type=str, default=None, help="Label for run A")
    parser.add_argument("--label-b", type=str, default=None, help="Label for run B")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    label_a = args.label_a or args.run_a.name
    label_b = args.label_b or args.run_b.name

    a = load_run(args.run_a, label_a)
    b = load_run(args.run_b, label_b)

    deltas = build_deltas(a, b)
    print_summary(a, b)
    print_changes(deltas, a, b)
    return 0


if __name__ == "__main__":
    sys.exit(main())

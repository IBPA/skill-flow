"""Compare task-level results across multiple experimental conditions.

Aggregates repeated runs per condition and prints:
  1. Aggregate summary table (pass rate, mean reward, error counts)
  2. Per-task reward matrix sorted by best-performing condition
  3. Pairwise delta analysis (unique wins, regressions, shared gains)

Usage::

    # Auto-discover all conditions from sk-* directories
    uv run python -m analysis.ablation.compare_conditions outputs/evaluation

    # Explicit conditions with labels
    uv run python -m analysis.ablation.compare_conditions \\
        outputs/evaluation \\
        --conditions baseline skillflow-inject skillsbench-inject \\
        --labels baseline skillflow skillsbench
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analysis.comparison.utils.display_conditions import (
    print_pairwise_deltas,
    print_summary,
    print_task_matrix,
)
from analysis.comparison.utils.display_stats import (
    print_significance,
    print_summary_with_ci,
)
from analysis.comparison.utils.loader import (
    ConditionResults,
    TaskReward,
    discover_all_conditions,
    load_condition,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Alignment
# ------------------------------------------------------------------


def align_conditions(conditions: list[ConditionResults]) -> list[ConditionResults]:
    """Pad all conditions to the same task set for consistent denominators.

    Tasks missing from a condition are filled with zero-reward error entries.
    """
    all_tasks: set[str] = set()
    for c in conditions:
        all_tasks.update(c.all_tasks)

    aligned: list[ConditionResults] = []
    for c in conditions:
        missing = all_tasks - c.all_tasks
        if not missing:
            aligned.append(c)
            continue
        updated = dict(c.task_rewards)
        n_runs = len(c.runs)
        for task in missing:
            updated[task] = TaskReward(
                task=task,
                rewards=[0.0] * n_runs,
                errors=[True] * n_runs,
                steps=[0] * n_runs,
                costs=[0.0] * n_runs,
            )
        aligned.append(c.model_copy(update={"task_rewards": updated}))
    return aligned


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare task-level results across multiple conditions",
    )
    parser.add_argument(
        "eval_dir",
        type=Path,
        help="Directory containing all run folders",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Condition prefixes to match (default: auto-discover from eval_dir)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Short display labels (defaults to condition names)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Reward threshold for 'pass' (default: %(default)s)",
    )
    parser.add_argument(
        "--prefix",
        default="sk",
        help="Run directory prefix (default: %(default)s, use 'tb' for terminal-bench)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show bootstrap 95%% CIs and significance tests",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cond_names = args.conditions or discover_all_conditions(
        args.eval_dir, prefix=args.prefix
    )
    if not cond_names:
        logger.error(
            "No conditions found in %s with prefix '%s'",
            args.eval_dir,
            args.prefix,
        )
        return 1

    labels = args.labels or cond_names
    if len(labels) != len(cond_names):
        logger.error(
            "Number of labels (%d) must match conditions (%d)",
            len(labels),
            len(cond_names),
        )
        return 1

    conditions = [
        load_condition(args.eval_dir, cond, label, prefix=args.prefix)
        for cond, label in zip(cond_names, labels, strict=True)
    ]

    for c in conditions:
        if not c.runs:
            logger.error("No runs found for condition '%s'", c.label)
            return 1
        logger.info("  %s: %d runs, %d tasks", c.label, len(c.runs), len(c.all_tasks))

    conditions = align_conditions(conditions)
    logger.info("  Aligned to %d tasks", len(conditions[0].all_tasks))

    if args.stats:
        print_summary_with_ci(conditions, args.threshold)
    else:
        print_summary(conditions, args.threshold)
    print_task_matrix(conditions)

    if len(conditions) >= 2:
        print_pairwise_deltas(conditions[0], conditions[1:], args.threshold)
        if args.stats:
            print_significance(conditions[0], conditions[1:], args.threshold)

    return 0


if __name__ == "__main__":
    sys.exit(main())

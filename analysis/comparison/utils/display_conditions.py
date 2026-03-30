"""Display helpers for multi-condition comparison output."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.comparison.utils.loader import ConditionResults


def print_summary(
    conditions: list[ConditionResults],
    threshold: float,
) -> None:
    """Print aggregate summary table."""
    col_w = max((len(c.label) for c in conditions), default=12)
    col_w = max(col_w, 12)
    header_w = 20
    sep = "-" * (header_w + (col_w + 2) * len(conditions))

    print("\n1. Aggregate Summary")
    print(sep)
    row_header = f"{'':>{header_w}}"
    for c in conditions:
        row_header += f"  {c.label:>{col_w}}"
    print(row_header)
    print(sep)

    _print_row("Runs", [len(c.runs) for c in conditions], header_w, col_w, fmt="d")
    _print_row(
        "Pass rate (mean)",
        [c.mean_pass_rate(threshold) for c in conditions],
        header_w,
        col_w,
        fmt=".1%",
    )
    _print_row(
        "Pass rate (std)",
        [c.std_pass_rate(threshold) for c in conditions],
        header_w,
        col_w,
        fmt=".1%",
    )
    for k in (1, 2, 3):
        _print_row(
            f"Pass@{k}",
            [c.mean_pass_at_k(threshold, k) for c in conditions],
            header_w,
            col_w,
            fmt=".1%",
        )
    for k in (1, 2, 3):
        _print_row(
            f"Pass^{k}",
            [c.mean_pass_pow_k(threshold, k) for c in conditions],
            header_w,
            col_w,
            fmt=".1%",
        )
    _print_row(
        "Mean reward",
        [c.mean_reward() for c in conditions],
        header_w,
        col_w,
        fmt=".3f",
    )
    _print_row(
        "Passed (avg)",
        [c.avg_passed(threshold) for c in conditions],
        header_w,
        col_w,
        fmt=".1f",
    )
    _print_row(
        "Errored (avg)",
        [c.avg_errored() for c in conditions],
        header_w,
        col_w,
        fmt=".1f",
    )
    _print_row(
        "Steps/task (avg)",
        [c.avg_steps_per_task() for c in conditions],
        header_w,
        col_w,
        fmt=".1f",
    )
    _print_cost_row(
        "Cost $/task (avg)",
        [c.avg_cost_per_task() for c in conditions],
        header_w,
        col_w,
    )
    print(sep)


def _print_row(
    label: str,
    values: list[int] | list[float],
    header_w: int,
    col_w: int,
    *,
    fmt: str = "",
) -> None:
    row = f"{label:>{header_w}}"
    for v in values:
        cell = f"{v:{fmt}}"
        row += f"  {cell:>{col_w}}"
    print(row)


def _print_cost_row(
    label: str,
    values: list[float],
    header_w: int,
    col_w: int,
) -> None:
    row = f"{label:>{header_w}}"
    for v in values:
        cell = f"${v:.4f}"
        row += f"  {cell:>{col_w}}"
    print(row)


def print_task_matrix(
    conditions: list[ConditionResults],
) -> None:
    """Print per-task reward matrix sorted by first non-baseline condition."""
    all_tasks: set[str] = set()
    for c in conditions:
        all_tasks.update(c.all_tasks)

    rows: list[tuple[str, list[float]]] = []
    for task in all_tasks:
        means = [
            tr.mean_reward if (tr := c.task_rewards.get(task)) else 0.0
            for c in conditions
        ]
        rows.append((task, means))

    sort_col = 1 if len(conditions) > 1 else 0
    rows.sort(key=lambda r: (-r[1][sort_col], -r[1][0], r[0]))

    task_w = max((len(r[0]) for r in rows), default=20)
    task_w = max(task_w, 8)
    col_w = max((len(c.label) for c in conditions), default=12)
    col_w = max(col_w, 12)

    print("\n2. Per-Task Reward Matrix")
    header = f"  {'task':<{task_w}}"
    for c in conditions:
        header += f"  {c.label:>{col_w}}"
    print(header)

    for task, means in rows:
        row = f"  {task:<{task_w}}"
        for m in means:
            row += f"  {m:>{col_w}.2f}"
        print(row)


def _classify_tasks(
    baseline: ConditionResults,
    cond: ConditionResults,
    all_tasks: set[str],
    threshold: float,
) -> tuple[set[str], set[str]]:
    """Return (improved, regressed) task sets for cond vs baseline."""
    improved: set[str] = set()
    regressed: set[str] = set()
    for task in all_tasks:
        base_tr = baseline.task_rewards.get(task)
        cond_tr = cond.task_rewards.get(task)
        base_mean = base_tr.mean_reward if base_tr else 0.0
        cond_mean = cond_tr.mean_reward if cond_tr else 0.0
        if base_mean < threshold <= cond_mean:
            improved.add(task)
        elif cond_mean < threshold <= base_mean:
            regressed.add(task)
    return improved, regressed


def print_pairwise_deltas(
    baseline: ConditionResults,
    others: list[ConditionResults],
    threshold: float,
) -> None:
    """Print pairwise delta analysis vs baseline."""
    print(f"\n3. Pairwise Deltas (vs {baseline.label})")

    all_tasks = set(baseline.all_tasks)
    for c in others:
        all_tasks.update(c.all_tasks)

    improved_by: dict[str, set[str]] = {}
    regressed_by: dict[str, set[str]] = {}
    for c in others:
        imp, reg = _classify_tasks(baseline, c, all_tasks, threshold)
        improved_by[c.label] = imp
        regressed_by[c.label] = reg

    if len(others) >= 2:
        _print_two_way_deltas(others, improved_by, regressed_by)
    else:
        _print_single_deltas(baseline, others, improved_by, regressed_by)


def _print_two_way_deltas(
    others: list[ConditionResults],
    improved_by: dict[str, set[str]],
    regressed_by: dict[str, set[str]],
) -> None:
    a_label, b_label = others[0].label, others[1].label
    a_imp, b_imp = improved_by[a_label], improved_by[b_label]
    a_reg, b_reg = regressed_by[a_label], regressed_by[b_label]

    unique_a = sorted(a_imp - b_imp)
    unique_b = sorted(b_imp - a_imp)
    both_improved = sorted(a_imp & b_imp)
    any_regressed = sorted(a_reg | b_reg)

    print(f"\n  Unique wins for {a_label} ({len(unique_a)}):")
    for t in unique_a:
        print(f"    {t}")

    print(f"\n  Unique wins for {b_label} ({len(unique_b)}):")
    for t in unique_b:
        print(f"    {t}")

    print(f"\n  Improved by both ({len(both_improved)}):")
    for t in both_improved:
        print(f"    {t}")

    print(f"\n  Regressed by either ({len(any_regressed)}):")
    for t in any_regressed:
        ra = "Y" if t in a_reg else " "
        rb = "Y" if t in b_reg else " "
        print(f"    [{a_label[0]}:{ra} {b_label[0]}:{rb}] {t}")


def _print_single_deltas(
    baseline: ConditionResults,
    others: list[ConditionResults],
    improved_by: dict[str, set[str]],
    regressed_by: dict[str, set[str]],
) -> None:
    for c in others:
        improved = sorted(improved_by[c.label])
        regressed = sorted(regressed_by[c.label])
        print(f"\n  {baseline.label} → {c.label}:")
        print(f"    Improved: {len(improved)}")
        for t in improved:
            print(f"      {t}")
        print(f"    Regressed: {len(regressed)}")
        for t in regressed:
            print(f"      {t}")

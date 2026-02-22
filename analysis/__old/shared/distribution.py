"""Task success distribution analysis functions."""


def compute_task_stats(task_results: dict[str, list[float]]) -> dict[str, dict]:
    """Compute success stats for each task.

    Args:
        task_results: Dict mapping task name to list of reward values (0.0 or 1.0)

    Returns:
        Dict mapping task name to stats dict with keys: total, successes, rate
    """
    stats = {}
    for task_name, rewards in task_results.items():
        total = len(rewards)
        successes = sum(1 for r in rewards if r == 1.0)
        stats[task_name] = {
            "total": total,
            "successes": successes,
            "rate": successes / total if total > 0 else 0,
        }
    return stats


def print_distribution(task_results: dict[str, list[float]], prefix: str) -> None:
    """Print task success distribution analysis for a single prefix."""
    task_stats = compute_task_stats(task_results)

    # Categorize by consistency
    always_success: list[str] = []
    mostly_success: list[str] = []
    mixed: list[str] = []
    mostly_failure: list[str] = []
    always_failure: list[str] = []

    for task_name, stats in task_stats.items():
        rate = stats["rate"]
        if rate == 1.0:
            always_success.append(task_name)
        elif rate >= 0.75:
            mostly_success.append(task_name)
        elif rate >= 0.25:
            mixed.append(task_name)
        elif rate > 0:
            mostly_failure.append(task_name)
        else:
            always_failure.append(task_name)

    # Print header
    print("=" * 70)
    print(f"TASK SUCCESS DISTRIBUTION: {prefix}*")
    print("=" * 70)
    print(f"Total tasks: {len(task_stats)}")
    n_runs = max(len(r) for r in task_results.values()) if task_results else 0
    print(f"Runs analyzed: {n_runs}")
    print()

    # Summary
    print(f"ALWAYS SUCCESSFUL (100%):      {len(always_success):3d} tasks")
    print(f"MOSTLY SUCCESSFUL (75-99%):    {len(mostly_success):3d} tasks")
    print(f"MIXED (25-75%):                {len(mixed):3d} tasks")
    print(f"MOSTLY FAILING (1-25%):        {len(mostly_failure):3d} tasks")
    print(f"ALWAYS FAILING (0%):           {len(always_failure):3d} tasks")
    print()

    # Detailed breakdown
    for label, tasks in [
        ("ALWAYS SUCCESSFUL (100%)", always_success),
        ("MOSTLY SUCCESSFUL (75-99%)", mostly_success),
        ("MIXED (25-75%)", mixed),
        ("MOSTLY FAILING (1-25%)", mostly_failure),
        ("ALWAYS FAILING (0%)", always_failure),
    ]:
        if not tasks:
            continue

        print("-" * 70)
        print(f"{label} ({len(tasks)} tasks)")
        print("-" * 70)

        for task_name in sorted(tasks):
            stats = task_stats[task_name]
            rate = float(stats["rate"])
            succ = int(stats["successes"])
            total = int(stats["total"])
            print(f"  {task_name:45s} | {succ:2d}/{total:2d} ({rate * 100:5.1f}%)")
        print()

    print("=" * 70)


def _categorize_tasks(
    all_tasks: list[str],
    baseline_stats: dict[str, dict],
    skills_stats: dict[str, dict],
) -> dict[str, list[tuple]]:
    """Categorize tasks by their change between baseline and skills.

    Returns dict with keys: improved, regressed, stable_success, stable_failure, mixed
    """
    categories: dict[str, list[tuple]] = {
        "improved": [],
        "regressed": [],
        "stable_success": [],
        "stable_failure": [],
        "mixed": [],
    }

    for task in all_tasks:
        b = baseline_stats.get(task, {"rate": 0, "successes": 0, "total": 0})
        s = skills_stats.get(task, {"rate": 0, "successes": 0, "total": 0})

        b_rate = b["rate"]
        s_rate = s["rate"]
        diff = s_rate - b_rate

        if b_rate == 1.0 and s_rate == 1.0:
            categories["stable_success"].append((task, b, s, diff))
        elif b_rate == 0.0 and s_rate == 0.0:
            categories["stable_failure"].append((task, b, s, diff))
        elif diff > 0.1:
            categories["improved"].append((task, b, s, diff))
        elif diff < -0.1:
            categories["regressed"].append((task, b, s, diff))
        else:
            categories["mixed"].append((task, b, s, diff))

    return categories


def _print_task_category(title: str, tasks: list[tuple], show_all: bool = True) -> None:
    """Print a category of tasks with their stats."""
    if not tasks:
        return

    print("-" * 85)
    print(f"{title} ({len(tasks)} tasks)")
    print("-" * 85)
    header = f"{'Task':<40s} {'Baseline':>12s} {'Skills':>12s} {'Change':>12s}"
    print(header)
    print("-" * 85)

    # Sort by diff (biggest improvement/regression first)
    sorted_tasks = sorted(tasks, key=lambda x: x[3], reverse=True)
    display = sorted_tasks if show_all else sorted_tasks[:10]

    for task, b, s, diff in display:
        b_str = f"{b['successes']}/{b['total']} ({b['rate'] * 100:5.1f}%)"
        s_str = f"{s['successes']}/{s['total']} ({s['rate'] * 100:5.1f}%)"
        diff_str = f"{diff * 100:+6.1f}%"
        print(f"  {task:<38s} {b_str:>12s} {s_str:>12s} {diff_str:>12s}")

    if not show_all and len(tasks) > 10:
        print(f"  ... and {len(tasks) - 10} more")
    print()


def print_distribution_comparison(
    baseline_results: dict[str, list[float]],
    skills_results: dict[str, list[float]],
    baseline_prefix: str,
    skills_prefix: str,
) -> None:
    """Print side-by-side task success distribution comparison."""
    baseline_stats = compute_task_stats(baseline_results)
    skills_stats = compute_task_stats(skills_results)
    all_tasks = sorted(set(baseline_stats.keys()) | set(skills_stats.keys()))

    categories = _categorize_tasks(all_tasks, baseline_stats, skills_stats)

    # Print header
    print("=" * 85)
    print("TASK SUCCESS DISTRIBUTION COMPARISON")
    print("=" * 85)
    print(f"  Baseline: {baseline_prefix}* ({len(baseline_results)} tasks)")
    print(f"  Skills:   {skills_prefix}* ({len(skills_results)} tasks)")
    print()

    # Summary
    stable_succ = len(categories["stable_success"])
    stable_fail = len(categories["stable_failure"])
    print(f"STABLE SUCCESS (100% both):    {stable_succ:3d} tasks")
    print(f"IMPROVED (skills > baseline):  {len(categories['improved']):3d} tasks")
    print(f"MIXED/UNCHANGED:               {len(categories['mixed']):3d} tasks")
    print(f"REGRESSED (skills < baseline): {len(categories['regressed']):3d} tasks")
    print(f"STABLE FAILURE (0% both):      {stable_fail:3d} tasks")
    print()

    # Print each category
    _print_task_category("IMPROVED (skills better)", categories["improved"])
    _print_task_category("REGRESSED (skills worse)", categories["regressed"])
    _print_task_category("STABLE FAILURE (always 0%)", categories["stable_failure"])
    _print_task_category("MIXED/UNCHANGED", categories["mixed"], show_all=False)
    _print_task_category(
        "STABLE SUCCESS (always 100%)", categories["stable_success"], show_all=False
    )

    print("=" * 85)

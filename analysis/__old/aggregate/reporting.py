"""Output and printing functions for aggregate comparison results."""

from pathlib import Path

from ..shared import TaskMetrics, format_tokens


def print_metrics_comparison(metrics: dict) -> None:
    """Print token usage, cost, time, and steps comparison."""
    print("\n" + "-" * 70)
    print("TOKEN USAGE & COST (aggregated across runs)")
    print("-" * 70)

    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return

    print(f"{'':20s} {'Baseline':>15s} {'Skills':>15s} {'Diff':>15s}")
    print("-" * 70)

    # Input tokens
    inp = metrics["input_tokens"]
    print(
        f"{'Input tokens (mean)':20s} {inp['baseline_mean']:>15,.0f} "
        f"{inp['skills_mean']:>15,.0f} {inp['diff_pct']:>+14.1f}%"
    )

    # Output tokens
    out = metrics["output_tokens"]
    print(
        f"{'Output tokens (mean)':20s} {out['baseline_mean']:>15,.0f} "
        f"{out['skills_mean']:>15,.0f} {out['diff_pct']:>+14.1f}%"
    )

    # Total tokens
    tot = metrics["total_tokens"]
    print(
        f"{'Total tokens (mean)':20s} {tot['baseline_mean']:>15,.0f} "
        f"{tot['skills_mean']:>15,.0f} {tot['diff_pct']:>+14.1f}%"
    )

    # Cost
    cost = metrics["cost_usd"]
    print(
        f"{'Cost USD (mean)':20s} ${cost['baseline_mean']:>14.4f} "
        f"${cost['skills_mean']:>14.4f} {cost['diff_pct']:>+14.1f}%"
    )

    # Execution time
    time = metrics["execution_time_sec"]
    print(
        f"{'Exec time (mean)':20s} {time['baseline_mean']:>14.1f}s "
        f"{time['skills_mean']:>14.1f}s {time['diff_pct']:>+14.1f}%"
    )

    # Steps
    steps = metrics["n_steps"]
    print(
        f"{'Steps (mean)':20s} {steps['baseline_mean']:>15.1f} "
        f"{steps['skills_mean']:>15.1f} {steps['diff_pct']:>+14.1f}%"
    )

    # Totals
    print()
    print(
        f"Total cost: baseline ${cost['baseline_total']:.2f}, "
        f"skills ${cost['skills_total']:.2f}"
    )
    print(
        f"Total time: baseline {time['baseline_total'] / 60:.1f} min, "
        f"skills {time['skills_total'] / 60:.1f} min"
    )


def print_skill_usage(
    skill_usage: dict,
    skill_effectiveness: dict,
    baseline_results: dict[str, list[float]],
    skills_results: dict[str, list[float]],
    baseline_metrics: dict[str, list[TaskMetrics]],
    skills_metrics: dict[str, list[TaskMetrics]],
    skill_token_analysis: dict,
) -> None:
    """Print skill usage summary and per-task breakdown with token analysis."""
    print("\n" + "-" * 70)
    print("SKILL USAGE SUMMARY (skills runs only)")
    print("-" * 70)

    n_tasks = skill_usage.get("n_tasks_analyzed", 0)
    tasks_with = len(skill_usage.get("tasks_with_skills", set()))
    total_reads = skill_usage.get("total_skill_reads", 0)

    if n_tasks > 0:
        pct = 100 * tasks_with / len(skills_results) if skills_results else 0
        n_skill_tasks = len(skills_results)
        print(f"Tasks with skill reads: {tasks_with}/{n_skill_tasks} ({pct:.1f}%)")
    else:
        print("No skill usage data available")

    print(f"Total skill read events: {total_reads}")

    # Success rate comparison
    with_s = skill_effectiveness.get("with_skill", {})
    without_s = skill_effectiveness.get("without_skill", {})

    if with_s.get("n_trials", 0) > 0 or without_s.get("n_trials", 0) > 0:
        print("\nSuccess rate comparison:")
        if with_s.get("n_trials", 0) > 0:
            print(
                f"  With skill usage:    {with_s['n_successes']}/{with_s['n_trials']} "
                f"({with_s['success_rate']:.1%})"
            )
        if without_s.get("n_trials", 0) > 0:
            n_succ = without_s["n_successes"]
            n_tot = without_s["n_trials"]
            rate = without_s["success_rate"]
            print(f"  Without skill usage: {n_succ}/{n_tot} ({rate:.1%})")

    # Skill frequency
    skill_freq = skill_usage.get("skill_frequency", {})
    if skill_freq:
        print("\nMost used skills:")
        for skill, count in sorted(skill_freq.items(), key=lambda x: -x[1])[:10]:
            print(f"  {skill}: {count} task(s)")

    # Per-task breakdown with token analysis
    _print_skill_usage_table(
        skill_effectiveness,
        baseline_results,
        skills_results,
        baseline_metrics,
        skills_metrics,
        skill_token_analysis,
    )


def _print_skill_usage_table(
    skill_effectiveness: dict,
    baseline_results: dict[str, list[float]],
    skills_results: dict[str, list[float]],
    baseline_metrics: dict[str, list[TaskMetrics]],
    skills_metrics: dict[str, list[TaskMetrics]],
    skill_token_analysis: dict,
) -> None:
    """Print per-task breakdown table with token usage."""
    print("\n" + "-" * 120)
    print("SKILL USAGE BY TASK (with token breakdown)")
    print("-" * 120)

    skill_by_task = skill_effectiveness.get("skill_usage_by_task", {})
    per_task_tokens = skill_token_analysis.get("per_task", {})

    # Build table data
    all_tasks = set(baseline_results.keys()) | set(skills_results.keys())
    table_data = _build_skill_table_data(
        all_tasks,
        baseline_results,
        skills_results,
        baseline_metrics,
        skills_metrics,
        skill_by_task,
        per_task_tokens,
    )

    # Print table header
    print(
        f"{'Task':<26s} {'Base In/Steps':>14s} {'Skill In/Steps':>15s} "
        f"{'Delta':>8s} {'SkillTok':>9s} {'Base':>5s} {'Skill':>5s}"
    )
    print("-" * 95)

    for row in table_data:
        _print_skill_table_row(row)


def _build_skill_table_data(
    all_tasks: set[str],
    baseline_results: dict[str, list[float]],
    skills_results: dict[str, list[float]],
    baseline_metrics: dict[str, list[TaskMetrics]],
    skills_metrics: dict[str, list[TaskMetrics]],
    skill_by_task: dict,
    per_task_tokens: dict,
) -> list[dict]:
    """Build table data for skill usage breakdown."""
    table_data = []

    def avg(vals: list[int]) -> float:
        return sum(vals) / len(vals) if vals else 0

    for task_name in sorted(all_tasks):
        base_rewards = baseline_results.get(task_name, [])
        base_n = len(base_rewards)
        base_success = sum(1 for r in base_rewards if r == 1.0)
        base_mets = baseline_metrics.get(task_name, [])

        skill_rewards = skills_results.get(task_name, [])
        skill_n = len(skill_rewards)
        skill_success = sum(1 for r in skill_rewards if r == 1.0)
        skill_mets = skills_metrics.get(task_name, [])

        task_token_data = per_task_tokens.get(task_name, {})
        skill_runs = skill_by_task.get(task_name, [])
        all_skills: set[str] = set()
        for _, skills in skill_runs:
            all_skills.update(skills)

        table_data.append(
            {
                "task": task_name,
                "base_in": avg([m.input_tokens for m in base_mets]),
                "base_out": avg([m.output_tokens for m in base_mets]),
                "base_steps": avg([m.n_steps for m in base_mets]),
                "skill_in": avg([m.input_tokens for m in skill_mets]),
                "skill_out": avg([m.output_tokens for m in skill_mets]),
                "skill_steps": avg([m.n_steps for m in skill_mets]),
                "skill_meta": task_token_data.get("avg_metadata_tokens", 0),
                "skill_content": task_token_data.get("avg_content_tokens", 0),
                "base_success": f"{base_success}/{base_n}" if base_n else "-",
                "skill_success": f"{skill_success}/{skill_n}" if skill_n else "-",
                "skill_names": sorted(all_skills) if all_skills else [],
            }
        )

    return table_data


def _print_skill_table_row(row: dict) -> None:
    """Print a single row of the skill usage table."""
    base_col = f"{format_tokens(row['base_in'])}/{row['base_steps']:.0f}"
    skill_col = f"{format_tokens(row['skill_in'])}/{row['skill_steps']:.0f}"
    delta = row["skill_in"] - row["base_in"]
    delta_str = f"+{format_tokens(delta)}" if delta >= 0 else format_tokens(delta)
    skill_tok = format_tokens(row["skill_meta"] + row["skill_content"])

    print(
        f"{row['task']:<26s} {base_col:>14s} {skill_col:>15s} "
        f"{delta_str:>8s} {skill_tok:>9s} {row['base_success']:>5s} "
        f"{row['skill_success']:>5s}"
    )


def print_comparison(
    baseline_prefix: str,
    skills_prefix: str,
    baseline_dirs: list[Path],
    skills_dirs: list[Path],
    baseline_stats: dict,
    skills_stats: dict,
    paired_comparison: dict,
    trial_comparison: dict,
    win_rate_comparison: dict | None = None,
) -> None:
    """Print formatted comparison results."""
    _print_comparison_header(baseline_prefix, skills_prefix, baseline_dirs, skills_dirs)
    _print_overall_statistics(baseline_stats, skills_stats)
    _print_fisher_test(trial_comparison)
    _print_mcnemar_test(paired_comparison)

    if win_rate_comparison is not None:
        _print_win_rate_test(win_rate_comparison)

    print("\n" + "=" * 70)


def _print_comparison_header(
    baseline_prefix: str,
    skills_prefix: str,
    baseline_dirs: list[Path],
    skills_dirs: list[Path],
) -> None:
    """Print comparison header with run information."""
    print("=" * 70)
    print("AGGREGATE COMPARISON: BASELINE vs SKILLS")
    print("=" * 70)

    print(f"\nBaseline prefix: {baseline_prefix}* ({len(baseline_dirs)} runs)")
    for d in baseline_dirs:
        print(f"  - {d.name}")

    print(f"\nSkills prefix: {skills_prefix}* ({len(skills_dirs)} runs)")
    for d in skills_dirs:
        print(f"  - {d.name}")


def _print_overall_statistics(baseline_stats: dict, skills_stats: dict) -> None:
    """Print overall statistics section."""
    print("\n" + "-" * 70)
    print("OVERALL STATISTICS")
    print("-" * 70)
    print(f"{'':20s} {'Baseline':>15s} {'Skills':>15s} {'Diff':>15s}")
    print("-" * 70)

    b_rate = baseline_stats["overall_success_rate"]
    s_rate = skills_stats["overall_success_rate"]
    diff = s_rate - b_rate
    diff_pct = diff / b_rate * 100 if b_rate > 0 else 0

    print(
        f"{'Tasks':20s} {baseline_stats['n_tasks']:>15d} {skills_stats['n_tasks']:>15d}"
    )
    print(
        f"{'Total trials':20s} {baseline_stats['n_trials']:>15d} "
        f"{skills_stats['n_trials']:>15d}"
    )
    print(
        f"{'Successes':20s} {baseline_stats['n_successes']:>15d} "
        f"{skills_stats['n_successes']:>15d}"
    )
    print(f"{'Success rate':20s} {b_rate:>14.1%} {s_rate:>14.1%} {diff:>+14.1%}")
    print(f"{'Relative change':20s} {'':>15s} {'':>15s} {diff_pct:>+13.1f}%")


def _print_fisher_test(trial_comparison: dict) -> None:
    """Print Fisher's exact test results."""
    print("\n" + "-" * 70)
    print("FISHER'S EXACT TEST (all individual trials)")
    print("-" * 70)

    if "error" in trial_comparison:
        print(f"Error: {trial_comparison['error']}")
        return

    b = trial_comparison["baseline"]
    s = trial_comparison["skills"]
    print(f"Baseline: {b['n_successes']}/{b['n_trials']} = {b['success_rate']:.1%}")
    print(f"Skills:   {s['n_successes']}/{s['n_trials']} = {s['success_rate']:.1%}")
    print(f"Improvement: {trial_comparison['improvement']:+.1%}")
    print(f"Odds ratio: {trial_comparison['odds_ratio']:.3f}")
    print(f"p-value: {trial_comparison['p_value']:.4f}")

    if trial_comparison["significant_at_0.05"]:
        print(">>> SIGNIFICANT at p < 0.05 <<<")
    else:
        print("Not significant at p < 0.05")


def _print_mcnemar_test(paired_comparison: dict) -> None:
    """Print McNemar's test results."""
    print("\n" + "-" * 70)
    print("McNEMAR'S TEST (paired by task, majority vote)")
    print("-" * 70)

    if "error" in paired_comparison:
        print(f"Error: {paired_comparison['error']}")
        return

    print(f"Common tasks: {paired_comparison['common_tasks']}")
    c = paired_comparison["contingency"]
    print(f"Both succeed:    {c['both_success']:3d}")
    print(f"Baseline only:   {c['baseline_only']:3d} (skills regression)")
    print(f"Skills only:     {c['skills_only']:3d} (skills improvement)")
    print(f"Both fail:       {c['both_fail']:3d}")
    print(f"\nBaseline rate: {paired_comparison['baseline_success_rate']:.1%}")
    print(f"Skills rate:   {paired_comparison['skills_success_rate']:.1%}")
    print(f"Improvement:   {paired_comparison['improvement']:+.1%}")
    print(f"p-value: {paired_comparison['p_value']:.4f}")

    if paired_comparison["significant_at_0.05"]:
        print(">>> SIGNIFICANT at p < 0.05 <<<")
    else:
        print("Not significant at p < 0.05")

    if paired_comparison["baseline_only_tasks"]:
        print(f"\nRegressions ({len(paired_comparison['baseline_only_tasks'])}):")
        for task in paired_comparison["baseline_only_tasks"]:
            print(f"  - {task}")

    if paired_comparison["skills_only_tasks"]:
        print(f"\nImprovements ({len(paired_comparison['skills_only_tasks'])}):")
        for task in paired_comparison["skills_only_tasks"]:
            print(f"  - {task}")


def _print_win_rate_test(win_rate_comparison: dict) -> None:
    """Print win rate comparison (sign test) results."""
    print("\n" + "-" * 70)
    print("WIN RATE COMPARISON (sign test)")
    print("-" * 70)

    if "error" in win_rate_comparison:
        print(f"Error: {win_rate_comparison['error']}")
        return

    total = win_rate_comparison["common_tasks"]
    sw = win_rate_comparison["skills_wins"]
    bw = win_rate_comparison["baseline_wins"]
    ties = win_rate_comparison["ties"]
    print(f"Common tasks: {total}")
    print(f"Skills wins:   {sw:3d} ({sw / total:.1%})")
    print(f"Baseline wins: {bw:3d} ({bw / total:.1%})")
    print(f"Ties:          {ties:3d} ({ties / total:.1%})")
    print(f"\np-value: {win_rate_comparison['p_value']:.4f}")

    if win_rate_comparison["significant_at_0.05"]:
        print(">>> SIGNIFICANT at p < 0.05 <<<")
    else:
        print("Not significant at p < 0.05")

    _print_top_improvements(win_rate_comparison)
    _print_top_regressions(win_rate_comparison)


def _print_top_improvements(win_rate_comparison: dict) -> None:
    """Print top improvements from win rate comparison."""
    top_imp = win_rate_comparison.get("top_improvements", [])
    if not top_imp:
        return

    print("\nTop 5 improvements (skills > baseline):")
    for item in top_imp[:5]:
        print(
            f"  {item['task']}: "
            f"{item['baseline_rate']:.0%} -> {item['skills_rate']:.0%} "
            f"(+{item['improvement']:.0%})"
        )


def _print_top_regressions(win_rate_comparison: dict) -> None:
    """Print top regressions from win rate comparison."""
    top_reg = win_rate_comparison.get("top_regressions", [])
    if not top_reg:
        return

    print("\nTop regressions (baseline > skills):")
    for item in top_reg[:5]:
        print(
            f"  {item['task']}: "
            f"{item['baseline_rate']:.0%} -> {item['skills_rate']:.0%} "
            f"(-{item['regression']:.0%})"
        )


def print_skill_token_breakdown(token_analysis: dict) -> None:
    """Print skill token usage breakdown."""
    print("\n" + "-" * 70)
    print("SKILL TOKEN BREAKDOWN (skills runs only)")
    print("-" * 70)

    total_meta = token_analysis.get("total_metadata_tokens", 0)
    total_content = token_analysis.get("total_content_tokens", 0)
    n_tasks = token_analysis.get("n_tasks_analyzed", 0)
    avg_meta = token_analysis.get("avg_metadata_per_task", 0)
    avg_content = token_analysis.get("avg_content_per_task", 0)

    print(f"Total tasks analyzed: {n_tasks}")
    print("\nToken counts (via tiktoken cl100k_base):")
    print(f"  Skill metadata (listing):  {total_meta:,} total, {avg_meta:,.0f}/task")
    print(f"  Skill content (read):   {total_content:,} total, {avg_content:,.0f}/task")
    print(f"  Combined skill tokens:     {total_meta + total_content:,} total")

    # Per-task breakdown for tasks with significant content
    per_task = token_analysis.get("per_task", {})
    if per_task:
        _print_skill_token_table(per_task)


def _print_skill_token_table(per_task: dict) -> None:
    """Print skill content tokens by task table."""
    print("\n" + "-" * 70)
    print("SKILL CONTENT TOKENS BY TASK (top 20 by content tokens)")
    print("-" * 70)
    hdr = f"{'Task':<30s} {'Meta':>8s} {'Content':>10s}  Skills Read"
    print(hdr)
    print("-" * 70)

    # Sort by content tokens descending
    sorted_tasks = sorted(
        per_task.items(),
        key=lambda x: x[1]["avg_content_tokens"],
        reverse=True,
    )

    for task_name, data in sorted_tasks[:20]:
        meta = data["avg_metadata_tokens"]
        content = data["avg_content_tokens"]
        skills = data.get("unique_skills", [])
        skills_str = ", ".join(skills[:2]) if skills else "-"
        if len(skills) > 2:
            skills_str += f" (+{len(skills) - 2})"
        print(f"{task_name:<30s} {meta:>8.0f} {content:>10.0f}  {skills_str}")

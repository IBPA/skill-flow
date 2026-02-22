"""Statistical comparison tests for evaluation results."""

from scipy import stats


def compare_paired_tasks(
    baseline_results: dict[str, list[float]], skills_results: dict[str, list[float]]
) -> dict:
    """Compare task outcomes between baseline and skills on common tasks.

    Uses majority vote per task across runs to determine task success.
    """
    common_tasks = set(baseline_results.keys()) & set(skills_results.keys())

    if len(common_tasks) < 5:
        return {"error": f"Not enough common tasks ({len(common_tasks)}). Need >= 5."}

    # For each task, compute majority outcome (weak majority: >= 50%)
    def majority_success(rewards: list[float]) -> bool:
        successes = sum(1 for r in rewards if r == 1.0)
        return successes >= len(rewards) / 2

    # Build contingency table for McNemar's test
    a = b = c = d = 0
    baseline_only_tasks = []
    skills_only_tasks = []

    for task in common_tasks:
        base_success = majority_success(baseline_results[task])
        skill_success = majority_success(skills_results[task])

        if base_success and skill_success:
            a += 1
        elif base_success and not skill_success:
            b += 1
            baseline_only_tasks.append(task)
        elif not base_success and skill_success:
            c += 1
            skills_only_tasks.append(task)
        else:
            d += 1

    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
        method = "no discordant pairs"
    elif n_discordant < 25:
        result = stats.binomtest(c, n_discordant, 0.5, alternative="two-sided")
        p_value = result.pvalue
        method = "exact binomial"
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        method = "chi-squared"

    baseline_rate = (a + b) / len(common_tasks)
    skills_rate = (a + c) / len(common_tasks)

    return {
        "test": "McNemar (majority vote)",
        "method": method,
        "common_tasks": len(common_tasks),
        "contingency": {
            "both_success": a,
            "baseline_only": b,
            "skills_only": c,
            "both_fail": d,
        },
        "baseline_success_rate": baseline_rate,
        "skills_success_rate": skills_rate,
        "improvement": skills_rate - baseline_rate,
        "improvement_pct": (skills_rate - baseline_rate) / baseline_rate * 100
        if baseline_rate > 0
        else 0,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
        "baseline_only_tasks": sorted(baseline_only_tasks),
        "skills_only_tasks": sorted(skills_only_tasks),
    }


def compare_win_rates(
    baseline_results: dict[str, list[float]], skills_results: dict[str, list[float]]
) -> dict:
    """Compare task outcomes using win rate (which treatment has higher success rate).

    For each task, compares the success rate between baseline and skills.
    Uses a sign test (binomial) to determine if skills win significantly more often.
    """
    common_tasks = set(baseline_results.keys()) & set(skills_results.keys())

    if len(common_tasks) < 5:
        return {"error": f"Not enough common tasks ({len(common_tasks)}). Need >= 5."}

    baseline_wins = 0
    skills_wins = 0
    ties = 0
    baseline_win_tasks: list[str] = []
    skills_win_tasks: list[str] = []
    tie_tasks: list[str] = []

    # Track rate differences for detailed analysis
    # Tuple structure: task name, baseline rate, skills rate, difference
    rate_diffs: list[tuple[str, float, float, float]] = []

    for task in common_tasks:
        base_rewards = baseline_results[task]
        skill_rewards = skills_results[task]

        base_rate = sum(base_rewards) / len(base_rewards) if base_rewards else 0
        skill_rate = sum(skill_rewards) / len(skill_rewards) if skill_rewards else 0
        diff = skill_rate - base_rate

        rate_diffs.append((task, base_rate, skill_rate, diff))

        if skill_rate > base_rate:
            skills_wins += 1
            skills_win_tasks.append(task)
        elif base_rate > skill_rate:
            baseline_wins += 1
            baseline_win_tasks.append(task)
        else:
            ties += 1
            tie_tasks.append(task)

    # Sign test: among non-tied tasks, is skills winning significantly more?
    n_decisive = baseline_wins + skills_wins
    if n_decisive == 0:
        p_value = 1.0
    else:
        result = stats.binomtest(skills_wins, n_decisive, 0.5, alternative="two-sided")
        p_value = result.pvalue

    # Sort rate_diffs by improvement (descending)
    rate_diffs.sort(key=lambda x: x[3], reverse=True)
    top_improvements = rate_diffs[:5]
    top_regressions = rate_diffs[-5:][::-1]  # Reverse to show worst first

    return {
        "test": "Win Rate (sign test)",
        "common_tasks": len(common_tasks),
        "skills_wins": skills_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "skills_win_rate": skills_wins / len(common_tasks),
        "baseline_win_rate": baseline_wins / len(common_tasks),
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
        "skills_win_tasks": sorted(skills_win_tasks),
        "baseline_win_tasks": sorted(baseline_win_tasks),
        "tie_tasks": sorted(tie_tasks),
        "top_improvements": [
            {"task": t, "baseline_rate": br, "skills_rate": sr, "improvement": d}
            for t, br, sr, d in top_improvements
        ],
        "top_regressions": [
            {"task": t, "baseline_rate": br, "skills_rate": sr, "regression": -d}
            for t, br, sr, d in top_regressions
            if d < 0
        ],
    }


def compare_all_trials(
    baseline_results: dict[str, list[float]], skills_results: dict[str, list[float]]
) -> dict:
    """Compare all individual trials using Fisher's exact test."""
    baseline_successes = sum(
        sum(1 for r in rewards if r == 1.0) for rewards in baseline_results.values()
    )
    baseline_total = sum(len(rewards) for rewards in baseline_results.values())

    skills_successes = sum(
        sum(1 for r in rewards if r == 1.0) for rewards in skills_results.values()
    )
    skills_total = sum(len(rewards) for rewards in skills_results.values())

    # 2x2 contingency table
    table = [
        [baseline_successes, baseline_total - baseline_successes],
        [skills_successes, skills_total - skills_successes],
    ]

    odds_ratio, p_value = stats.fisher_exact(table)

    baseline_rate = baseline_successes / baseline_total if baseline_total else 0
    skills_rate = skills_successes / skills_total if skills_total else 0

    return {
        "test": "Fisher's Exact (all trials)",
        "baseline": {
            "n_trials": baseline_total,
            "n_successes": baseline_successes,
            "success_rate": baseline_rate,
        },
        "skills": {
            "n_trials": skills_total,
            "n_successes": skills_successes,
            "success_rate": skills_rate,
        },
        "improvement": skills_rate - baseline_rate,
        "improvement_pct": (skills_rate - baseline_rate) / baseline_rate * 100
        if baseline_rate > 0
        else 0,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
    }

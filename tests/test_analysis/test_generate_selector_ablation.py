"""Tests for the selector ablation table generator and the loader model filter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from analysis.comparison.utils.loader import discover_runs
from analysis.results.t20_generate_selector_ablation import render_table

if TYPE_CHECKING:
    from pathlib import Path


def _write_run(run_dir: Path, rewards: dict[str, float]) -> None:
    """Write a minimal result.json mapping tasks to rewards."""
    run_dir.mkdir(parents=True)
    by_reward: dict[str, list[str]] = {}
    for task, reward in rewards.items():
        by_reward.setdefault(str(reward), []).append(f"{task}__trial")
    result = {
        "stats": {
            "evals": {
                "eval": {"reward_stats": {"reward": by_reward}, "exception_stats": {}}
            }
        }
    }
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _build_eval_dir(root: Path) -> Path:
    """Create gpt5mini baseline/skillflow/no-selector runs plus a claude run."""
    eval_dir = root / "evaluation"
    rewards = {"task-a": 1.0, "task-b": 0.0}
    specs = [
        "sk-baseline-gpt5mini-medium-20260101-000000",
        "sk-skillflow-inject-gpt5mini-medium-20260101-000000",
        "sk-skillflow-deep-reranker-inject-gpt5mini-medium-20260101-000000",
        # Same condition, different model: must be separable by the filter.
        "sk-skillflow-inject-claudehaiku45-20260101-000000",
    ]
    for name in specs:
        _write_run(eval_dir / name, rewards)
    return eval_dir


def test_discover_runs_model_filter(tmp_path: Path) -> None:
    """The model substring filter splits same-condition runs across models."""
    eval_dir = _build_eval_dir(tmp_path)

    unfiltered = discover_runs(eval_dir, "skillflow-inject")
    gpt_only = discover_runs(eval_dir, "skillflow-inject", model="gpt5mini")

    assert len(unfiltered) == 2  # gpt5mini + claude both extract to the condition
    assert len(gpt_only) == 1
    assert "gpt5mini" in gpt_only[0].name


def test_discover_runs_excludes_other_condition(tmp_path: Path) -> None:
    """The deep-reranker (no-selector) runs are not matched by skillflow-inject."""
    eval_dir = _build_eval_dir(tmp_path)

    nosel = discover_runs(eval_dir, "skillflow-deep-reranker-inject", model="gpt5mini")

    assert len(nosel) == 1
    assert "deep-reranker" in nosel[0].name


def test_render_table_rows(tmp_path: Path) -> None:
    """The table renders the three ablation rows for the filtered model."""
    eval_dir = _build_eval_dir(tmp_path)

    text = "\n".join(render_table(eval_dir, model="gpt5mini"))

    assert r"\begin{tabular}{lcc}" in text
    assert "No Skills" in text
    assert "SkillFlow (with selector)" in text
    assert "Deep-reranker top-10 (no selector)" in text
    # The claude run is filtered out, so only one SkillFlow-with-selector row.
    assert text.count("SkillFlow (with selector)") == 1

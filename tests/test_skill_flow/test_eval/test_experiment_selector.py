"""Tests for skill_flow.eval.experiments.selector."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from skill_flow.config import (
    SelectorExperimentConfig,
    SelectorVariant,
)
from skill_flow.eval.experiments.selector import (
    _selector_label,
    _to_selector_config,
    run_selector_experiment,
)
from skill_flow.eval.models import (
    EvalReport,
    EvalSummary,
)


def _make_report(mrr: float = 0.5) -> EvalReport:
    return EvalReport(
        summary=EvalSummary(
            num_tasks_total=10,
            num_tasks_evaluated=10,
            num_tasks_no_skills=0,
            num_skills_injected=5,
            mean_recall_at={1: 0.3, 2: 0.4, 5: 0.6, 10: 0.8, 100: 1.0},
            mean_precision_at={1: 0.3, 2: 0.2, 5: 0.12, 10: 0.08, 100: 0.01},
            mean_hit_at={1: 0.3, 2: 0.4, 5: 0.6, 10: 0.8, 100: 1.0},
            mrr=mrr,
        ),
        task_results=[],
        config={"tasks_dir": "tasks", "index_dir": "index"},
    )


class TestToSelectorConfig:
    def test_builds_config(self) -> None:
        variant = SelectorVariant(
            model="gpt-4o-mini", input_top_k=20, max_tokens=512, temperature=0.3
        )
        config = _to_selector_config(variant)
        assert config.enabled is True
        assert config.model == "gpt-4o-mini"
        assert config.input_top_k == 20
        assert config.max_tokens == 512

    def test_derives_cache_path_from_cache_dir(self) -> None:
        variant = SelectorVariant(model="gpt-4o-mini", input_top_k=20)
        config = _to_selector_config(variant, cache_dir="outputs/experiments/")
        assert "selector_cache_" in config.cache_path
        assert config.cache_path.startswith("outputs/experiments/")

    def test_rejects_list_input_top_k(self) -> None:
        variant = SelectorVariant(model="gpt-4o-mini", input_top_k=[10, 20])
        with pytest.raises(TypeError, match="must be a single int"):
            _to_selector_config(variant)


class TestSelectorLabel:
    def test_without_top_k(self) -> None:
        variant = SelectorVariant(model="gpt-4o-mini")
        assert _selector_label(variant) == "gpt-4o-mini-top5"

    def test_with_top_k(self) -> None:
        variant = SelectorVariant(model="gpt-4o-mini", input_top_k=20)
        assert _selector_label(variant) == "gpt-4o-mini-top20"

    def test_with_include_tasks(self) -> None:
        variant = SelectorVariant(model="gpt-4o-mini")
        label = _selector_label(variant, include_tasks=["a", "b", "c"])
        assert label == "gpt-4o-mini-top5-i3"


class TestRunSelectorExperiment:
    @patch("skill_flow.eval.experiments.selector.run_selector_evaluation")
    def test_runs_all_variants(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        mock_eval.return_value = _make_report(0.7)
        config = SelectorExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            input_report_path="outputs/stage3.json",
            selectors=[
                SelectorVariant(model="gpt-4o-mini"),
                SelectorVariant(model="gpt-4o", input_top_k=20),
            ],
        )
        results = run_selector_experiment(config)
        assert len(results) == 2
        assert results[0][0] == "gpt-4o-mini-top5"
        assert results[1][0] == "gpt-4o-top20"

    @patch("skill_flow.eval.experiments.selector.run_selector_evaluation")
    def test_skips_disabled(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        config = SelectorExperimentConfig(
            name="test",
            output_dir=str(tmp_path),
            selectors=[
                SelectorVariant(model="a", enabled=False),
                SelectorVariant(model="b"),
            ],
        )
        mock_eval.return_value = _make_report(0.6)
        results = run_selector_experiment(config)
        assert len(results) == 1

    @patch("skill_flow.eval.experiments.selector.run_selector_evaluation")
    def test_empty_selectors(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        config = SelectorExperimentConfig(
            name="empty", output_dir=str(tmp_path), selectors=[]
        )
        results = run_selector_experiment(config)
        assert results == []

    @patch("skill_flow.eval.experiments.selector.run_selector_evaluation")
    def test_list_input_top_k_runs_each(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        mock_eval.return_value = _make_report(0.7)
        config = SelectorExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            input_report_path="outputs/stage3.json",
            selectors=[SelectorVariant(model="gpt-4o-mini", input_top_k=[10, 20, 50])],
        )
        results = run_selector_experiment(config)
        assert len(results) == 3
        labels = [r[0] for r in results]
        assert "gpt-4o-mini-top10" in labels
        assert "gpt-4o-mini-top20" in labels
        assert "gpt-4o-mini-top50" in labels

"""Tests for skill_flow.eval.experiments.reranker."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from skill_flow.config import (
    QueryGenConfig,
    RerankerExperimentConfig,
    RerankerVariant,
)
from skill_flow.eval.experiments.reranker import (
    _reranker_label,
    _to_reranker_config,
    run_reranker_experiment,
)
from skill_flow.eval.models import (
    EvalReport,
    EvalSummary,
    RetrievedSkill,
    TaskGroundTruth,
    TaskResult,
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


class TestToRerankerConfig:
    def test_builds_config(self) -> None:
        variant = RerankerVariant(
            model_name="BAAI/bge-reranker-v2-m3",
            batch_size=32,
            max_content_chars=4096,
        )
        config = _to_reranker_config(variant)
        assert config.enabled is True
        assert config.model_name == "BAAI/bge-reranker-v2-m3"
        assert config.batch_size == 32
        assert config.max_content_chars == 4096

    def test_rejects_list_content_chars(self) -> None:
        variant = RerankerVariant(model_name="test", max_content_chars=[512, 1024])
        with pytest.raises(TypeError, match="must be a single int"):
            _to_reranker_config(variant)

    def test_passes_query_gen(self) -> None:
        qg = QueryGenConfig(enabled=True, model="gpt-4o-mini")
        variant = RerankerVariant(model_name="test", query_gen=qg)
        config = _to_reranker_config(variant)
        assert config.query_gen is not None
        assert config.query_gen.enabled is True


class TestRerankerLabel:
    def test_without_content_chars(self) -> None:
        variant = RerankerVariant(model_name="BAAI/bge-reranker-v2-m3")
        assert _reranker_label(variant) == "baai-bge-reranker-v2-m3"

    def test_with_content_chars(self) -> None:
        variant = RerankerVariant(
            model_name="BAAI/bge-reranker-v2-m3", max_content_chars=4096
        )
        assert _reranker_label(variant) == "baai-bge-reranker-v2-m3-4096chars"

    def test_with_include_tasks(self) -> None:
        variant = RerankerVariant(model_name="BAAI/bge-reranker-v2-m3")
        label = _reranker_label(variant, include_tasks=["a", "b", "c"])
        assert label == "baai-bge-reranker-v2-m3-i3"


class TestRunRerankerExperiment:
    @patch("skill_flow.eval.experiments.reranker.run_reranker_evaluation")
    def test_runs_all_variants(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        mock_eval.return_value = _make_report(0.7)
        config = RerankerExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            input_report_path="outputs/stage1.json",
            rerankers=[
                RerankerVariant(model_name="reranker-a"),
                RerankerVariant(model_name="reranker-b", max_content_chars=4096),
            ],
        )
        results = run_reranker_experiment(config)
        assert len(results) == 2
        assert mock_eval.call_count == 2
        assert results[0][0] == "reranker-a"
        assert results[1][0] == "reranker-b-4096chars"

    @patch("skill_flow.eval.experiments.reranker.run_reranker_evaluation")
    def test_skips_disabled(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        config = RerankerExperimentConfig(
            name="test",
            output_dir=str(tmp_path),
            rerankers=[
                RerankerVariant(model_name="a", enabled=False),
                RerankerVariant(model_name="b"),
            ],
        )
        mock_eval.return_value = _make_report(0.6)
        results = run_reranker_experiment(config)
        assert len(results) == 1

    @patch("skill_flow.eval.experiments.reranker.run_reranker_evaluation")
    def test_empty_rerankers(self, mock_eval: MagicMock, tmp_path: Path) -> None:
        config = RerankerExperimentConfig(
            name="empty", output_dir=str(tmp_path), rerankers=[]
        )
        results = run_reranker_experiment(config)
        assert results == []

    @patch("skill_flow.eval.experiments.reranker.run_reranker_evaluation")
    def test_list_content_chars_scores_each(
        self, mock_eval: MagicMock, tmp_path: Path
    ) -> None:
        mock_eval.return_value = _make_report(0.7)
        config = RerankerExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            input_report_path="outputs/stage1.json",
            rerankers=[
                RerankerVariant(
                    model_name="test-reranker", max_content_chars=[512, 1024, 4096]
                )
            ],
        )
        results = run_reranker_experiment(config)
        assert len(results) == 3
        labels = [r[0] for r in results]
        assert "test-reranker-512chars" in labels
        assert "test-reranker-1024chars" in labels
        assert "test-reranker-4096chars" in labels

    @patch("skill_flow.eval.experiments.reranker.load_ground_truth")
    @patch("skill_flow.eval.experiments.reranker.run_reranker_evaluation")
    def test_content_chars_cross_aggregation(
        self,
        mock_eval: MagicMock,
        mock_gt: MagicMock,
        tmp_path: Path,
    ) -> None:
        task_result = TaskResult(
            task_id="t1",
            query="q",
            num_ground_truth=1,
            num_injected=1,
            retrieved_skills=[
                RetrievedSkill(key="s0", score=0.9, query_scores=[0.9, 0.1]),
            ],
            recall_at={1: 1.0},
            hit_at={1: 1.0},
            reciprocal_rank=1.0,
        )
        report = _make_report(0.7)
        report.task_results = [task_result]
        mock_eval.return_value = report
        gt = [
            TaskGroundTruth(
                task_id="t1",
                query="q",
                ground_truth_keys=["s0"],
                all_skill_names=[],
                injected_skills=["s0"],
            )
        ]
        mock_gt.return_value = (gt, [], [])
        qg = QueryGenConfig(enabled=True, num_queries=2, aggregation=["max", "mean"])
        config = RerankerExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            input_report_path="outputs/stage1.json",
            rerankers=[
                RerankerVariant(
                    model_name="test-reranker",
                    max_content_chars=[512, 1024],
                    query_gen=qg,
                )
            ],
        )
        results = run_reranker_experiment(config)
        assert len(results) == 4
        labels = [r[0] for r in results]
        assert "test-reranker-512chars-q2-max" in labels
        assert "test-reranker-512chars-q2-mean" in labels
        assert "test-reranker-1024chars-q2-max" in labels
        assert "test-reranker-1024chars-q2-mean" in labels

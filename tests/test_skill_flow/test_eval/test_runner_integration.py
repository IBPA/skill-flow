"""Integration tests for skill_flow.eval.runner."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from skill_flow.config import QueryGenConfig, RerankerConfig
from skill_flow.eval.models import (
    EvalRunConfig,
    InjectedSkill,
    RerankerEvalConfig,
    TaskGroundTruth,
)
from skill_flow.eval.runner import run_evaluation, run_reranker_evaluation
from skill_flow.retriever.retriever import SearchResult


def _make_task_gt(
    task_id: str = "task-1",
    query: str = "test query",
    gt_keys: list[str] | None = None,
) -> TaskGroundTruth:
    keys = gt_keys or ["skillsbench/task-1/a"]
    return TaskGroundTruth(
        task_id=task_id,
        query=query,
        ground_truth_keys=keys,
        all_skill_names=[k.split("/")[-1] for k in keys],
        injected_skills=keys,
    )


class TestRunEvaluation:
    @patch("skill_flow.eval.runner.IndexSearcher")
    @patch("skill_flow.eval.runner.Encoder")
    @patch("skill_flow.eval.runner.load_ground_truth")
    def test_end_to_end(
        self,
        mock_gt: MagicMock,
        mock_encoder_cls: MagicMock,
        mock_searcher_cls: MagicMock,
        tmp_path: Path,
    ):
        task = _make_task_gt(gt_keys=["skillsbench/task-1/a"])
        injected = [
            InjectedSkill(
                key="skillsbench/task-1/a",
                name="a",
                description="d",
            )
        ]
        mock_gt.return_value = ([task], injected, [])

        mock_encoder = MagicMock()
        mock_encoder_cls.return_value = mock_encoder

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = [
            SearchResult(key="skillsbench/task-1/a", score=0.9),
        ]
        mock_searcher.add_descriptions = MagicMock()
        mock_searcher.add_contents = MagicMock()
        mock_searcher_cls.return_value = mock_searcher

        output = tmp_path / "report.json"
        config = EvalRunConfig(
            tasks_dir=tmp_path / "tasks",
            index_dir=tmp_path / "index",
            output_path=output,
        )

        report = run_evaluation(config)

        assert report.summary.num_tasks_evaluated == 1
        assert report.summary.mrr == 1.0
        assert report.task_results[0].recall_at[1] == 1.0
        assert output.exists()


class TestRunRerankerEvaluation:
    @patch("skill_flow.eval.runner.Reranker")
    @patch("skill_flow.eval.runner.load_ground_truth")
    def test_end_to_end(
        self,
        mock_gt: MagicMock,
        mock_reranker_cls: MagicMock,
        tmp_path: Path,
    ):
        task = _make_task_gt(gt_keys=["skillsbench/task-1/a"])
        mock_gt.return_value = ([task], [], [])

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            SearchResult(key="skillsbench/task-1/a", score=0.95, description="d"),
            SearchResult(key="skillsmp/x", score=0.5, description="desc x"),
        ]
        mock_reranker_cls.return_value = mock_reranker

        stage1_report = {
            "summary": {
                "num_tasks_total": 1,
                "num_tasks_evaluated": 1,
                "num_tasks_no_skills": 0,
                "num_skills_injected": 1,
                "mean_recall_at": {"1": 0.0, "5": 1.0},
                "mean_precision_at": {"1": 0.0, "5": 0.2},
                "mean_hit_at": {"1": 0.0, "5": 1.0},
                "mrr": 0.5,
            },
            "task_results": [
                {
                    "task_id": "task-1",
                    "num_ground_truth": 1,
                    "num_injected": 1,
                    "retrieved_skills": [
                        {"key": "skillsmp/x", "score": 0.9, "description": "desc x"},
                        {
                            "key": "skillsbench/task-1/a",
                            "score": 0.8,
                            "description": "d",
                            "content": "# Full content",
                        },
                    ],
                    "recall_at": {"1": 0.0, "5": 1.0},
                    "precision_at": {"1": 0.0, "5": 0.2},
                    "hit_at": {"1": 0.0, "5": 1.0},
                    "reciprocal_rank": 0.5,
                },
            ],
            "config": {
                "tasks_dir": "tasks",
                "index_dir": "index",
            },
        }
        stage1_path = tmp_path / "stage1.json"
        stage1_path.write_text(json.dumps(stage1_report))

        output = tmp_path / "reranker-report.json"
        config = RerankerEvalConfig(
            stage1_report_path=stage1_path,
            tasks_dir=tmp_path / "tasks",
            index_dir=tmp_path / "index",
            reranker=RerankerConfig(enabled=True),
            output_path=output,
        )

        report = run_reranker_evaluation(config)

        assert report.summary.num_tasks_evaluated == 1
        assert report.summary.mrr == 1.0
        assert report.task_results[0].recall_at[1] == 1.0
        assert output.exists()
        mock_reranker.rerank.assert_called_once()

    @patch("skill_flow.eval.runner.Reranker")
    @patch("skill_flow.eval.runner.load_ground_truth")
    def test_threads_content_to_candidates(
        self,
        mock_gt: MagicMock,
        mock_reranker_cls: MagicMock,
        tmp_path: Path,
    ):
        task = _make_task_gt(gt_keys=["skillsbench/task-1/a"])
        mock_gt.return_value = ([task], [], [])

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            SearchResult(key="skillsbench/task-1/a", score=0.95),
        ]
        mock_reranker_cls.return_value = mock_reranker

        stage1_report = {
            "summary": {
                "num_tasks_total": 1,
                "num_tasks_evaluated": 1,
                "num_tasks_no_skills": 0,
                "num_skills_injected": 1,
                "mean_recall_at": {"1": 1.0},
                "mean_precision_at": {"1": 1.0},
                "mean_hit_at": {"1": 1.0},
                "mrr": 1.0,
            },
            "task_results": [
                {
                    "task_id": "task-1",
                    "num_ground_truth": 1,
                    "num_injected": 1,
                    "retrieved_skills": [
                        {"key": "skillsmp/x", "score": 0.9, "description": "desc x"},
                        {
                            "key": "skillsbench/task-1/a",
                            "score": 0.8,
                            "description": "d",
                            "content": "# Full",
                        },
                    ],
                    "recall_at": {"1": 1.0},
                    "precision_at": {"1": 1.0},
                    "hit_at": {"1": 1.0},
                    "reciprocal_rank": 1.0,
                },
            ],
            "config": {
                "tasks_dir": "tasks",
                "index_dir": "index",
            },
        }
        stage1_path = tmp_path / "stage1.json"
        stage1_path.write_text(json.dumps(stage1_report))

        config = RerankerEvalConfig(
            stage1_report_path=stage1_path,
            tasks_dir=tmp_path / "tasks",
            index_dir=tmp_path / "index",
            reranker=RerankerConfig(enabled=True),
        )

        run_reranker_evaluation(config)

        candidates = mock_reranker.rerank.call_args[0][1]
        gt_candidate = next(c for c in candidates if c.key == "skillsbench/task-1/a")
        assert gt_candidate.content == "# Full"

        corpus_candidate = next(c for c in candidates if c.key == "skillsmp/x")
        assert corpus_candidate.content == ""

    @patch("skill_flow.eval.runner.QueryGenerator")
    @patch("skill_flow.eval.runner.Reranker")
    @patch("skill_flow.eval.runner.load_ground_truth")
    def test_uses_generated_query(
        self,
        mock_gt: MagicMock,
        mock_reranker_cls: MagicMock,
        mock_qgen_cls: MagicMock,
        tmp_path: Path,
    ):
        task = _make_task_gt(gt_keys=["skillsbench/task-1/a"])
        mock_gt.return_value = ([task], [], [])

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            SearchResult(key="skillsbench/task-1/a", score=0.95, description="d"),
        ]
        mock_reranker_cls.return_value = mock_reranker

        mock_qgen = MagicMock()
        mock_qgen.generate.return_value = "concise query"
        mock_qgen_cls.return_value = mock_qgen

        stage1_report = {
            "summary": {
                "num_tasks_total": 1,
                "num_tasks_evaluated": 1,
                "num_tasks_no_skills": 0,
                "num_skills_injected": 1,
                "mean_recall_at": {"1": 1.0},
                "mean_precision_at": {"1": 1.0},
                "mean_hit_at": {"1": 1.0},
                "mrr": 1.0,
            },
            "task_results": [
                {
                    "task_id": "task-1",
                    "num_ground_truth": 1,
                    "num_injected": 1,
                    "retrieved_skills": [
                        {
                            "key": "skillsbench/task-1/a",
                            "score": 0.8,
                            "description": "d",
                        },
                    ],
                    "recall_at": {"1": 1.0},
                    "precision_at": {"1": 1.0},
                    "hit_at": {"1": 1.0},
                    "reciprocal_rank": 1.0,
                },
            ],
            "config": {
                "tasks_dir": "tasks",
                "index_dir": "index",
            },
        }
        stage1_path = tmp_path / "stage1.json"
        stage1_path.write_text(json.dumps(stage1_report))

        config = RerankerEvalConfig(
            stage1_report_path=stage1_path,
            tasks_dir=tmp_path / "tasks",
            index_dir=tmp_path / "index",
            reranker=RerankerConfig(
                enabled=True,
                query_gen=QueryGenConfig(
                    enabled=True,
                    cache_path=str(tmp_path / "c.json"),
                ),
            ),
        )

        report = run_reranker_evaluation(config)

        mock_qgen.generate.assert_called_once_with("task-1", "test query")
        rerank_call_query = mock_reranker.rerank.call_args[0][0]
        assert rerank_call_query == "concise query"
        assert report.task_results[0].rerank_query == "concise query"

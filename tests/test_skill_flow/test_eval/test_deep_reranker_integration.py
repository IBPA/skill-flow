"""Integration tests for Stage 3 deep-reranker evaluation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from skill_flow.config import DeepRerankerConfig
from skill_flow.eval.models import TaskGroundTruth
from skill_flow.eval.runner_stages import run_deep_reranker_evaluation
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


class TestRunDeepRerankerEvaluation:
    @patch("skill_flow.eval.runner_stages.Reranker")
    @patch("skill_flow.eval.runner_stages.load_ground_truth")
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
            SearchResult(key="skillsbench/task-1/a", score=0.99, description="d"),
            SearchResult(key="skillsmp/x", score=0.3, description="desc x"),
        ]
        mock_reranker_cls.return_value = mock_reranker

        stage2_report = {
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
                }
            ],
            "config": {"tasks_dir": "tasks", "index_dir": "index"},
        }
        stage2_path = tmp_path / "stage2.json"
        stage2_path.write_text(json.dumps(stage2_report))

        output = tmp_path / "deep-reranker-report.json"
        report = run_deep_reranker_evaluation(
            stage2_path,
            DeepRerankerConfig(enabled=True),
            tmp_path / "tasks",
            tmp_path / "index",
            output_path=output,
        )

        assert report.summary.num_tasks_evaluated == 1
        assert report.summary.mrr == 1.0
        assert report.task_results[0].recall_at[1] == 1.0
        assert output.exists()
        mock_reranker.rerank.assert_called_once()

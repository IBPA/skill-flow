"""Integration tests for Stage 4 selector evaluation."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from skill_flow.config import SelectorConfig
from skill_flow.eval.models import TaskGroundTruth
from skill_flow.eval.runner_stages import run_selector_evaluation
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


class TestRunSelectorEvaluation:
    @patch("skill_flow.eval.runner_stages.Selector")
    @patch("skill_flow.eval.runner_stages.load_ground_truth")
    def test_end_to_end(
        self,
        mock_gt: MagicMock,
        mock_selector_cls: MagicMock,
        tmp_path: Path,
    ):
        task = _make_task_gt(gt_keys=["skillsbench/task-1/a"])
        mock_gt.return_value = ([task], [], [])

        mock_selector = MagicMock()
        mock_selector.select.return_value = [
            SearchResult(
                key="skillsbench/task-1/a",
                score=0.99,
                description="d",
                content="# Skill A",
            ),
        ]
        mock_selector_cls.return_value = mock_selector

        stage3_report = {
            "summary": {
                "num_tasks_total": 1,
                "num_tasks_evaluated": 1,
                "num_tasks_no_skills": 0,
                "num_skills_injected": 1,
                "mean_recall_at": {"1": 1.0, "5": 1.0},
                "mean_precision_at": {"1": 1.0, "5": 0.2},
                "mean_hit_at": {"1": 1.0, "5": 1.0},
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
                            "score": 0.99,
                            "description": "d",
                            "content": "# Skill A",
                        },
                        {
                            "key": "skillsmp/x",
                            "score": 0.3,
                            "description": "desc x",
                            "content": "# X",
                        },
                    ],
                    "recall_at": {"1": 1.0, "5": 1.0},
                    "precision_at": {"1": 1.0, "5": 0.2},
                    "hit_at": {"1": 1.0, "5": 1.0},
                    "reciprocal_rank": 1.0,
                }
            ],
            "config": {"tasks_dir": "tasks", "index_dir": "index"},
        }
        stage3_path = tmp_path / "stage3.json"
        stage3_path.write_text(json.dumps(stage3_report))

        output = tmp_path / "selector-report.json"
        report = run_selector_evaluation(
            stage3_path,
            SelectorConfig(enabled=True),
            tmp_path / "tasks",
            tmp_path / "index",
            output_path=output,
        )

        assert report.summary.num_tasks_evaluated == 1
        assert report.summary.mrr == 1.0
        assert report.task_results[0].recall_at[1] == 1.0
        assert output.exists()
        mock_selector.select.assert_called_once()

    @patch("skill_flow.eval.runner_stages.Selector")
    @patch("skill_flow.eval.runner_stages.load_ground_truth")
    def test_filters_false_positives(
        self,
        mock_gt: MagicMock,
        mock_selector_cls: MagicMock,
        tmp_path: Path,
    ):
        task = _make_task_gt(gt_keys=["skillsbench/task-1/a"])
        mock_gt.return_value = ([task], [], [])

        mock_selector = MagicMock()
        mock_selector.select.return_value = [
            SearchResult(key="skillsmp/x", score=0.3, description="desc x"),
        ]
        mock_selector_cls.return_value = mock_selector

        stage3_report = {
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
                            "score": 0.99,
                            "description": "d",
                        },
                        {"key": "skillsmp/x", "score": 0.3, "description": "desc x"},
                    ],
                    "recall_at": {"1": 1.0},
                    "precision_at": {"1": 1.0},
                    "hit_at": {"1": 1.0},
                    "reciprocal_rank": 1.0,
                }
            ],
            "config": {"tasks_dir": "tasks", "index_dir": "index"},
        }
        stage3_path = tmp_path / "stage3.json"
        stage3_path.write_text(json.dumps(stage3_report))

        output = tmp_path / "selector-report.json"
        report = run_selector_evaluation(
            stage3_path,
            SelectorConfig(enabled=True),
            tmp_path / "tasks",
            tmp_path / "index",
            output_path=output,
        )

        assert report.summary.mrr == 0.5
        assert report.task_results[0].recall_at[1] == 0.0
        assert report.task_results[0].recall_at[2] == 1.0

"""Tests for grid search helpers (aggregation, content chars, reaggregation)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from skill_flow.config import (
    QueryGenConfig,
    RerankerExperimentConfig,
    RerankerVariant,
)
from skill_flow.eval.experiments.grid import (
    get_aggregations,
    get_num_queries_list,
    get_top_k_per_query_list,
    reaggregate_report,
    update_query_gen_aggregation,
    update_query_gen_num_queries,
    update_query_gen_top_k_per_query,
)
from skill_flow.eval.experiments.reranker import (
    _get_content_chars_list,
    _set_content_chars,
    _set_variant_aggregation,
    _set_variant_num_queries,
    run_reranker_experiment,
)
from skill_flow.eval.models import (
    EvalReport,
    EvalSummary,
    RetrievedSkill,
    TaskGroundTruth,
    TaskResult,
)
from skill_flow.eval.utils.reporting import print_reranker_comparison


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


class TestGetAggregations:
    def test_none_query_gen(self) -> None:
        assert get_aggregations(None) == ["max"]

    def test_disabled_query_gen(self) -> None:
        qg = QueryGenConfig(enabled=False, aggregation="mean")
        assert get_aggregations(qg) == ["max"]

    def test_single_string(self) -> None:
        qg = QueryGenConfig(enabled=True, aggregation="rrf")
        assert get_aggregations(qg) == ["rrf"]

    def test_list(self) -> None:
        qg = QueryGenConfig(enabled=True, aggregation=["max", "mean", "rrf"])
        assert get_aggregations(qg) == ["max", "mean", "rrf"]


class TestGetNumQueriesList:
    def test_none_query_gen(self) -> None:
        assert get_num_queries_list(None) == [1]

    def test_disabled(self) -> None:
        qg = QueryGenConfig(enabled=False, num_queries=3)
        assert get_num_queries_list(qg) == [1]

    def test_single_int(self) -> None:
        qg = QueryGenConfig(enabled=True, num_queries=3)
        assert get_num_queries_list(qg) == [3]

    def test_list(self) -> None:
        qg = QueryGenConfig(enabled=True, num_queries=[1, 3, 5])
        assert get_num_queries_list(qg) == [1, 3, 5]


class TestGetTopKPerQueryList:
    def test_none_query_gen(self) -> None:
        assert get_top_k_per_query_list(None) == [0]

    def test_disabled_query_gen(self) -> None:
        qg = QueryGenConfig(enabled=False, top_k_per_query=100)
        assert get_top_k_per_query_list(qg) == [0]

    def test_single_int(self) -> None:
        qg = QueryGenConfig(enabled=True, top_k_per_query=200)
        assert get_top_k_per_query_list(qg) == [200]

    def test_list(self) -> None:
        qg = QueryGenConfig(enabled=True, top_k_per_query=[100, 200])
        assert get_top_k_per_query_list(qg) == [100, 200]

    def test_zero(self) -> None:
        qg = QueryGenConfig(enabled=True, top_k_per_query=0)
        assert get_top_k_per_query_list(qg) == [0]


class TestSetNumQueries:
    def test_with_cache_dir(self) -> None:
        qg = QueryGenConfig(enabled=True, num_queries=[1, 3])
        v = RerankerVariant(model_name="test", query_gen=qg)
        result = _set_variant_num_queries(v, 3, cache_dir="outputs/experiments/")
        assert result.query_gen is not None
        assert result.query_gen.num_queries == 3
        assert (
            result.query_gen.cache_path == "outputs/experiments/query_gen_cache_3.json"
        )

    def test_without_cache_dir_preserves_path(self) -> None:
        qg = QueryGenConfig(enabled=True, num_queries=1, cache_path="my_cache.json")
        v = RerankerVariant(model_name="test", query_gen=qg)
        result = _set_variant_num_queries(v, 5)
        assert result.query_gen is not None
        assert result.query_gen.num_queries == 5
        assert result.query_gen.cache_path == "my_cache.json"

    def test_no_query_gen(self) -> None:
        v = RerankerVariant(model_name="test")
        result = _set_variant_num_queries(v, 5)
        assert result is v


class TestUpdateQueryGenHelpers:
    def test_update_num_queries(self) -> None:
        qg = QueryGenConfig(enabled=True, num_queries=[1, 3])
        result = update_query_gen_num_queries(qg, 5, cache_dir="out/")
        assert result.num_queries == 5
        assert result.cache_path == "out/query_gen_cache_5.json"

    def test_update_aggregation(self) -> None:
        qg = QueryGenConfig(enabled=True, aggregation=["max", "mean"])
        result = update_query_gen_aggregation(qg, "rrf")
        assert result.aggregation == "rrf"

    def test_update_top_k_per_query(self) -> None:
        qg = QueryGenConfig(enabled=True, top_k_per_query=[100, 200])
        result = update_query_gen_top_k_per_query(qg, 200)
        assert result.top_k_per_query == 200


class TestGetContentCharsList:
    def test_single_int(self) -> None:
        v = RerankerVariant(model_name="test", max_content_chars=512)
        assert _get_content_chars_list(v) == [512]

    def test_list(self) -> None:
        v = RerankerVariant(model_name="test", max_content_chars=[512, 1024, 4096])
        assert _get_content_chars_list(v) == [512, 1024, 4096]

    def test_zero(self) -> None:
        v = RerankerVariant(model_name="test", max_content_chars=0)
        assert _get_content_chars_list(v) == [0]


class TestSetContentChars:
    def test_sets_value(self) -> None:
        v = RerankerVariant(model_name="test", max_content_chars=[512, 1024])
        result = _set_content_chars(v, 1024)
        assert result.max_content_chars == 1024

    def test_preserves_other_fields(self) -> None:
        v = RerankerVariant(model_name="test", batch_size=32, max_content_chars=[512])
        result = _set_content_chars(v, 512)
        assert result.batch_size == 32
        assert result.model_name == "test"


class TestSetAggregation:
    def test_sets_aggregation(self) -> None:
        qg = QueryGenConfig(enabled=True, aggregation=["max", "mean"])
        variant = RerankerVariant(model_name="test", query_gen=qg)
        result = _set_variant_aggregation(variant, "rrf")
        assert result.query_gen is not None
        assert result.query_gen.aggregation == "rrf"

    def test_no_query_gen(self) -> None:
        variant = RerankerVariant(model_name="test")
        result = _set_variant_aggregation(variant, "mean")
        assert result is variant


_REAGG_GT: dict[str, TaskGroundTruth] = {
    "t1": TaskGroundTruth(
        task_id="t1",
        query="q",
        ground_truth_keys=["skill-0"],
        all_skill_names=[],
        injected_skills=["skill-0"],
    ),
}
_REAGG_CFG: dict[str, object] = {
    "prev_report_path": "s1.json",
    "tasks_dir": "tasks",
}


def _reagg_source(skills: list[RetrievedSkill]) -> EvalReport:
    tr = TaskResult(
        task_id="t1",
        query="q",
        num_ground_truth=1,
        num_injected=1,
        retrieved_skills=skills,
        recall_at={1: 1.0},
        hit_at={1: 1.0},
        reciprocal_rank=1.0,
    )
    return EvalReport(
        summary=EvalSummary(
            num_tasks_total=1,
            num_tasks_evaluated=1,
            num_tasks_no_skills=0,
            num_skills_injected=1,
            mean_recall_at={1: 1.0},
            mean_hit_at={1: 1.0},
            mrr=1.0,
        ),
        task_results=[tr],
        config=_REAGG_CFG,
    )


class TestReaggregate:
    def test_reaggregate_reorders_by_new_scores(self) -> None:
        source = _reagg_source(
            [
                RetrievedSkill(key="skill-1", score=0.9, query_scores=[0.9, 0.2]),
                RetrievedSkill(key="skill-0", score=0.8, query_scores=[0.1, 0.8]),
            ]
        )
        report = reaggregate_report(
            source,
            "mean",
            _REAGG_GT,
            [1, 2],
            1,
            0,
            1,
            _REAGG_CFG,
        )
        assert len(report.task_results) == 1
        skills = report.task_results[0].retrieved_skills
        assert skills[0].key == "skill-1"
        assert skills[1].key == "skill-0"

    def test_reaggregate_no_query_scores_preserves_order(self) -> None:
        source = _reagg_source(
            [
                RetrievedSkill(key="skill-0", score=0.9),
                RetrievedSkill(key="skill-1", score=0.5),
            ]
        )
        report = reaggregate_report(
            source,
            "max",
            _REAGG_GT,
            [1, 2],
            1,
            0,
            1,
            _REAGG_CFG,
        )
        skills = report.task_results[0].retrieved_skills
        assert skills[0].key == "skill-0"
        assert skills[1].key == "skill-1"


class TestPrintRerankerComparison:
    def test_prints_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [
            ("reranker-a", _make_report(0.8)),
            ("reranker-b-4096chars", _make_report(0.6)),
        ]
        print_reranker_comparison(results)
        captured = capsys.readouterr()
        assert "reranker-a" in captured.out
        assert "0.8000" in captured.out

    def test_empty_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_reranker_comparison([])
        captured = capsys.readouterr()
        assert "Model" in captured.out


class TestSingleQuerySkipsAggregation:
    @patch("skill_flow.eval.experiments.reranker.run_reranker_evaluation")
    def test_single_query_only_uses_max(
        self,
        mock_eval: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_eval.return_value = _make_report(0.7)
        qg = QueryGenConfig(
            enabled=True, num_queries=1, aggregation=["max", "mean", "rrf"]
        )
        config = RerankerExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            input_report_path="outputs/stage1.json",
            rerankers=[
                RerankerVariant(
                    model_name="test-reranker", max_content_chars=512, query_gen=qg
                )
            ],
        )
        results = run_reranker_experiment(config)
        assert len(results) == 1
        assert results[0][0] == "test-reranker-512chars-q1-max"

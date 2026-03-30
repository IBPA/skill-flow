"""Tests for skill_flow.eval.experiments.retriever (retriever experiments)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from skill_flow.config import (
    QueryGenConfig,
    RetrieverExperimentConfig,
    RetrieverVariant,
)
from skill_flow.eval.experiments.grid import (
    get_num_queries_list,
)
from skill_flow.eval.experiments.retriever import (
    _ensure_index,
    _retriever_label,
    _to_cfg,
    _with_aggregation,
    _with_num_queries,
    _with_top_k_per_query,
    print_comparison,
    run_experiment,
)
from skill_flow.eval.models import (
    EvalReport,
    EvalSummary,
    TaskGroundTruth,
)
from skill_flow.retriever.multi_search import CollectedScores
from skill_flow.retriever.retriever import SearchResult


def _make_summary(mrr: float = 0.5) -> EvalSummary:
    return EvalSummary(
        num_tasks_total=10,
        num_tasks_evaluated=10,
        num_tasks_no_skills=0,
        num_skills_injected=5,
        mean_recall_at={1: 0.3, 2: 0.4, 5: 0.6, 10: 0.8, 100: 1.0},
        mean_precision_at={1: 0.3, 2: 0.2, 5: 0.12, 10: 0.08, 100: 0.01},
        mean_hit_at={1: 0.3, 2: 0.4, 5: 0.6, 10: 0.8, 100: 1.0},
        mrr=mrr,
    )


def _make_report(mrr: float = 0.5) -> EvalReport:
    return EvalReport(
        summary=_make_summary(mrr),
        task_results=[],
        config={"tasks_dir": "tasks", "index_dir": "index"},
    )


class TestToCfg:
    def test_builds_config(self) -> None:
        variant = RetrieverVariant(retriever_type="bm25", model_name="bm25", top_k=50)
        config = _to_cfg(variant)
        assert config.retriever_type == "bm25"
        assert config.top_k == 50

    def test_passes_query_gen(self) -> None:
        qg = QueryGenConfig(enabled=True, model="gpt-4o-mini")
        variant = RetrieverVariant(model_name="test", query_gen=qg)
        config = _to_cfg(variant)
        assert config.query_gen is not None
        assert config.query_gen.enabled is True


class TestRunExperiment:
    @patch("skill_flow.eval.experiments.retriever._ensure_index")
    @patch("skill_flow.eval.experiments.retriever.run_evaluation")
    def test_runs_all_variants(
        self,
        mock_eval: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_eval.return_value = _make_report(0.8)
        config = RetrieverExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            retrievers=[
                RetrieverVariant(model_name="model-a", index_dir="out/model-a/"),
                RetrieverVariant(model_name="model-b", index_dir="out/model-b/"),
            ],
        )
        results = run_experiment(config)
        assert len(results) == 2
        assert mock_eval.call_count == 2
        assert mock_ensure.call_count == 2
        assert results[0][0] == "model-a"
        assert results[1][0] == "model-b"

    @patch("skill_flow.eval.experiments.retriever._ensure_index")
    @patch("skill_flow.eval.experiments.retriever.run_evaluation")
    def test_empty_retrievers(
        self,
        mock_eval: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = RetrieverExperimentConfig(
            name="empty",
            output_dir=str(tmp_path),
            retrievers=[],
        )
        results = run_experiment(config)
        assert results == []
        mock_eval.assert_not_called()


class TestPrintComparison:
    def test_prints_table(self, capsys: pytest.CaptureFixture[str]) -> None:
        results = [("bge-base", _make_report(0.8)), ("bm25", _make_report(0.5))]
        print_comparison(results)
        captured = capsys.readouterr()
        assert "bge-base" in captured.out
        assert "0.8000" in captured.out

    def test_empty_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_comparison([])
        captured = capsys.readouterr()
        assert "Model" in captured.out


class TestEnsureIndex:
    def test_dense_skips_when_index_exists(self, tmp_path: Path) -> None:
        (tmp_path / "faiss.index").touch()
        v = RetrieverVariant(
            retriever_type="dense", model_name="test", index_dir=str(tmp_path)
        )
        _ensure_index(v, "/unused/corpus")

    @patch("skill_flow.eval.experiments.retriever.pick_device", return_value="cpu")
    @patch("skill_flow.eval.experiments.retriever.build_index")
    @patch("skill_flow.eval.experiments.retriever.Encoder")
    @patch("skill_flow.eval.experiments.retriever.load_corpus", return_value=[])
    def test_dense_builds_when_missing(
        self,
        mock_load: MagicMock,
        mock_enc: MagicMock,
        mock_build: MagicMock,
        _: MagicMock,
        tmp_path: Path,
    ) -> None:
        v = RetrieverVariant(
            retriever_type="dense", model_name="test", index_dir=str(tmp_path)
        )
        _ensure_index(v, str(tmp_path / "corpus"))
        mock_load.assert_called_once()
        mock_build.assert_called_once()

    def test_bm25_raises_when_artifacts_missing(self, tmp_path: Path) -> None:
        v = RetrieverVariant(
            retriever_type="bm25", model_name="bm25", index_dir=str(tmp_path)
        )
        with pytest.raises(FileNotFoundError, match="BM25 requires"):
            _ensure_index(v, "/unused/corpus")

    def test_bm25_skips_when_artifacts_exist(self, tmp_path: Path) -> None:
        (tmp_path / "skill_ids.json").touch()
        v = RetrieverVariant(
            retriever_type="bm25", model_name="bm25", index_dir=str(tmp_path)
        )
        _ensure_index(v, "/unused/corpus")


class TestRetrieverQueryGenHelpers:
    def test_get_num_queries_no_qg(self) -> None:
        assert get_num_queries_list(None) == [1]

    def test_set_num_queries(self) -> None:
        qg = QueryGenConfig(enabled=True, num_queries=[1, 3])
        v = RetrieverVariant(query_gen=qg)
        result = _with_num_queries(v, 3)
        assert result.query_gen is not None
        assert result.query_gen.num_queries == 3

    def test_set_aggregation(self) -> None:
        qg = QueryGenConfig(enabled=True, aggregation=["max", "mean"])
        v = RetrieverVariant(query_gen=qg)
        result = _with_aggregation(v, "mean")
        assert result.query_gen is not None
        assert result.query_gen.aggregation == "mean"

    def test_set_top_k_per_query(self) -> None:
        qg = QueryGenConfig(enabled=True, top_k_per_query=[100, 200])
        v = RetrieverVariant(query_gen=qg)
        result = _with_top_k_per_query(v, 200)
        assert result.query_gen is not None
        assert result.query_gen.top_k_per_query == 200


class TestRetrieverLabel:
    def test_without_query_gen(self) -> None:
        v = RetrieverVariant(index_dir="outputs/indices/bge-base/")
        assert _retriever_label(v) == "bge-base"

    def test_bm25_prefix(self) -> None:
        v = RetrieverVariant(
            retriever_type="bm25", index_dir="outputs/indices/bge-base/"
        )
        assert _retriever_label(v) == "bm25-bge-base"

    def test_with_query_gen(self) -> None:
        qg = QueryGenConfig(enabled=True, num_queries=3, aggregation="max")
        v = RetrieverVariant(index_dir="outputs/indices/bge-base/", query_gen=qg)
        assert _retriever_label(v) == "bge-base-q3-max"

    def test_union_label_includes_top_k_per_query(self) -> None:
        qg = QueryGenConfig(
            enabled=True, num_queries=5, aggregation="union", top_k_per_query=200
        )
        v = RetrieverVariant(index_dir="outputs/indices/bge-base/", query_gen=qg)
        assert _retriever_label(v) == "bge-base-q5-union-tk200"


class TestRunExperimentGridSearch:
    @patch("skill_flow.eval.experiments.retriever._ensure_index")
    @patch("skill_flow.eval.experiments.retriever.run_evaluation")
    def test_no_query_gen_runs_once(
        self,
        mock_eval: MagicMock,
        mock_ensure: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_eval.return_value = _make_report(0.8)
        config = RetrieverExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            retrievers=[RetrieverVariant(index_dir="out/bge-base/")],
        )
        results = run_experiment(config)
        assert len(results) == 1
        assert results[0][0] == "bge-base"

    @patch("skill_flow.eval.experiments.retriever.QueryGenerator")
    @patch("skill_flow.eval.experiments.retriever.collect_multi_scores")
    @patch("skill_flow.eval.experiments.retriever._augment_searcher")
    @patch("skill_flow.eval.experiments.retriever.build_searcher")
    @patch("skill_flow.eval.experiments.retriever.load_ground_truth")
    @patch("skill_flow.eval.experiments.retriever._ensure_index")
    @patch("skill_flow.eval.experiments.retriever.run_evaluation")
    def test_grid_search_num_queries_x_aggregation(
        self,
        mock_eval: MagicMock,
        mock_ensure: MagicMock,
        mock_gt: MagicMock,
        mock_bs: MagicMock,
        mock_aug: MagicMock,
        mock_collect: MagicMock,
        mock_qgen_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_eval.return_value = _make_report(0.7)
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
        mock_qgen = MagicMock()
        mock_qgen.generate_multi.return_value = ["q1", "q2", "q3"]
        mock_qgen_cls.return_value = mock_qgen
        mock_collect.return_value = CollectedScores(
            per_key_scores={"s0": [0.9, 0.1, 0.5]},
            metadata={"s0": SearchResult(key="s0", score=0.9, description="d")},
            per_query_results=[[SearchResult(key="s0", score=0.9, description="d")]]
            * 3,
            queries=["q1", "q2", "q3"],
        )
        qg = QueryGenConfig(
            enabled=True, num_queries=[1, 3], aggregation=["max", "mean"]
        )
        config = RetrieverExperimentConfig(
            name="test",
            tasks_dir="tasks",
            output_dir=str(tmp_path),
            retrievers=[RetrieverVariant(index_dir="out/bge-base/", query_gen=qg)],
        )
        results = run_experiment(config)
        assert len(results) == 3
        labels = [r[0] for r in results]
        assert "bge-base-q1-max" in labels
        assert "bge-base-q3-max" in labels
        assert "bge-base-q3-mean" in labels

"""Tests for analysis.results.t5_generate_latency."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_mod = importlib.import_module("analysis.results.t5_generate_latency")
load_latency_data = _mod.load_latency_data
render_latex_table = _mod.render_latex_table


class TestLoadLatencyData:
    def test_loads_from_summary_json(self, tmp_path: Path) -> None:
        summary = {
            "stages": {
                "retriever": {
                    "per_task_ms": {"t1": 10.0},
                    "aggregate": {"median_ms": 10.0, "mean_ms": 10.0, "p95_ms": 10.0},
                },
            }
        }
        (tmp_path / "latency_summary.json").write_text(json.dumps(summary))

        result = load_latency_data(tmp_path)

        assert "retriever" in result
        assert result["retriever"]["aggregate"]["median_ms"] == 10.0

    def test_falls_back_to_stage_files(self, tmp_path: Path) -> None:
        stage_data = {
            "stage": "reranker",
            "num_tasks": 1,
            "task_results": [
                {"task_id": "t1", "query": "q", "skills": [], "elapsed_ms": 25.0},
            ],
        }
        (tmp_path / "pipeline-stage2-reranker.json").write_text(json.dumps(stage_data))

        result = load_latency_data(tmp_path)

        assert "reranker" in result
        assert result["reranker"]["aggregate"]["median_ms"] == 25.0

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = load_latency_data(tmp_path)
        assert result == {}


class TestRenderLatexTable:
    def _make_stages(
        self,
        **kwargs: dict[str, float],
    ) -> dict[str, dict[str, object]]:
        """Build stages dict in the format expected by render_latex_table."""
        return {
            name: {
                "per_task_ms": {"t1": agg["median_ms"]},
                "aggregate": agg,
            }
            for name, agg in kwargs.items()
        }

    def test_renders_known_stages_in_seconds(self) -> None:
        stages = self._make_stages(
            retriever={"median_ms": 10000.0, "mean_ms": 12000.0, "p95_ms": 20000.0},
            selector={"median_ms": 100000.0, "mean_ms": 120000.0, "p95_ms": 200000.0},
        )

        table = render_latex_table(stages)

        assert "Dense Retriever" in table
        assert "LLM Selector" in table
        assert "10.0" in table
        assert "200.0" in table

    def test_includes_total_row(self) -> None:
        stages = self._make_stages(
            retriever={"median_ms": 1000.0, "mean_ms": 2000.0, "p95_ms": 3000.0},
            selector={"median_ms": 4000.0, "mean_ms": 5000.0, "p95_ms": 6000.0},
        )

        table = render_latex_table(stages)

        assert "Total" in table
        assert r"\textbf{5.0}" in table

    def test_skips_missing_stages(self) -> None:
        stages = self._make_stages(
            retriever={"median_ms": 5000.0, "mean_ms": 5000.0, "p95_ms": 5000.0},
        )

        table = render_latex_table(stages)

        assert "Reranker" not in table

    def test_contains_tabular_structure(self) -> None:
        stages = self._make_stages(
            retriever={"median_ms": 1000.0, "mean_ms": 1000.0, "p95_ms": 1000.0},
        )

        table = render_latex_table(stages)

        assert r"\begin{tabular}" in table
        assert r"\end{tabular}" in table

    def test_bold_column_headers(self) -> None:
        stages = self._make_stages(
            retriever={"median_ms": 1000.0, "mean_ms": 1000.0, "p95_ms": 1000.0},
        )

        table = render_latex_table(stages)

        assert r"\textbf{Stage}" in table
        assert r"\textbf{Median}" in table
        assert r"\textbf{Mean}" in table
        assert r"\textbf{P95}" in table

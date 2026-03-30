"""Tests for skill_flow.pipeline.cache."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from skill_flow.pipeline.cache import (
    find_latest_cache,
    load_content_maps,
    load_stage_cache,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadContentMaps:
    def test_loads_both_files(self, tmp_path: Path) -> None:
        (tmp_path / "skill_descriptions.json").write_text(json.dumps({"k1": "desc1"}))
        (tmp_path / "skill_contents.json").write_text(json.dumps({"k1": "content1"}))

        descs, contents = load_content_maps(tmp_path)

        assert descs == {"k1": "desc1"}
        assert contents == {"k1": "content1"}

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        descs, contents = load_content_maps(tmp_path)

        assert descs == {}
        assert contents == {}


class TestLoadStageCache:
    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        result = load_stage_cache(tmp_path / "nonexistent.json", {"t1"}, {}, {})

        assert result is None

    def test_returns_none_when_tasks_missing(self, tmp_path: Path) -> None:
        path = tmp_path / "stage.json"
        data = {
            "stage": "retriever",
            "num_tasks": 1,
            "task_results": [
                {"task_id": "t1", "query": "q", "skills": []},
            ],
        }
        path.write_text(json.dumps(data))

        result = load_stage_cache(path, {"t1", "t2"}, {}, {})

        assert result is None

    def test_loads_valid_cache(self, tmp_path: Path) -> None:
        path = tmp_path / "stage.json"
        data = {
            "stage": "retriever",
            "num_tasks": 1,
            "task_results": [
                {
                    "task_id": "t1",
                    "query": "q",
                    "skills": [{"key": "k1", "score": 0.9, "rank": 1}],
                }
            ],
        }
        path.write_text(json.dumps(data))
        descriptions = {"k1": "desc"}
        contents = {"k1": "full content"}

        result = load_stage_cache(path, {"t1"}, descriptions, contents)

        assert result is not None
        assert len(result["t1"]) == 1
        assert result["t1"][0].key == "k1"
        assert result["t1"][0].description == "desc"
        assert result["t1"][0].content == "full content"

    def test_filters_to_requested_tasks(self, tmp_path: Path) -> None:
        path = tmp_path / "stage.json"
        data = {
            "stage": "retriever",
            "num_tasks": 2,
            "task_results": [
                {"task_id": "t1", "query": "q1", "skills": []},
                {"task_id": "t2", "query": "q2", "skills": []},
            ],
        }
        path.write_text(json.dumps(data))

        result = load_stage_cache(path, {"t1"}, {}, {})

        assert result is not None
        assert "t1" in result
        assert "t2" not in result


class TestFindLatestCache:
    def test_returns_zero_when_no_cache(self, tmp_path: Path) -> None:
        cached, stage = find_latest_cache(tmp_path, {"t1"}, {}, {}, 4)

        assert cached is None
        assert stage == 0

    def test_finds_latest_stage(self, tmp_path: Path) -> None:
        stage_data = {
            "stage": "reranker",
            "num_tasks": 1,
            "task_results": [
                {
                    "task_id": "t1",
                    "query": "q",
                    "skills": [{"key": "k1", "score": 0.9, "rank": 1}],
                }
            ],
        }
        (tmp_path / "pipeline-stage1-retriever.json").write_text(
            json.dumps(stage_data),
        )
        (tmp_path / "pipeline-stage2-reranker.json").write_text(
            json.dumps(stage_data),
        )

        cached, stage = find_latest_cache(tmp_path, {"t1"}, {}, {}, 4)

        assert cached is not None
        assert stage == 2

    def test_respects_max_stage(self, tmp_path: Path) -> None:
        stage_data = {
            "stage": "test",
            "num_tasks": 1,
            "task_results": [
                {
                    "task_id": "t1",
                    "query": "q",
                    "skills": [],
                }
            ],
        }
        (tmp_path / "pipeline-stage1-retriever.json").write_text(
            json.dumps(stage_data),
        )
        (tmp_path / "pipeline-stage3-deep-reranker.json").write_text(
            json.dumps(stage_data),
        )

        # max_stage=2 means stage 3 cache should be ignored
        _cached, stage = find_latest_cache(tmp_path, {"t1"}, {}, {}, 2)

        assert stage == 1

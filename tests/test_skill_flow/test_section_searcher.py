"""Tests for skill_flow.retriever.section_searcher."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from skill_flow.retriever.retriever import SearchResult
from skill_flow.retriever.section_searcher import (
    _MISSING_SCORE,
    SectionSearcher,
    _fuse,
)


def _result(key: str, score: float) -> SearchResult:
    return SearchResult(key=key, score=score)


def _stub_searcher(results_by_query: dict[str, list[SearchResult]]) -> MagicMock:
    s = MagicMock()
    s.search.side_effect = lambda query, top_k=None: results_by_query[query][:top_k]
    return s


def _searcher(per_query: list[SearchResult]) -> MagicMock:
    s = MagicMock()
    s.search.return_value = per_query
    return s


def test_fuse_max_picks_highest_per_key() -> None:
    per_section = {
        "yaml": {"a": 0.8, "b": 0.3},
        "prose": {"a": 0.2, "b": 0.9},
        "code": {"a": 0.5, "c": 0.7},
    }
    fused = _fuse(per_section, "max")
    assert fused["a"] == pytest.approx(0.8)
    assert fused["b"] == pytest.approx(0.9)
    assert fused["c"] == pytest.approx(0.7)


def test_fuse_mean_includes_missing_penalty() -> None:
    per_section = {
        "yaml": {"a": 0.6},
        "prose": {"a": 0.4},
        "code": {"a": 0.8},
    }
    fused = _fuse(per_section, "mean")
    assert fused["a"] == pytest.approx((0.6 + 0.4 + 0.8) / 3)

    # Missing key gets the floor penalty.
    per_section_missing = {
        "yaml": {"x": 0.9},
        "prose": {},
        "code": {},
    }
    fused = _fuse(per_section_missing, "mean")
    assert fused["x"] == pytest.approx((0.9 + _MISSING_SCORE + _MISSING_SCORE) / 3)


def test_fuse_rrf_ranks_per_section() -> None:
    per_section = {
        "yaml": {"a": 0.9, "b": 0.5, "c": 0.1},
        "prose": {"a": 0.8, "b": 0.7, "c": 0.6},
        "code": {"a": 0.3, "b": 0.9, "c": 0.2},
    }
    fused = _fuse(per_section, "rrf")
    # 'b' is rank 2,2,1 → 1/62 + 1/62 + 1/61 ≈ 0.04866
    # 'a' is rank 1,1,2 → 1/61 + 1/61 + 1/62 ≈ 0.04891
    # 'c' is rank 3,3,3 → 3/63
    assert fused["a"] > fused["b"] > fused["c"]


def test_fuse_sum_norm_collapses_to_zero_when_section_flat() -> None:
    per_section = {
        "yaml": {"a": 0.5, "b": 0.5},  # flat → all-zero contribution
        "prose": {"a": 0.9, "b": 0.1},  # span 0.8
        "code": {"a": 0.0, "b": 1.0},
    }
    fused = _fuse(per_section, "sum_norm")
    # yaml contributes 0+0; prose contributes 1+0; code contributes 0+1
    assert fused["a"] == pytest.approx(1.0)
    assert fused["b"] == pytest.approx(1.0)


def test_search_late_fuses_three_indices() -> None:
    yaml = _searcher([_result("a", 0.9), _result("b", 0.4)])
    prose = _searcher([_result("a", 0.7), _result("b", 0.6)])
    code = _searcher([_result("a", 0.3), _result("b", 0.8)])
    ss = SectionSearcher(
        sub_searchers={"yaml": yaml, "prose": prose, "code": code},
        descriptions={"a": "desc-a", "b": "desc-b"},
        contents={"a": "full-a", "b": "full-b"},
        aggregation="max",
        pool_size=100,
    )
    out = ss.search("query", top_k=2)
    keys = [r.key for r in out]
    assert set(keys) == {"a", "b"}
    by_key = {r.key: r for r in out}
    assert by_key["a"].score == pytest.approx(0.9)
    assert by_key["b"].score == pytest.approx(0.8)
    assert by_key["a"].description == "desc-a"
    assert by_key["a"].content == "full-a"


def test_add_contents_splits_and_augments_each_subindex() -> None:
    yaml = MagicMock()
    prose = MagicMock()
    code = MagicMock()
    ss = SectionSearcher(
        sub_searchers={"yaml": yaml, "prose": prose, "code": code},
        descriptions={},
        contents={},
    )
    md = "---\nname: x\n---\n\nSome prose.\n\n```python\nprint(1)\n```\n"
    ss.add_contents({"injected/x": md})
    yaml.augment.assert_called_once()
    prose.augment.assert_called_once()
    code.augment.assert_called_once()
    yaml_text = yaml.augment.call_args.args[1][0]
    prose_text = prose.augment.call_args.args[1][0]
    code_text = code.augment.call_args.args[1][0]
    assert "name: x" in yaml_text
    assert "Some prose." in prose_text
    assert "print(1)" in code_text
    assert ss._contents["injected/x"] == md


def test_augment_is_noop() -> None:
    yaml = MagicMock()
    prose = MagicMock()
    code = MagicMock()
    ss = SectionSearcher(
        sub_searchers={"yaml": yaml, "prose": prose, "code": code},
        descriptions={},
        contents={},
    )
    ss.augment(["k"], ["d"])
    yaml.augment.assert_not_called()
    prose.augment.assert_not_called()
    code.augment.assert_not_called()


def test_add_descriptions_updates_shared_map() -> None:
    ss = SectionSearcher(
        sub_searchers={
            "yaml": MagicMock(),
            "prose": MagicMock(),
            "code": MagicMock(),
        },
        descriptions={"existing": "old"},
        contents={},
    )
    ss.add_descriptions({"new": "fresh", "existing": "updated"})
    assert ss._descriptions == {"existing": "updated", "new": "fresh"}

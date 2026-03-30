"""Tests for skill quality proxy comparison utilities."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from analysis.results.utils.quality_comparison_utils import (
    ComparisonResult,
    compare_populations,
    load_community_skills,
    load_oracle_skills,
)
from analysis.results.utils.quality_utils import SkillMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "code_fraction": 0.0,
    "ordered_list_count": 0,
    "inline_code_density": 0.0,
    "bundled_file_count": 0,
    "has_bundled_files": False,
    "has_scripts": False,
    "has_references": False,
}


def _make_metrics(
    wc: int = 100,
    hc: int = 3,
    cc: int = 1,
    ex: bool = True,
    fm: bool = True,
    **overrides: object,
) -> SkillMetrics:
    vals = {
        "word_count": wc,
        "heading_count": hc,
        "code_block_count": cc,
        "has_examples": ex,
        "has_yaml_frontmatter": fm,
        **_DEFAULTS,
        **overrides,
    }
    return SkillMetrics(**vals)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Population comparison
# ---------------------------------------------------------------------------


class TestComparePopulations:
    def test_basic_comparison(self) -> None:
        oracle = [_make_metrics(wc=200, hc=5, cc=3) for _ in range(10)]
        community = [
            _make_metrics(wc=50, hc=1, cc=0, ex=False, fm=False) for _ in range(10)
        ]
        results = compare_populations(oracle, community)
        assert len(results) == len(SkillMetrics.model_fields) - 3  # 3 excluded
        wc_result = results[0]
        assert wc_result.metric == "word_count"
        assert wc_result.oracle.mean > wc_result.community.mean
        assert wc_result.p_value < 0.05
        # Adjusted p should be >= raw p
        assert wc_result.p_adjusted >= wc_result.p_value

    def test_identical_populations(self) -> None:
        both = [_make_metrics() for _ in range(10)]
        results = compare_populations(both, both)
        for r in results:
            assert r.p_value == 1.0
            assert r.p_adjusted == 1.0

    def test_empty_oracle(self) -> None:
        community = [_make_metrics(wc=50)]
        results = compare_populations([], community)
        assert len(results) == len(SkillMetrics.model_fields) - 3
        for r in results:
            assert r.p_value == 1.0
            assert r.p_adjusted == 1.0
            assert r.oracle.n == 0

    def test_result_model_fields(self) -> None:
        oracle = [_make_metrics(wc=100)]
        community = [_make_metrics(wc=50)]
        results = compare_populations(oracle, community)
        r = results[0]
        assert isinstance(r, ComparisonResult)
        assert r.oracle.n == 1
        assert r.community.n == 1
        assert r.p_adjusted >= r.p_value


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


class TestLoadOracleSkills:
    def test_loads_from_task_dirs(self, tmp_path: Path) -> None:
        task = tmp_path / "task-1" / "environment" / "skills" / "my-skill"
        task.mkdir(parents=True)
        (task / "SKILL.md").write_text("# Skill\nContent here")
        result = load_oracle_skills(tmp_path)
        assert "task-1/my-skill" in result
        text, skill_dir = result["task-1/my-skill"]
        assert "Content here" in text
        assert skill_dir == task

    def test_empty_dir(self, tmp_path: Path) -> None:
        assert load_oracle_skills(tmp_path) == {}

    def test_skips_missing_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "task-1" / "environment" / "skills" / "no-skill"
        skill_dir.mkdir(parents=True)
        assert load_oracle_skills(tmp_path) == {}


class TestLoadCommunitySkills:
    def test_from_json(self, tmp_path: Path) -> None:
        data = {"skill-a": "# A\ncontent", "skill-b": "# B\ncontent"}
        json_path = tmp_path / "skill_contents.json"
        json_path.write_text(json.dumps(data))
        result = load_community_skills(contents_path=json_path)
        assert len(result) == 2
        assert "skill-a" in result

    def test_from_corpus_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "author" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# My Skill")
        result = load_community_skills(corpus_dir=tmp_path)
        assert len(result) == 1
        text, path = next(iter(result.values()))
        assert "# My Skill" in text
        assert path == skill_dir

    def test_no_sources(self) -> None:
        result = load_community_skills()
        assert result == {}

    def test_corpus_preferred_over_json(self, tmp_path: Path) -> None:
        data = {"from-json": "json content"}
        json_path = tmp_path / "skill_contents.json"
        json_path.write_text(json.dumps(data))
        skill_dir = tmp_path / "corpus" / "from-dir"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("dir content")
        result = load_community_skills(
            contents_path=json_path,
            corpus_dir=tmp_path / "corpus",
        )
        assert "from-dir" in result
        assert len(result) == 1

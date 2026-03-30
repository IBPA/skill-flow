"""Tests for quality_metrics: text-based and bundle-based proxy metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from analysis.results.utils.quality_utils import (
    code_block_count,
    code_fraction,
    compute_bundle_metrics,
    compute_metrics,
    compute_text_metrics,
    has_bundled_files,
    has_examples,
    has_references,
    has_scripts,
    has_yaml_frontmatter,
    heading_count,
    inline_code_density,
    ordered_list_count,
    word_count,
)

if TYPE_CHECKING:
    from pathlib import Path

SAMPLE_SKILL = """\
---
name: test-skill
description: A test skill
---

# Overview

This skill demonstrates how to use pytest.

## Example

```python
def test_hello():
    assert True
```

### Tips

1. Use fixtures for setup.
2. Use `parametrize` for variants.
"""


# ---------------------------------------------------------------------------
# Text-based metrics
# ---------------------------------------------------------------------------


class TestWordCount:
    def test_basic(self) -> None:
        assert word_count("hello world") == 2

    def test_empty(self) -> None:
        assert word_count("") == 0

    def test_multiline(self) -> None:
        assert word_count("one\ntwo three") == 3


class TestHeadingCount:
    def test_multiple_levels(self) -> None:
        assert heading_count("# H1\n## H2\n### H3\nnot a heading") == 3

    def test_empty(self) -> None:
        assert heading_count("") == 0

    def test_indented_heading(self) -> None:
        assert heading_count("  # Indented") == 1


class TestCodeBlockCount:
    def test_single_block(self) -> None:
        assert code_block_count("```python\nprint('hi')\n```") == 1

    def test_no_blocks(self) -> None:
        assert code_block_count("no code here") == 0

    def test_multiple_blocks(self) -> None:
        assert code_block_count("```\na\n```\n\n```\nb\n```") == 2

    def test_odd_backticks_counts_pairs(self) -> None:
        assert code_block_count("```\na\n```\n```\n") == 1


class TestHasExamples:
    def test_lowercase(self) -> None:
        assert has_examples("here is an example") is True

    def test_eg(self) -> None:
        assert has_examples("use e.g. this") is True

    def test_absent(self) -> None:
        assert has_examples("nothing relevant here") is False


class TestHasYamlFrontmatter:
    def test_present(self) -> None:
        assert has_yaml_frontmatter("---\nname: x\n---\nbody") is True

    def test_absent(self) -> None:
        assert has_yaml_frontmatter("# Just a heading") is False


class TestCodeFraction:
    def test_with_code(self) -> None:
        text = "intro\n```\ncode here\n```\noutro"
        ratio = code_fraction(text)
        assert 0.0 < ratio < 1.0

    def test_no_code(self) -> None:
        assert code_fraction("just text") == 0.0

    def test_empty(self) -> None:
        assert code_fraction("") == 0.0


class TestOrderedListCount:
    def test_numbered_items(self) -> None:
        text = "1. First\n2. Second\n3. Third"
        assert ordered_list_count(text) == 3

    def test_no_lists(self) -> None:
        assert ordered_list_count("no lists here") == 0

    def test_indented(self) -> None:
        assert ordered_list_count("  1. indented item") == 1


class TestInlineCodeDensity:
    def test_with_spans(self) -> None:
        text = "Run `foo` then `bar` to finish"
        density = inline_code_density(text)
        # 2 spans / 6 words * 100 ≈ 33.3
        assert 30.0 < density < 40.0

    def test_no_spans(self) -> None:
        assert inline_code_density("no code spans") == 0.0

    def test_empty(self) -> None:
        assert inline_code_density("") == 0.0


# ---------------------------------------------------------------------------
# Bundle-based metrics
# ---------------------------------------------------------------------------


class TestBundleMetrics:
    def test_has_bundled_files(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text("# Skill")
        (tmp_path / "helper.py").write_text("pass")
        assert has_bundled_files(tmp_path) is True

    def test_no_bundled_files(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text("# Skill")
        assert has_bundled_files(tmp_path) is False

    def test_has_scripts(self, tmp_path: Path) -> None:
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "run.py").write_text("pass")
        assert has_scripts(tmp_path) is True

    def test_has_references(self, tmp_path: Path) -> None:
        refs = tmp_path / "references"
        refs.mkdir()
        (refs / "guide.md").write_text("# Guide")
        (tmp_path / "SKILL.md").write_text("# Skill")
        assert has_references(tmp_path) is True

    def test_skill_md_not_counted_as_reference(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text("# Skill")
        assert has_references(tmp_path) is False

    def test_compute_bundle_metrics(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text("# Skill")
        (tmp_path / "run.sh").write_text("echo hi")
        (tmp_path / "notes.md").write_text("# Notes")
        bm = compute_bundle_metrics(tmp_path)
        assert bm.bundled_file_count == 2
        assert bm.has_bundled_files is True
        assert bm.has_scripts is True
        assert bm.has_references is True


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class TestComputeTextMetrics:
    def test_sample_skill(self) -> None:
        m = compute_text_metrics(SAMPLE_SKILL)
        assert m.word_count > 0
        assert m.heading_count == 3
        assert m.code_block_count == 1
        assert m.has_examples is True
        assert m.has_yaml_frontmatter is True
        assert m.ordered_list_count == 2
        assert m.inline_code_density > 0.0
        assert m.code_fraction > 0.0


class TestComputeMetrics:
    def test_combined(self, tmp_path: Path) -> None:
        (tmp_path / "SKILL.md").write_text(SAMPLE_SKILL)
        (tmp_path / "helper.py").write_text("pass")
        m = compute_metrics(SAMPLE_SKILL, tmp_path)
        assert m.word_count > 0
        assert m.has_bundled_files is True
        assert m.has_scripts is True
        assert m.bundled_file_count == 1

"""Structural proxy quality metrics for SKILL.md files.

Each function computes a single metric from either the skill text or the
skill directory.  ``compute_text_metrics`` and ``compute_bundle_metrics``
aggregate these into frozen Pydantic models suitable for population-level
statistical comparison.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

# Script / reference file extensions used by ``has_scripts`` / ``has_references``.
_SCRIPT_EXTS = frozenset({"py", "sh", "js", "ts", "mjs", "rb", "go", "java"})
_REFERENCE_EXTS = frozenset({"md"})


# ---------------------------------------------------------------------------
# Text-based metrics (computed from SKILL.md content only)
# ---------------------------------------------------------------------------


def word_count(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(text.split())


def heading_count(text: str) -> int:
    """Count Markdown headings (lines starting with ``#``)."""
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("#"))


def code_block_count(text: str) -> int:
    """Count fenced code blocks (triple-backtick pairs)."""
    return len(re.findall(r"^```", text, re.MULTILINE)) // 2


def has_examples(text: str) -> bool:
    """Check for example-related content."""
    lower = text.lower()
    return "example" in lower or "e.g." in lower


def has_yaml_frontmatter(text: str) -> bool:
    """Check if text starts with YAML frontmatter delimiter."""
    return text.startswith("---")


def code_fraction(text: str) -> float:
    """Fraction of total characters inside fenced code blocks."""
    blocks = re.findall(r"^```[^\n]*\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
    code_chars = sum(len(b) for b in blocks)
    total = len(text)
    return code_chars / total if total > 0 else 0.0


def ordered_list_count(text: str) -> int:
    """Count ordered-list items (``1.``, ``2.``, …)."""
    return len(re.findall(r"^\s*\d+\.\s", text, re.MULTILINE))


def inline_code_density(text: str) -> float:
    """Inline code spans per 100 words."""
    spans = len(re.findall(r"(?<!`)`(?!`)[^`]+`(?!`)", text))
    words = word_count(text)
    return (spans / words) * 100 if words > 0 else 0.0


# ---------------------------------------------------------------------------
# Bundle-based metrics (computed from the skill directory)
# ---------------------------------------------------------------------------


def bundled_file_count(skill_dir: Path) -> int:
    """Count files in the skill directory beyond SKILL.md."""
    return sum(1 for f in skill_dir.rglob("*") if f.is_file() and f.name != "SKILL.md")


def has_bundled_files(skill_dir: Path) -> bool:
    """Whether the skill directory contains any file besides SKILL.md."""
    return bundled_file_count(skill_dir) > 0


def has_scripts(skill_dir: Path) -> bool:
    """Whether the bundle contains script files (.py, .sh, .js, …)."""
    return any(
        f.suffix.lstrip(".") in _SCRIPT_EXTS
        for f in skill_dir.rglob("*")
        if f.is_file()
    )


def has_references(skill_dir: Path) -> bool:
    """Whether the bundle contains extra .md reference files."""
    return any(
        f.suffix.lstrip(".") in _REFERENCE_EXTS and f.name != "SKILL.md"
        for f in skill_dir.rglob("*")
        if f.is_file()
    )


# ---------------------------------------------------------------------------
# Aggregate models
# ---------------------------------------------------------------------------


class TextMetrics(BaseModel, frozen=True):
    """Proxy quality metrics derived from SKILL.md content."""

    word_count: int
    heading_count: int
    code_block_count: int
    has_examples: bool
    has_yaml_frontmatter: bool
    code_fraction: float
    ordered_list_count: int
    inline_code_density: float


class BundleMetrics(BaseModel, frozen=True):
    """Proxy quality metrics derived from the skill directory structure."""

    bundled_file_count: int
    has_bundled_files: bool
    has_scripts: bool
    has_references: bool


class SkillMetrics(BaseModel, frozen=True):
    """Combined text + bundle metrics for a single skill."""

    word_count: int
    heading_count: int
    code_block_count: int
    has_examples: bool
    has_yaml_frontmatter: bool
    code_fraction: float
    ordered_list_count: int
    inline_code_density: float
    bundled_file_count: int
    has_bundled_files: bool
    has_scripts: bool
    has_references: bool


def compute_text_metrics(text: str) -> TextMetrics:
    """Compute all text-based proxy metrics for a SKILL.md."""
    return TextMetrics(
        word_count=word_count(text),
        heading_count=heading_count(text),
        code_block_count=code_block_count(text),
        has_examples=has_examples(text),
        has_yaml_frontmatter=has_yaml_frontmatter(text),
        code_fraction=code_fraction(text),
        ordered_list_count=ordered_list_count(text),
        inline_code_density=inline_code_density(text),
    )


def compute_bundle_metrics(skill_dir: Path) -> BundleMetrics:
    """Compute all bundle-based proxy metrics for a skill directory."""
    return BundleMetrics(
        bundled_file_count=bundled_file_count(skill_dir),
        has_bundled_files=has_bundled_files(skill_dir),
        has_scripts=has_scripts(skill_dir),
        has_references=has_references(skill_dir),
    )


def compute_metrics(text: str, skill_dir: Path) -> SkillMetrics:
    """Compute all proxy metrics (text + bundle) for a single skill."""
    tm = compute_text_metrics(text)
    bm = compute_bundle_metrics(skill_dir)
    return SkillMetrics(**tm.model_dump(), **bm.model_dump())


def compute_metrics_text_only(text: str) -> SkillMetrics:
    """Compute metrics from text only; bundle metrics are zeroed."""
    tm = compute_text_metrics(text)
    return SkillMetrics(
        **tm.model_dump(),
        bundled_file_count=0,
        has_bundled_files=False,
        has_scripts=False,
        has_references=False,
    )

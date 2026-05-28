"""Tests for skill_flow.index.section_builder."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest
from skill_flow.index.section_builder import build_section_indices
from skill_flow.models import SkillRecord

if TYPE_CHECKING:
    from pathlib import Path

RNG = np.random.default_rng(0)
DIM = 8


def _make_encoder() -> MagicMock:
    enc = MagicMock()

    def _encode(texts: list[str], batch_size: int | None = None) -> np.ndarray:
        v = RNG.random((len(texts), DIM)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    enc.encode_documents.side_effect = _encode
    return enc


def _write_skill(corpus_path: Path, key: str, md: str) -> SkillRecord:
    local = corpus_path / key
    local.mkdir(parents=True, exist_ok=True)
    (local / "SKILL.md").write_text(md, encoding="utf-8")
    return SkillRecord(
        key=key,
        name=key.split("/")[-1],
        description=f"desc-{key}",
        source="skillsmp",
        local_path=key,
    )


@pytest.fixture()
def small_corpus(tmp_path: Path) -> tuple[Path, list[SkillRecord]]:
    skills = [
        _write_skill(
            tmp_path,
            f"skillsmp/skill-{i}",
            f"---\nname: skill-{i}\n---\n\nProse {i}.\n\n```python\nprint({i})\n```\n",
        )
        for i in range(4)
    ]
    return tmp_path, skills


def test_builds_three_subdirs(
    small_corpus: tuple[Path, list[SkillRecord]], tmp_path: Path
) -> None:
    corpus, skills = small_corpus
    parent = tmp_path / "section"
    encoders = {
        "yaml": _make_encoder(),
        "prose": _make_encoder(),
        "code": _make_encoder(),
    }
    build_section_indices(skills, parent, encoders, corpus, batch_size=4)
    for sub in ("yaml", "prose", "code"):
        sub_dir = parent / sub
        assert (sub_dir / "faiss.index").exists()
        assert (sub_dir / "embeddings.npy").exists()
        assert (sub_dir / "skill_ids.json").exists()
        assert (sub_dir / "skill_section_texts.json").exists()
        index = faiss.read_index(str(sub_dir / "faiss.index"))
        assert index.ntotal == len(skills)


def test_shared_metadata_written(
    small_corpus: tuple[Path, list[SkillRecord]], tmp_path: Path
) -> None:
    corpus, skills = small_corpus
    parent = tmp_path / "section"
    encoders = {
        "yaml": _make_encoder(),
        "prose": _make_encoder(),
        "code": _make_encoder(),
    }
    build_section_indices(skills, parent, encoders, corpus, batch_size=4)
    descs = json.loads((parent / "skill_descriptions.json").read_text())
    contents = json.loads((parent / "skill_contents.json").read_text())
    assert set(descs) == {s.key for s in skills}
    assert set(contents) == {s.key for s in skills}
    assert all("print" in v for v in contents.values())


def test_partial_section_build(
    small_corpus: tuple[Path, list[SkillRecord]], tmp_path: Path
) -> None:
    """Passing encoders for only some sections skips the others."""
    corpus, skills = small_corpus
    parent = tmp_path / "section"
    encoders = {"code": _make_encoder()}
    build_section_indices(skills, parent, encoders, corpus, batch_size=4)
    assert (parent / "code" / "faiss.index").exists()
    assert not (parent / "yaml").exists()
    assert not (parent / "prose").exists()


def test_section_texts_match_splitter(
    small_corpus: tuple[Path, list[SkillRecord]], tmp_path: Path
) -> None:
    corpus, skills = small_corpus
    parent = tmp_path / "section"
    encoders = {"prose": _make_encoder(), "code": _make_encoder()}
    build_section_indices(skills, parent, encoders, corpus, batch_size=4)
    prose_texts = json.loads(
        (parent / "prose" / "skill_section_texts.json").read_text()
    )
    code_texts = json.loads((parent / "code" / "skill_section_texts.json").read_text())
    for s in skills:
        assert "Prose" in prose_texts[s.key]
        assert "print" in code_texts[s.key]
        assert "```" not in prose_texts[s.key]

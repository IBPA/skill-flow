"""Build per-section FAISS indices for 3-way structure-aware retrieval.

Each SKILL.md is split into ``(yaml, prose, code)`` by
:func:`skill_flow.corpus.splitter.split_skill_sections`. Each section is
encoded independently (potentially by a different encoder, e.g. a code-aware
encoder for the ``code`` section) and persisted as a sub-index under
``parent_dir/<section>/``. A shared ``skill_descriptions.json`` and
``skill_contents.json`` are persisted at ``parent_dir/`` so downstream
rerankers can populate ``SearchResult.description`` / ``.content``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import faiss
import numpy as np

from skill_flow.corpus.loader import load_content
from skill_flow.corpus.splitter import safe_section_text, split_skill_sections

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from skill_flow.index.encoder import Encoder
    from skill_flow.models import SkillRecord

logger = logging.getLogger(__name__)

SECTIONS: tuple[str, str, str] = ("yaml", "prose", "code")


def _write_subdir(
    section: str,
    skill_keys: list[str],
    texts: list[str],
    encoder: Encoder,
    parent_dir: Path,
    batch_size: int,
) -> None:
    """Encode *texts* and persist a FAISS sub-index under ``parent_dir/section/``."""
    sub_dir = parent_dir / section
    sub_dir.mkdir(parents=True, exist_ok=True)

    safe_texts = [safe_section_text(t) for t in texts]
    logger.info("Encoding %d %s sections …", len(safe_texts), section)
    embeddings = encoder.encode_documents(safe_texts, batch_size=batch_size)

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP for %s (dim=%d) …", section, dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    np.save(sub_dir / "embeddings.npy", embeddings)
    faiss.write_index(index, str(sub_dir / "faiss.index"))
    (sub_dir / "skill_ids.json").write_text(json.dumps(skill_keys), encoding="utf-8")
    section_texts = dict(zip(skill_keys, texts, strict=True))
    (sub_dir / "skill_section_texts.json").write_text(
        json.dumps(section_texts), encoding="utf-8"
    )
    logger.info("Wrote %d vectors to %s", index.ntotal, sub_dir)


def build_section_indices(
    skills: list[SkillRecord],
    parent_dir: Path,
    encoders: Mapping[str, Encoder],
    corpus_path: Path,
    batch_size: int = 256,
    sections: tuple[str, ...] = SECTIONS,
) -> None:
    """Build a structure-aware index of sub-indices, one per section.

    *encoders* maps section name → :class:`Encoder`. Sections not present in
    *encoders* are skipped (useful when ``yaml``/``prose`` indices already
    exist and only the ``code`` section needs (re-)building under a different
    encoder).
    """
    parent_dir.mkdir(parents=True, exist_ok=True)

    skill_keys: list[str] = []
    descriptions: dict[str, str] = {}
    contents: dict[str, str] = {}
    per_section: dict[str, list[str]] = {s: [] for s in sections}

    for skill in skills:
        try:
            content = load_content(corpus_path, skill)
        except FileNotFoundError:
            logger.warning("SKILL.md not found for %s, skipping", skill.key)
            continue
        yaml, prose, code = split_skill_sections(content)
        section_text = {"yaml": yaml, "prose": prose, "code": code}
        skill_keys.append(skill.key)
        descriptions[skill.key] = skill.description
        contents[skill.key] = content
        for s in sections:
            per_section[s].append(section_text[s])

    logger.info("Prepared %d skills for sectioning", len(skill_keys))

    for section in sections:
        encoder = encoders.get(section)
        if encoder is None:
            logger.info("No encoder for section %r, skipping", section)
            continue
        _write_subdir(
            section,
            skill_keys,
            per_section[section],
            encoder,
            parent_dir,
            batch_size,
        )

    (parent_dir / "skill_descriptions.json").write_text(
        json.dumps(descriptions), encoding="utf-8"
    )
    (parent_dir / "skill_contents.json").write_text(
        json.dumps(contents), encoding="utf-8"
    )
    (parent_dir / "skill_ids.json").write_text(json.dumps(skill_keys), encoding="utf-8")
    logger.info(
        "Wrote shared metadata for %d skills to %s", len(skill_keys), parent_dir
    )

"""FAISS index builder — encodes skills and persists the index to disk."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import faiss
import numpy as np

from skill_flow.corpus.loader import load_content

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.index.encoder import Encoder
    from skill_flow.models import SkillRecord

logger = logging.getLogger(__name__)


def build_index(
    skills: list[SkillRecord],
    encoder: Encoder,
    output_dir: Path,
    batch_size: int = 256,
    corpus_path: Path | None = None,
) -> None:
    """Build a FAISS index from skill descriptions and save to disk.

    Persists artifacts in *output_dir*:

    * ``embeddings.npy`` — ``(N, dim)`` float32 matrix
    * ``faiss.index`` — serialized FAISS ``IndexFlatIP``
    * ``skill_ids.json`` — ordered list of skill keys
    * ``skill_descriptions.json`` — key → description mapping
    * ``skill_contents.json`` — key → full SKILL.md content (when *corpus_path* given)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptions = [s.description for s in skills]
    skill_keys = [s.key for s in skills]

    logger.info("Encoding %d skill descriptions …", len(descriptions))
    embeddings = encoder.encode_documents(descriptions, batch_size=batch_size)

    dim = embeddings.shape[1]
    logger.info("Building FAISS IndexFlatIP (dim=%d) …", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info("Saving artifacts to %s …", output_dir)
    np.save(output_dir / "embeddings.npy", embeddings)
    faiss.write_index(index, str(output_dir / "faiss.index"))

    ids_path = output_dir / "skill_ids.json"
    ids_path.write_text(json.dumps(skill_keys), encoding="utf-8")

    desc_map = dict(zip(skill_keys, descriptions, strict=True))
    desc_path = output_dir / "skill_descriptions.json"
    desc_path.write_text(json.dumps(desc_map), encoding="utf-8")

    if corpus_path is not None:
        content_map: dict[str, str] = {}
        for skill in skills:
            try:
                content_map[skill.key] = load_content(corpus_path, skill)
            except FileNotFoundError:
                logger.warning("SKILL.md not found for %s, skipping content", skill.key)
        content_path = output_dir / "skill_contents.json"
        content_path.write_text(json.dumps(content_map), encoding="utf-8")
        logger.info("Saved content for %d skills", len(content_map))

    logger.info("Index built: %d vectors, dim=%d", index.ntotal, dim)

"""Corpus loader for reading skills from the skill-crawler data directory."""

import json
import logging
from pathlib import Path
from typing import Any

from skill_flow.models import SkillRecord

logger = logging.getLogger(__name__)


def load_corpus(corpus_path: Path) -> list[SkillRecord]:
    """Load skill records from the corpus index.

    Reads ``_metadata/index.json`` and converts each entry to a
    :class:`SkillRecord`. Skills with empty descriptions are filtered
    out since they cannot be embedded.
    """
    index_file = corpus_path / "_metadata" / "index.json"
    data = json.loads(index_file.read_text(encoding="utf-8"))

    skills_dict: dict[str, dict[str, Any]] = data["skills"]
    records: list[SkillRecord] = []

    for key, entry in skills_dict.items():
        description = (entry.get("description") or "").strip()
        if not description:
            continue

        # Derive corpus-relative path from the key (e.g. "skillsmp/foo")
        local_path = key

        metadata = {
            k: v
            for k, v in entry.items()
            if k not in {"name", "description", "source", "local_path"}
        }

        records.append(
            SkillRecord(
                key=key,
                name=entry.get("name", ""),
                description=description,
                source=entry.get("source", ""),
                local_path=local_path,
                metadata=metadata,
            )
        )

    logger.info(
        "Loaded %d skills from corpus (filtered empty descriptions)",
        len(records),
    )
    return records


def load_content(corpus_path: Path, record: SkillRecord) -> str:
    """Load the full SKILL.md content for a given skill record."""
    skill_file = corpus_path / record.local_path / "SKILL.md"
    return skill_file.read_text(encoding="utf-8", errors="replace")

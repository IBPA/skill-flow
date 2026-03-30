"""Ground-truth detection and loading for eval-mode pipeline runs."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from skill_flow.eval.models import InjectedSkill, TaskGroundTruth
from skill_flow.eval.utils.ground_truth import (
    load_content_map,
    load_ground_truth,
)

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.retriever.protocol import Searcher

logger = logging.getLogger(__name__)


class GroundTruthContext(BaseModel):
    """Holds all GT data needed to run eval-mode in the pipeline."""

    tasks: list[TaskGroundTruth]
    injected_skills: list[InjectedSkill]
    skipped: list[tuple[str, str]]
    content_map: dict[str, str]


def has_ground_truth(tasks_dir: Path) -> bool:
    """Check if any task directory contains GT skills."""
    for task_dir in tasks_dir.iterdir():
        if not task_dir.is_dir():
            continue
        skills_dir = task_dir / "environment" / "skills"
        if skills_dir.exists() and any(
            (s / "SKILL.md").exists() for s in skills_dir.iterdir() if s.is_dir()
        ):
            return True
    return False


def load_ground_truth_context(
    tasks_dir: Path,
    index_dir: Path,
    max_query_chars: int = 0,
) -> GroundTruthContext | None:
    """Detect GT skills and build context; returns None if no GT found."""
    if not has_ground_truth(tasks_dir):
        return None

    tasks, injected_skills, skipped = load_ground_truth(
        tasks_dir,
        max_query_chars,
    )
    if not tasks:
        return None

    content_map = load_content_map(index_dir, injected_skills)

    logger.info(
        "GT detected: %d tasks, %d skills to inject, %d skipped",
        len(tasks),
        len(injected_skills),
        len(skipped),
    )
    return GroundTruthContext(
        tasks=tasks,
        injected_skills=injected_skills,
        skipped=skipped,
        content_map=content_map,
    )


def augment_searcher(
    searcher: Searcher,
    injected_skills: list[InjectedSkill],
) -> None:
    """Inject GT skills into the searcher via its protocol."""
    if not injected_skills:
        return
    searcher.augment(
        [s.key for s in injected_skills],
        [s.description for s in injected_skills],
    )
    searcher.add_descriptions({s.key: s.description for s in injected_skills})
    searcher.add_contents({s.key: s.content for s in injected_skills if s.content})
    logger.info("Injected %d GT skills into searcher", len(injected_skills))


def load_content_map_for_gt(
    index_dir: Path,
    injected_skills: list[InjectedSkill],
) -> dict[str, str]:
    """Load skill key→content map from index + injected GT skills."""
    content_path = index_dir / "skill_contents.json"
    contents: dict[str, str] = (
        json.loads(content_path.read_text(encoding="utf-8"))
        if content_path.exists()
        else {}
    )
    for skill in injected_skills:
        contents[skill.key] = skill.content
    return contents

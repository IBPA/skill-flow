"""Load ground-truth skills from SkillsBench tasks for retriever evaluation."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import yaml

from skill_flow.eval.models import InjectedSkill, TaskGroundTruth

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def load_corpus_keys(index_dir: Path) -> set[str]:
    """Load the set of skill keys from a persisted FAISS index."""
    ids_path = index_dir / "skill_ids.json"
    keys: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    return set(keys)


def _parse_frontmatter(content: str) -> tuple[str, str]:
    """Extract name and description from YAML frontmatter.

    Returns (name, description). Falls back to empty strings if missing.
    """
    if not content.startswith("---"):
        return "", ""

    end = content.find("---", 3)
    if end == -1:
        return "", ""

    frontmatter = content[3:end].strip()
    try:
        meta = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return "", ""

    if not isinstance(meta, dict):
        return "", ""

    return str(meta.get("name", "")), str(meta.get("description", ""))


def load_content_map(
    index_dir: Path,
    injected_skills: list[InjectedSkill],
) -> dict[str, str]:
    """Load skill key→content map from the index and injected GT skills."""
    content_path = index_dir / "skill_contents.json"
    contents: dict[str, str] = (
        json.loads(content_path.read_text(encoding="utf-8"))
        if content_path.exists()
        else {}
    )
    for skill in injected_skills:
        contents[skill.key] = skill.content
    return contents


def load_ground_truth(
    tasks_dir: Path,
    max_query_chars: int = 0,
) -> tuple[list[TaskGroundTruth], list[InjectedSkill], list[tuple[str, str]]]:
    """Load ground-truth skills from SkillsBench tasks.

    Every GT skill gets a task-scoped key (``skillsbench/{task_id}/{name}``)
    and is injected into the index, so the retriever is evaluated against
    each task's exact skill content.

    Returns:
        evaluable: Tasks with at least one ground-truth skill.
        injected_skills: All GT skills to inject into the index.
        skipped: (task_id, reason) pairs for tasks without skills.
    """
    evaluable: list[TaskGroundTruth] = []
    injected_skills: list[InjectedSkill] = []
    skipped: list[tuple[str, str]] = []

    task_dirs = sorted(d for d in tasks_dir.iterdir() if d.is_dir())

    for task_dir in task_dirs:
        task_id = task_dir.name

        instruction_file = task_dir / "instruction.md"
        if not instruction_file.exists():
            skipped.append((task_id, "no instruction.md"))
            continue

        query = instruction_file.read_text(encoding="utf-8").strip()
        if max_query_chars > 0:
            query = query[:max_query_chars]

        skills_dir = task_dir / "environment" / "skills"
        if not skills_dir.exists():
            skipped.append((task_id, "no skills directory"))
            continue

        gt_keys: list[str] = []
        all_skill_names: list[str] = []

        for entry in sorted(skills_dir.iterdir()):
            if not entry.is_dir():
                continue

            skill_file = entry / "SKILL.md"
            if not skill_file.exists():
                continue

            skill_name = entry.name
            all_skill_names.append(skill_name)
            key = f"skillsbench/{task_id}/{skill_name}"
            gt_keys.append(key)

            raw_content = skill_file.read_text(encoding="utf-8")
            _, description = _parse_frontmatter(raw_content)
            if not description:
                description = skill_name

            injected_skills.append(
                InjectedSkill(
                    key=key,
                    name=skill_name,
                    description=description,
                    content=raw_content,
                )
            )

        if not gt_keys:
            skipped.append((task_id, "no skills with SKILL.md"))
            continue

        evaluable.append(
            TaskGroundTruth(
                task_id=task_id,
                query=query,
                ground_truth_keys=gt_keys,
                all_skill_names=all_skill_names,
                injected_skills=gt_keys,
            )
        )

    logger.info(
        "Ground truth: %d evaluable tasks, %d skills to inject, %d skipped",
        len(evaluable),
        len(injected_skills),
        len(skipped),
    )

    return evaluable, injected_skills, skipped

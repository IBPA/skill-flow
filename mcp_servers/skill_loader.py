"""Load SKILL.md files from SkillsBench tasks, indexed by task name.

Used by skillsbench_mcp_server.py to serve real skills via MCP.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class LoadedSkill:
    """A loaded skill with parsed metadata."""

    name: str
    description: str
    content: str


@dataclass
class ResolvedSkill:
    """A skill resolved from eval results to a folder path on disk."""

    name: str
    description: str
    folder_path: Path


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


def _load_skill_from_dir(skill_dir: Path) -> LoadedSkill | None:
    """Load a skill from a directory containing SKILL.md."""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        return None

    content = skill_file.read_text()
    name, description = _parse_frontmatter(content)
    if not name:
        name = skill_dir.name

    return LoadedSkill(name=name, description=description, content=content)


def _load_skill_from_file(skill_file: Path) -> LoadedSkill | None:
    """Load a skill from a standalone .md file (e.g. reference.md)."""
    if not skill_file.exists():
        return None

    content = skill_file.read_text()
    name, description = _parse_frontmatter(content)
    if not name:
        name = skill_file.stem

    return LoadedSkill(name=name, description=description, content=content)


def load_task_skills(
    tasks_dir: Path, task_names: list[str] | None = None
) -> dict[str, list[LoadedSkill]]:
    """Load skills for each task, indexed by task name.

    Args:
        tasks_dir: Root directory containing task subdirectories.
        task_names: Optional list of task names to load. If None, loads all.

    Returns:
        Mapping of task name to list of loaded skills.
    """
    result: dict[str, list[LoadedSkill]] = {}

    if task_names:
        task_dirs = [tasks_dir / name for name in task_names]
    else:
        task_dirs = sorted(d for d in tasks_dir.iterdir() if d.is_dir())

    for task_dir in task_dirs:
        skills_path = task_dir / "environment" / "skills"
        if not skills_path.exists():
            continue

        task_name = task_dir.name
        skills: list[LoadedSkill] = []

        for entry in sorted(skills_path.iterdir()):
            if entry.is_dir():
                skill = _load_skill_from_dir(entry)
                if skill:
                    skills.append(skill)
            elif entry.suffix == ".md":
                skill = _load_skill_from_file(entry)
                if skill:
                    skills.append(skill)

        if skills:
            result[task_name] = skills
            logger.info(
                "Loaded %d skills for task %s: %s",
                len(skills),
                task_name,
                [s.name for s in skills],
            )

    return result


def _resolve_skill_folder(
    key: str,
    tasks_dir: Path,
    corpus_dir: Path | None,
) -> tuple[str, Path] | None:
    """Resolve a single skill key to (name, folder_path).

    Supported key formats:
    - ``skillsbench/{source_task}/{skill_name}`` (3 segments)
    - ``skillsmp/{skill_name}`` (2 segments, requires corpus_dir)

    Returns None if the key format is unrecognised or the folder is missing.
    """
    parts = key.split("/")

    if len(parts) == 3 and parts[0] == "skillsbench":
        source_task, skill_name = parts[1], parts[2]
        folder = tasks_dir / source_task / "environment" / "skills" / skill_name
    elif len(parts) == 2 and parts[0] == "skillsmp":
        if corpus_dir is None:
            logger.warning("No corpus_dir provided, cannot resolve: %s", key)
            return None
        skill_name = parts[1]
        folder = corpus_dir / "skillsmp" / skill_name
    else:
        logger.warning("Unexpected skill key format: %s", key)
        return None

    if not folder.is_dir():
        logger.warning("Skill folder not found on disk: %s", folder)
        return None

    return skill_name, folder


def resolve_eval_skill_folders(
    eval_results_path: Path,
    tasks_dir: Path,
    task_name: str,
    corpus_dir: Path | None = None,
) -> list[ResolvedSkill]:
    """Resolve eval results to skill folder paths on disk.

    Reads the eval results JSON, finds the task_results entry matching
    task_name, parses each retrieved skill's key to locate the folder.

    Supported key formats:
    - ``skillsbench/{source_task}/{skill_name}``
      → ``{tasks_dir}/{source_task}/environment/skills/{skill_name}/``
    - ``skillsmp/{skill_name}``
      → ``{corpus_dir}/skillsmp/{skill_name}/``

    Args:
        eval_results_path: Path to the eval results JSON file.
        tasks_dir: Root directory containing SkillsBench task directories.
        task_name: Task name to resolve skills for.
        corpus_dir: Root directory of the skills corpus (for skillsmp keys).

    Returns:
        List of resolved skills with folder paths.
    """
    with eval_results_path.open() as f:
        data = json.load(f)

    task_results: list[dict[str, object]] = data.get("task_results", [])
    entry = next(
        (t for t in task_results if t.get("task_id") == task_name),
        None,
    )
    if entry is None:
        logger.warning("Task '%s' not found in eval results", task_name)
        return []

    resolved: list[ResolvedSkill] = []
    retrieved: list[dict[str, object]] = entry.get("retrieved_skills", [])  # type: ignore[assignment]

    for skill in retrieved:
        key = str(skill.get("key", ""))
        result = _resolve_skill_folder(key, tasks_dir, corpus_dir)
        if result is None:
            continue

        skill_name, folder = result
        resolved.append(
            ResolvedSkill(
                name=skill_name,
                description=str(skill.get("description", "")),
                folder_path=folder,
            )
        )

    logger.info(
        "Resolved %d/%d skills for task '%s': %s",
        len(resolved),
        len(retrieved),
        task_name,
        [s.name for s in resolved],
    )
    return resolved

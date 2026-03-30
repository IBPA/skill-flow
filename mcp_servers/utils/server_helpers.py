"""Shared helpers for the SkillFlow MCP server."""

from __future__ import annotations

import io
import json
import logging
import tarfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.retriever.retriever import SearchResult

logger = logging.getLogger(__name__)

CONTAINER_SKILLS_DIR = "/logs/agent/skills"

RETRIEVE_SKILL_DOC = """\
Search for an expert skill before attempting any task.

IMPORTANT: You MUST call this tool at the start of every task,
before writing any code or running any commands. Expert skills
contain critical guidance, shortcuts, and pitfall warnings that
dramatically improve success rates.

Even if a task seems straightforward, call this tool — there may
be non-obvious tricks or common failure modes you'd otherwise
miss.

This tool returns shell commands. You MUST execute every command
returned by this tool, then read the SKILL.md files and follow
their guidance before starting the task.

Args:
    query: Describe what you're trying to accomplish. Be specific
        about the domain, technology, and goal.
"""


def skill_name(key: str) -> str:
    """Extract the short skill name from a key (e.g. 'skillsmp/foo' -> 'foo')."""
    return key.rsplit("/", maxsplit=1)[-1]


def resolve_skill_folder(
    key: str,
    corpus_dir: Path,
    tasks_dir: Path | None,
) -> Path | None:
    """Resolve a skill key to its on-disk folder path.

    - ``skillsmp/X``  → ``corpus_dir/skillsmp/X``
    - ``skillsbench/task/skill`` → ``tasks_dir/task/environment/skills/skill``
    """
    if tasks_dir and key.startswith("skillsbench/"):
        parts = key.split("/")
        if len(parts) == 3:
            folder = tasks_dir / parts[1] / "environment" / "skills" / parts[2]
            if folder.is_dir():
                return folder
    folder = corpus_dir / key
    return folder if folder.is_dir() else None


def create_tar_gz(folder_path: Path) -> bytes:
    """Create a tar.gz archive of a skill folder's contents."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for child in sorted(folder_path.iterdir()):
            tar.add(str(child), arcname=child.name)
    return buf.getvalue()


def format_results(
    results: list[SearchResult],
    base_url: str,
    corpus_dir: Path,
    tasks_dir: Path | None = None,
) -> str:
    """Format pipeline results as download commands for the agent."""
    if not results:
        return "No matching skills found for this query."

    valid: list[tuple[SearchResult, str]] = []
    for r in results:
        name = skill_name(r.key)
        folder = resolve_skill_folder(r.key, corpus_dir, tasks_dir)
        if folder:
            valid.append((r, name))
        else:
            logger.warning("Skill folder not found: %s", r.key)

    if not valid:
        return "No matching skills found for this query."

    lines = [f"Found {len(valid)} skills. Download them by running:", ""]
    paths: list[str] = []
    for r, name in valid:
        path = f"{CONTAINER_SKILLS_DIR}/{name}"
        url = f"{base_url}/download/{r.key}"
        lines.append(f'mkdir -p {path} && curl -sL "{url}" | tar xz -C {path}')
        paths.append(path)

    lines.append("")
    lines.append("Use skills at:")
    for path in paths:
        lines.append(f"- {path}")

    return "\n".join(lines)


def log_query(
    query: str,
    results: list[SearchResult],
    latency_ms: float,
    log_file: Path,
) -> None:
    """Append a JSONL log entry for a retrieval query."""
    entry = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "query": query,
        "retrieved_skills": [
            {"key": r.key, "score": round(r.score, 4)} for r in results
        ],
        "n_results": len(results),
        "latency_ms": round(latency_ms, 1),
    }
    with log_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")

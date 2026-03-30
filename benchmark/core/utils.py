"""Shared utilities for evaluation scripts."""

from __future__ import annotations

import contextlib
import os
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

from benchmark.core.config import EvalMode
from benchmark.core.paths import get_project_root

if TYPE_CHECKING:
    from pathlib import Path

    from benchmark.core.config import EvalConfig

__all__ = [
    "_get_benchmark_prefix",
    "_get_short_model_name",
    "generate_job_name",
    "get_project_root",
    "load_tasks_from_file",
]


def _get_benchmark_prefix(config: EvalConfig) -> str:
    """Get short benchmark prefix for job names.

    Args:
        config: Evaluation configuration

    Returns:
        Short prefix string (e.g. 'tb' for terminal-bench, 'sb' for skillbench)
    """
    if config.tasks_path:
        return config.tasks_path.parent.name[:2]
    if config.dataset:
        # e.g. "terminal-bench@2.0" -> "tb", "skillbench@1.0" -> "sk"
        name = config.dataset.split("@")[0]
        parts = name.split("-")
        if len(parts) >= 2:
            return parts[0][0] + parts[1][0]
        return name[:2]
    return "ev"


def _get_short_model_name(model: str) -> str:
    """Extract a short model identifier for job names.

    Examples:
        "openai/gpt-5-nano-2025-08-07" -> "gpt5nano"
        "anthropic/claude-sonnet-4-5" -> "sonnet45"
        "openai/codex-mini" -> "codexmini"

    Args:
        model: Full model identifier string

    Returns:
        Short model name
    """
    # Strip provider prefix (e.g. "openai/")
    name = model.split("/")[-1]
    # Remove date suffix (e.g. "-2025-08-07")
    # Match pattern: -YYYY-MM-DD at the end
    parts = name.split("-")
    cleaned: list[str] = []
    i = 0
    while i < len(parts):
        # Skip date segments (4-digit year followed by 2-digit month and day)
        if len(parts[i]) == 4 and parts[i].isdigit() and i + 2 < len(parts):
            break
        cleaned.append(parts[i])
        i += 1
    return "".join(cleaned)


def generate_job_name(config: EvalConfig) -> str:
    """Generate a job name based on configuration.

    Format: {benchmark}-{mode}-{model}-{effort}-{timestamp}[-{task_suffix}]

    Args:
        config: Evaluation configuration

    Returns:
        Generated job name string
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bp = _get_benchmark_prefix(config)

    if config.mode == EvalMode.BASELINE:
        prefix = f"{bp}-baseline"
    elif config.mode == EvalMode.MCP:
        prefix = f"{bp}-mcp"
    elif config.mode == EvalMode.SKILLFLOW_INJECTION:
        prefix = f"{bp}-skillflow-injection"
    elif config.mode == EvalMode.SKILLFLOW_CACHED:
        prefix = f"{bp}-skillflow-cached"
    else:  # SKILLS mode
        prefix = _generate_skills_job_prefix(config)

    # Append model and reasoning effort
    short_model = _get_short_model_name(config.model)
    prefix = f"{prefix}-{short_model}"
    if config.reasoning_effort:
        prefix = f"{prefix}-{config.reasoning_effort}"

    job_name = f"{prefix}-{timestamp}"

    # Append task suffix
    all_tasks = config.tasks.get_all_tasks()
    if len(all_tasks) == 1:
        job_name = f"{job_name}-{all_tasks[0]}"
    elif len(all_tasks) > 1:
        job_name = f"{job_name}-{len(all_tasks)}tasks"

    return job_name


def _generate_skills_job_prefix(config: EvalConfig) -> str:
    """Generate job name prefix for skills mode.

    Args:
        config: Evaluation configuration

    Returns:
        Job name prefix
    """
    bp = _get_benchmark_prefix(config)
    # Use skill list name or skills dir basename
    assert config.skills
    skill_name_part = config.skills.skills_dir.name

    parts = [f"{bp}-{skill_name_part}"]

    if config.skills.match_skill_to_task:
        parts.append("matched")

    return "-".join(parts)


def load_tasks_from_file(file_path: Path) -> list[str]:
    """Load task names from a file.

    Args:
        file_path: Path to file containing task names

    Returns:
        List of task names
    """
    tasks: list[str] = []

    with file_path.open() as f:
        for raw_line in f:
            # Remove comments and strip whitespace
            line = raw_line.split("#")[0].strip()
            if line:
                tasks.append(line)

    return tasks


def fix_docker_file_ownership(directory: Path) -> None:
    """Fix ownership of Docker-created files.

    Docker containers create files as root, which prevents deletion
    during retry operations. This runs a Docker container to chown
    files to the current user.

    Args:
        directory: Directory to fix ownership in
    """
    if not directory.exists():
        return

    uid = os.getuid()
    gid = os.getgid()

    with contextlib.suppress(FileNotFoundError):
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{directory}:/cleanup",
                "alpine",
                "chown",
                "-R",
                f"{uid}:{gid}",
                "/cleanup",
            ],
            capture_output=True,
            check=False,
        )


def count_skills(skills_dir: Path, skills_list_file: Path | None = None) -> int:
    """Count available skills.

    Args:
        skills_dir: Directory containing skills
        skills_list_file: Optional file listing specific skills to use

    Returns:
        Number of skills
    """
    if skills_list_file and skills_list_file.exists():
        tasks = load_tasks_from_file(skills_list_file)
        return len(tasks)

    # Count directories containing SKILL.md
    return sum(1 for _ in skills_dir.rglob("SKILL.md"))

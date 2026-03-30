"""Download complete skill folders via ``npx skills add``."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

NPX_TIMEOUT_SECONDS = 120


def download_skill(
    source: str,
    skill_id: str,
    task_name: str,
    output_dir: Path,
    *,
    timeout: int = NPX_TIMEOUT_SECONDS,
) -> bool:
    """Download a single skill and place it at ``output_dir/task_name/``.

    Uses ``npx skills add`` to fetch the complete skill folder (SKILL.md,
    scripts/, references/, assets/).  Returns *True* on success.
    """
    dest = output_dir / task_name
    if (dest / "SKILL.md").exists():
        logger.info("Skipping %s: already downloaded", task_name)
        return True

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cmd = [
            "npx",
            "skills",
            "add",
            source,
            "-s",
            skill_id,
            "-y",
            "--copy",
            "-a",
            "claude-code",
        ]
        logger.info("Running: %s (cwd=%s)", " ".join(cmd), tmp_path)
        try:
            subprocess.run(
                cmd,
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning("npx skills add failed for %s/%s: %s", source, skill_id, exc)
            return False

        # npx skills add places files at {cwd}/.claude/skills/{skillId}/
        expected_dir = tmp_path / ".claude" / "skills" / skill_id
        skill_dir = expected_dir if expected_dir.is_dir() else _find_skill_dir(tmp_path)
        if skill_dir is None:
            logger.warning(
                "No skill folder found after download for %s/%s",
                source,
                skill_id,
            )
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(skill_dir, dest)

    logger.info("Downloaded %s/%s -> %s", source, skill_id, dest)
    return True


def download_skills_batch(
    skills_by_source: dict[str, list[tuple[str, str]]],
    output_dir: Path,
    *,
    timeout: int = NPX_TIMEOUT_SECONDS,
) -> dict[str, bool]:
    """Download multiple skills, grouped by source repo.

    *skills_by_source* maps ``source`` to a list of ``(skill_id, task_name)``
    tuples.  Returns a dict mapping ``task_name`` to success status.
    """
    results: dict[str, bool] = {}
    for source, items in skills_by_source.items():
        for skill_id, task_name in items:
            ok = download_skill(
                source, skill_id, task_name, output_dir, timeout=timeout
            )
            results[task_name] = ok
    return results


def _find_skill_dir(root: Path) -> Path | None:
    """Walk *root* looking for a directory that contains SKILL.md."""
    for path in root.rglob("SKILL.md"):
        return path.parent
    return None

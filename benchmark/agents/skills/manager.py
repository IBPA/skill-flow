"""
Skill discovery and filtering logic for evaluation agents.
"""

import logging
from pathlib import Path

from benchmark.agents.base import TASK_NAME_SEPARATOR


def extract_task_name_from_trial_dir(trial_dir_name: str) -> str | None:
    """
    Extract task name from trial directory name.

    Trial directory names follow the pattern: "task-name__hash"

    Args:
        trial_dir_name: Name of the trial directory.

    Returns:
        Task name extracted from the directory name, or None if not found.
    """
    if TASK_NAME_SEPARATOR in trial_dir_name:
        return str(trial_dir_name.rsplit(TASK_NAME_SEPARATOR, 1)[0])
    return None


class SkillManager:
    """
    Manages skill discovery and filtering for evaluation agents.

    Supports:
    - Recursive skill folder discovery (finds SKILL.md files)
    - Filtering by skill list file
    - Task-to-skill matching
    """

    def __init__(
        self,
        source_dir: Path,
        skills_list_file: Path | None = None,
        match_skill_to_task: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize skill manager.

        Args:
            source_dir: Directory containing skill folders.
            skills_list_file: Optional file listing skill names to include.
            match_skill_to_task: If True, filter to skill matching task name.
            logger: Optional logger instance.
        """
        self._source_dir = source_dir
        self._skills_list_file = skills_list_file
        self._match_skill_to_task = match_skill_to_task
        self._logger = logger or logging.getLogger(__name__)

    def find_skill_folders(self, base_dir: Path | None = None) -> list[Path]:
        """
        Recursively find all skill folders (directories containing SKILL.md).

        Supports nested structures like:
        - base_dir/skill-name/SKILL.md (flat)
        - base_dir/repo/skill-name/SKILL.md (nested by repo)

        Args:
            base_dir: Directory to search. Defaults to source_dir.

        Returns:
            List of paths to skill folders.
        """
        if base_dir is None:
            base_dir = self._source_dir

        skill_folders = []
        for item in base_dir.iterdir():
            if item.name.startswith(".") or item.name == "__pycache__":
                continue
            if item.is_dir():
                if (item / "SKILL.md").exists():
                    skill_folders.append(item)
                else:
                    skill_folders.extend(self.find_skill_folders(item))
        return skill_folders

    def load_skill_names_from_file(self) -> set[str]:
        """
        Load skill names from the skills list file.

        Returns:
            Set of skill names to include.
        """
        if not self._skills_list_file or not self._skills_list_file.exists():
            return set()

        skill_names: set[str] = set()
        with self._skills_list_file.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                skill_names.add(line)

        return skill_names

    def filter_skills(
        self, skill_folders: list[Path], task_name: str | None = None
    ) -> list[Path]:
        """
        Filter skill folders based on configuration.

        Args:
            skill_folders: List of all discovered skill folders.
            task_name: Current task name for task-to-skill matching.

        Returns:
            Filtered list of skill folders to use.
        """
        filtered = skill_folders

        # Filter by skills list file if provided
        if self._skills_list_file:
            skill_names = self.load_skill_names_from_file()
            if skill_names:
                self._logger.debug(f"Using skill list with {len(skill_names)} skills")
                filtered = [f for f in filtered if f.name in skill_names]

        # Filter to matching skill if task matching is enabled
        if self._match_skill_to_task:
            if task_name:
                matching = [f for f in filtered if f.name == task_name]
                if matching:
                    self._logger.debug(
                        f"Task-skill match found: {task_name} -> {matching[0].name}"
                    )
                    return matching
                self._logger.debug(
                    f"No matching skill found for task '{task_name}', "
                    "skipping skill injection"
                )
                return []
            self._logger.warning(
                "Could not extract task name, skipping skill injection"
            )
            return []

        return filtered

    def get_skills(self, task_name: str | None = None) -> list[Path]:
        """
        Discover and filter skills based on configuration.

        Args:
            task_name: Current task name for task-to-skill matching.

        Returns:
            List of skill folder paths to inject.
        """
        if not self._source_dir.exists():
            self._logger.debug(f"Skills source not found: {self._source_dir}")
            return []

        all_folders = self.find_skill_folders()
        if not all_folders:
            self._logger.warning(
                f"No skill folders (with SKILL.md) found in {self._source_dir}"
            )
            return []

        return self.filter_skills(all_folders, task_name)

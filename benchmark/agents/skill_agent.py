"""
Codex agent with skill injection for evaluation.
"""

from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment

from benchmark.agents.base import BaseCodexAgent, get_project_root
from benchmark.agents.skills import SkillManager, TarGzSkillInjector
from benchmark.agents.skills.manager import extract_task_name_from_trial_dir


class SkillAgent(BaseCodexAgent):
    """
    Codex agent that injects skills into the evaluation environment.

    Loads skills from a directory and uploads them to the container before
    running the agent. Optionally applies instruction templates.
    """

    def __init__(
        self,
        *args: Any,
        skills_source_dir: str | None = None,
        skills_list_file: str | None = None,
        match_skill_to_task: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize SkillAgent.

        Args:
            skills_source_dir: Path to directory containing custom skill folders.
                If not provided, uses outputs/skills/downloaded/.
            skills_list_file: Path to a text file containing skill names to load
                (one per line). If provided, only skills in this list will be loaded.
            match_skill_to_task: If True, only load the skill whose name matches the
                current task name (extracted from trial directory). Enables 1:1
                task-to-skill mapping for targeted skill injection experiments.
        """
        super().__init__(*args, **kwargs)

        if skills_source_dir:
            source_dir = Path(skills_source_dir)
        else:
            source_dir = get_project_root() / "outputs" / "skills" / "downloaded"

        list_file = Path(skills_list_file) if skills_list_file else None

        self._skill_manager = SkillManager(
            source_dir=source_dir,
            skills_list_file=list_file,
            match_skill_to_task=match_skill_to_task,
            logger=self.logger,
        )
        self._skill_injector = TarGzSkillInjector(logger=self.logger)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent and inject custom skills."""
        await super().setup(environment)

        task_name = self._extract_task_name()
        skill_folders = self._skill_manager.get_skills(task_name)
        await self._skill_injector.inject(environment, skill_folders, self.logs_dir)

    def _extract_task_name(self) -> str | None:
        """Extract task name from logs_dir path."""
        trial_dir = self.logs_dir.parent.name
        return extract_task_name_from_trial_dir(trial_dir)

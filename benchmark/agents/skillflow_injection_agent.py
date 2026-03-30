"""Codex agent that injects skills into evaluation containers.

Supports three mutually exclusive skill sources:
- skills_dir: Scan a local directory tree for SKILL.md files (golden injection)
- eval_results: SkillFlow eval results JSON (from ``skill_flow.cli eval``)
- selector_cache: Selector cache JSON (from selector stage)

If no source is provided, no skills are injected (baseline mode).
"""

import json
import logging
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from mcp_servers.utils.skill_loader import (
    _resolve_skill_folder,
    resolve_eval_skill_folders,
)

from benchmark.agents.base import BaseCodexAgent
from benchmark.agents.skills import SkillManager, TarGzSkillInjector
from benchmark.agents.skills.manager import extract_task_name_from_trial_dir

logger = logging.getLogger(__name__)


class SkillFlowInjectionAgent(BaseCodexAgent):
    """Injects skills into evaluation containers from various sources.

    Three mutually exclusive sources:
    - ``skills_dir``: scan directory for SKILL.md folders (golden injection)
    - ``eval_results``: SkillFlow eval results JSON
    - ``selector_cache``: selector cache JSON

    If none is provided, no skills are injected (baseline).
    """

    def __init__(
        self,
        *args: Any,
        skills_dir: str | None = None,
        skills_list_file: str | None = None,
        match_skill_to_task: bool = False,
        eval_results: str | None = None,
        selector_cache: str | None = None,
        tasks_dir: str | None = None,
        corpus_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        sources = sum(1 for s in (skills_dir, eval_results, selector_cache) if s)
        if sources > 1:
            msg = "At most one of skills_dir, eval_results, or selector_cache"
            raise ValueError(msg)

        self._eval_results = Path(eval_results) if eval_results else None
        self._selector_cache = Path(selector_cache) if selector_cache else None
        self._tasks_dir = Path(tasks_dir) if tasks_dir else None
        self._corpus_dir = Path(corpus_dir) if corpus_dir else None

        if skills_dir:
            list_file = Path(skills_list_file) if skills_list_file else None
            self._skill_manager: SkillManager | None = SkillManager(
                source_dir=Path(skills_dir),
                skills_list_file=list_file,
                match_skill_to_task=match_skill_to_task,
                logger=self.logger,
            )
        else:
            self._skill_manager = None

        self._injector = TarGzSkillInjector(logger=self.logger)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Resolve skills from configured source and inject into container."""
        await super().setup(environment)

        task_name = self._extract_task_name()

        if self._skill_manager:
            folders = self._skill_manager.get_skills(task_name)
        elif self._eval_results or self._selector_cache:
            if not task_name:
                self.logger.warning("Could not extract task name, skipping injection")
                return
            folders = self._resolve_from_json(task_name)
            if not folders:
                self.logger.warning("No skills resolved for task '%s'", task_name)
                return
        else:
            return  # No source configured (baseline)

        if not folders:
            return

        n_injected = await self._injector.inject(environment, folders, self.logs_dir)
        self.logger.info("Injected %d skills for task '%s'", n_injected, task_name)

    def _extract_task_name(self) -> str | None:
        """Extract task name from logs_dir path."""
        trial_dir = self.logs_dir.parent.name
        return extract_task_name_from_trial_dir(trial_dir)

    def _resolve_from_json(self, task_name: str) -> list[Path]:
        """Resolve skill folders from eval results or selector cache."""
        if self._eval_results:
            return self._resolve_from_eval_results(task_name)
        return self._resolve_from_selector_cache(task_name)

    def _resolve_from_eval_results(self, task_name: str) -> list[Path]:
        """Resolve skill folders from eval results JSON."""
        assert self._eval_results
        resolved = resolve_eval_skill_folders(
            self._eval_results,
            self._tasks_dir or Path(),
            task_name,
            self._corpus_dir,
        )
        return [s.folder_path for s in resolved]

    def _resolve_from_selector_cache(self, task_name: str) -> list[Path]:
        """Resolve skill folders from selector cache JSON."""
        assert self._selector_cache
        cache: dict[str, list[str]] = json.loads(
            self._selector_cache.read_text(encoding="utf-8")
        )
        keys = cache.get(task_name, [])
        if not keys:
            logger.info("No cached keys for task '%s'", task_name)
            return []

        folders: list[Path] = []
        for key in keys:
            result = _resolve_skill_folder(
                key, self._tasks_dir or Path(), self._corpus_dir
            )
            if result is not None:
                folders.append(result[1])
        return folders

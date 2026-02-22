"""Codex agent that injects SkillFlow-retrieved skills for evaluation.

Combines skill folder injection (via TarGzSkillInjector) with MCP server
registration so the agent can discover injected skill paths at runtime.
"""

from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment
from mcp_servers.skill_loader import resolve_eval_skill_folders

from benchmark.agents.base import BaseCodexAgent, McpServer
from benchmark.agents.skills.injector import TarGzSkillInjector
from benchmark.agents.skills.manager import extract_task_name_from_trial_dir


class SkillFlowEvalAgent(BaseCodexAgent):
    """Injects SkillFlow-retrieved skills and registers MCP for path discovery.

    Two-phase setup:
    1. Resolves eval results to skill folders and injects them into the
       container via ``TarGzSkillInjector``.
    2. Registers an MCP server so the agent can call ``retrieve_skill()``
       to discover where the injected skills live.
    """

    DEFAULT_MCP_URL = "http://host.docker.internal:8765/mcp"

    def __init__(
        self,
        *args: Any,
        eval_results: str,
        tasks_dir: str,
        mcp_url: str | None = None,
        corpus_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        mcp_servers = [McpServer(name="skillflow", url=mcp_url)] if mcp_url else []
        super().__init__(*args, mcp_servers=mcp_servers, **kwargs)
        self._eval_results = Path(eval_results)
        self._tasks_dir = Path(tasks_dir)
        self._corpus_dir = Path(corpus_dir) if corpus_dir else None
        self._injector = TarGzSkillInjector(logger=self.logger)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Resolve skills from eval results and inject into container."""
        await super().setup(environment)

        task_name = self._extract_task_name()
        if not task_name:
            self.logger.warning("Could not extract task name, skipping skill injection")
            return

        resolved = resolve_eval_skill_folders(
            self._eval_results, self._tasks_dir, task_name, self._corpus_dir
        )
        if not resolved:
            self.logger.warning("No skills resolved for task '%s'", task_name)
            return

        folders = [s.folder_path for s in resolved]
        n_injected = await self._injector.inject(environment, folders, self.logs_dir)
        self.logger.info(
            "Injected %d skills for task '%s': %s",
            n_injected,
            task_name,
            [s.name for s in resolved],
        )

    def _extract_task_name(self) -> str | None:
        """Extract task name from logs_dir path."""
        trial_dir = self.logs_dir.parent.name
        return extract_task_name_from_trial_dir(trial_dir)

"""Codex agent for cached SkillFlow MCP experiments.

Extends SkillFlowMCPAgent to call the server's ``/set-task`` endpoint before
each task, sending the pre-computed skill keys so the cached server knows
which skills to return when the LLM calls ``retrieve_skill``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmark.agents.skillflow_mcp_agent import SkillFlowMCPAgent
from benchmark.agents.skills.manager import extract_task_name_from_trial_dir

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


def _derive_base_url(mcp_url: str) -> str:
    """Strip the ``/mcp`` suffix from an MCP URL to get the server base URL."""
    return mcp_url.removesuffix("/mcp")


class SkillFlowMCPCachedAgent(SkillFlowMCPAgent):
    """SkillFlowMCPAgent that sends cached skill keys to the server per task.

    Reads the selector cache JSON once at init, then before each task
    POSTs the task's skill keys to the server's ``/set-task`` endpoint.
    """

    def __init__(
        self,
        *args: Any,
        mcp_url: str | None = None,
        selector_cache: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, mcp_url=mcp_url, **kwargs)
        effective_url = mcp_url or self.DEFAULT_MCP_URL
        self._server_base_url = _derive_base_url(effective_url)
        self._cache = self._load_cache(selector_cache)

    @staticmethod
    def _load_cache(path: str) -> dict[str, list[str]]:
        cache_path = Path(path)
        if not cache_path.exists():
            logger.warning("Selector cache not found: %s", path)
            return {}
        data: dict[str, list[str]] = json.loads(cache_path.read_text(encoding="utf-8"))
        logger.info("Loaded selector cache: %d tasks from %s", len(data), path)
        return data

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)

        trial_dir = self.logs_dir.parent.name
        task_name = extract_task_name_from_trial_dir(trial_dir)
        if not task_name:
            logger.warning("Could not extract task name from %s", trial_dir)
            return

        skill_keys = self._cache.get(task_name, [])
        self._notify_server(task_name, skill_keys)

    def _notify_server(self, task_id: str, skill_keys: list[str]) -> None:
        """POST the task_id and skill keys to the server's /set-task endpoint."""
        url = f"{self._server_base_url}/set-task"
        if not url.startswith(("http://", "https://")):
            logger.error("Refusing non-HTTP URL: %s", url)
            return
        payload = json.dumps({"task_id": task_id, "skill_keys": skill_keys}).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                logger.info(
                    "Set task=%s (%d keys) on server (status=%d)",
                    task_id,
                    len(skill_keys),
                    resp.status,
                )
        except urllib.error.URLError:
            logger.exception("Failed to set task on server: %s", url)

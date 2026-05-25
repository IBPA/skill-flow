"""Skill-injecting evaluation agents.

The CLI-agnostic resolution/injection logic lives in :class:`SkillInjectionMixin`,
which is bound to each supported agent backend:

- :class:`SkillFlowCodexAgent` — Codex CLI
- :class:`SkillFlowGeminiAgent` — Gemini CLI

Both resolve skills from one of three mutually exclusive sources:

- ``skills_dir``: scan a local directory tree for SKILL.md files (golden injection)
- ``eval_results``: SkillFlow eval results JSON (from ``skill_flow.cli eval``)
- ``selector_cache``: selector cache JSON (from the selector stage)

If no source is provided, no skills are injected (baseline mode).
"""

import json
import logging
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from mcp_servers.utils.skill_loader import (
    _resolve_skill_folder,
    resolve_eval_skill_folders,
)

from benchmark.agents.codex_adapter import AdaptedCodex
from benchmark.agents.skills import SkillManager, TarGzSkillInjector
from benchmark.agents.skills.manager import extract_task_name_from_trial_dir

logger = logging.getLogger(__name__)


class SkillInjectionMixin(BaseAgent):
    """Resolves skills from a configured source and injects them at setup.

    Must be combined with a concrete Harbor agent base class (Codex or
    GeminiCli) that provides the CLI ``setup``/``run`` implementation. The
    mixin's ``setup`` chains to the cooperating base via ``super().setup``.

    Three mutually exclusive sources:
    - ``skills_dir``: scan directory for SKILL.md folders (golden/oracle/Vercel)
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


class SkillFlowCodexAgent(SkillInjectionMixin, AdaptedCodex):
    """Codex CLI agent that injects skills resolved from a configured source.

    See :class:`SkillInjectionMixin` for the supported skill sources.
    """


class SkillFlowGeminiAgent(SkillInjectionMixin, GeminiCli):
    """Gemini CLI agent that injects skills resolved from a configured source.

    Wraps Harbor's :class:`~harbor.agents.installed.gemini_cli.GeminiCli`
    (which runs Google's ``gemini`` CLI, authenticates via ``GEMINI_API_KEY``,
    and parses Gemini trajectories). The model ID is supplied via Harbor's
    ``--model`` flag and must be in ``provider/model`` form, e.g.
    ``google/gemini-3.1-flash-lite``. Unlike the Codex backend, the Gemini CLI has
    no ``reasoning_effort`` setting, so callers should not pass one.
    """

    # Gemini's experimental skills feature only surfaces skills (via the
    # ``activate_skill`` tool) from its native skills directory.
    GEMINI_SKILLS_DIR = "~/.gemini/skills"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Inject skills, then mirror them into Gemini's native skills dir.

        The shared injector stages skills under ``/logs/agent/skills`` (where
        the Codex prompt template points the agent). The Gemini CLI, however,
        only discovers skills from ``~/.gemini/skills``, so copy them there to
        make injected skills usable via ``activate_skill`` -- matching how
        SkillsBench's bundled skills are exposed. A no-op when nothing was
        injected (e.g. baseline/bundled conditions).
        """
        await super().setup(environment)
        src = TarGzSkillInjector.CONTAINER_SKILLS_DIR
        await environment.exec(
            command=(
                f"if [ -d {src} ]; then mkdir -p {self.GEMINI_SKILLS_DIR} && "
                f"cp -r {src}/. {self.GEMINI_SKILLS_DIR}/; fi"
            )
        )

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """Run the Gemini CLI, trusting the (untrusted) container workspace.

        In a fresh container the working directory is not a "trusted folder",
        so the CLI silently downgrades ``-y`` (auto-approve) to manual approval
        and refuses to run any tools in headless mode. Setting
        ``GEMINI_CLI_TRUST_WORKSPACE=true`` restores autonomous execution.
        """
        commands = super().create_run_agent_commands(instruction)
        updated: list[ExecInput] = []
        for cmd in commands:
            env = {**(cmd.env or {}), "GEMINI_CLI_TRUST_WORKSPACE": "true"}
            updated.append(cmd.model_copy(update={"env": env}))
        return updated

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the agent, then capture the Gemini session as a trajectory.

        Harbor's built-in copy looks for ``~/.gemini/tmp/**/session-*.json`` and
        runs after ``populate_context_post_run``, so with current Gemini CLI
        versions (which write JSONL to ``.../chats/session-*.jsonl``) no
        trajectory or token counts are recorded. We read the JSONL session,
        reshape it into the single-object form Harbor's parser expects, and
        re-run population so ``trajectory.json`` and token counts are produced.
        """
        await super().run(instruction, environment, context)
        await self._capture_session_trajectory(environment, context)

    async def _capture_session_trajectory(
        self, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Read the JSONL session log and materialize it for Harbor's parser."""
        result = await environment.exec(
            command=(
                "f=$(find ~/.gemini/tmp -type f -name 'session-*.jsonl' 2>/dev/null "
                '| sort | tail -n 1); [ -n "$f" ] && cat "$f" || true'
            )
        )
        raw = (getattr(result, "stdout", "") or "").strip()
        if not raw:
            return

        messages: list[dict[str, Any]] = []
        session_id = "unknown"
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            session_id = record.get("sessionId", session_id)
            # Newer Gemini CLI stores content as a list of parts
            # (e.g. [{"text": ...}]); Harbor's ATIF converter expects a
            # plain string, so flatten before handing it off.
            if "content" in record:
                record["content"] = self._flatten_content(record["content"])
            messages.append(record)

        if not messages:
            return

        (self.logs_dir / "gemini-cli.session.jsonl").write_text(raw, encoding="utf-8")
        trajectory = {"sessionId": session_id, "messages": messages}
        (self.logs_dir / "gemini-cli.trajectory.json").write_text(
            json.dumps(trajectory, indent=2), encoding="utf-8"
        )
        # Reuse Harbor's converter/token tally now that the file exists.
        self.populate_context_post_run(context)

    @staticmethod
    def _flatten_content(content: Any) -> str:
        """Flatten a Gemini ``PartListUnion`` content value to plain text.

        Content may be a string, or a list of parts where each part is a
        string or a dict such as ``{"text": ...}``. Non-text parts (function
        calls, inline data, etc.) are dropped from the readable text.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

"""Gemini CLI backend for SkillFlow skill injection."""

import json
from typing import Any

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.gemini_cli import GeminiCli
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from benchmark.agents.skill_injection_mixin import SkillInjectionMixin
from benchmark.agents.skills import TarGzSkillInjector


class SkillFlowGeminiAgent(SkillInjectionMixin, GeminiCli):
    """Gemini CLI agent that injects skills resolved from a configured source."""

    # Gemini's experimental skills feature only surfaces skills (via the
    # ``activate_skill`` tool) from its native skills directory.
    GEMINI_SKILLS_DIR = "~/.gemini/skills"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Inject skills, then mirror them into Gemini's native skills dir."""
        await super().setup(environment)
        src = TarGzSkillInjector.CONTAINER_SKILLS_DIR
        await environment.exec(
            command=(
                f"if [ -d {src} ]; then mkdir -p {self.GEMINI_SKILLS_DIR} && "
                f"cp -r {src}/. {self.GEMINI_SKILLS_DIR}/; fi"
            )
        )

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """Run Gemini CLI with workspace trust enabled for headless execution."""
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
        """Run the agent, then capture the Gemini session as a trajectory."""
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
        self.populate_context_post_run(context)

    @staticmethod
    def _flatten_content(content: Any) -> str:
        """Flatten a Gemini ``PartListUnion`` content value to plain text."""
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

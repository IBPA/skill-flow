"""Codex backend for SkillFlow evaluation agents."""

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.codex import Codex
from harbor.models.trial.paths import EnvironmentPaths
from mcp_servers.utils.skill_loader import resolve_eval_skill_folders

from benchmark.agents.constants import TASK_NAME_SEPARATOR
from benchmark.agents.skill_injection_mixin import SkillInjectionMixin


@dataclass(frozen=True)
class McpServer:
    """MCP server configuration for Codex agents."""

    name: str
    url: str


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


class AdaptedCodex(Codex):
    """Shared base class for evaluation Codex agents."""

    def __init__(
        self,
        *args: Any,
        reasoning_effort: str | None = None,
        mcp_servers: list[McpServer] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reasoning_effort = reasoning_effort
        self._mcp_servers = mcp_servers or []

    def _build_codex_command(self, model: str, escaped_instruction: str) -> str:
        """Build the codex exec command string."""
        parts = [
            "codex exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            f"--model {model}",
        ]

        if self.reasoning_effort:
            parts.append(f'--config model_reasoning_effort="{self.reasoning_effort}"')

        parts.extend(
            [
                "--json",
                "--",
                escaped_instruction,
                "2>&1 </dev/null | tee",
                str(EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME),
                "&& rm -rf $CODEX_HOME/auth.json",
            ]
        )

        return " ".join(parts)

    def _build_mcp_registration_commands(self) -> str:
        """Build shell commands to register MCP servers with Codex."""
        lines: list[str] = []
        for server in self._mcp_servers:
            lines.append(f"codex mcp add {server.name} --url {server.url}")
        return "\n".join(lines)

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        """Create commands to run the Codex agent."""
        escaped_instruction = shlex.quote(instruction)

        if not self.model_name:
            raise ValueError("Model name is required")

        model = self.model_name.split("/")[-1]

        env = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "CODEX_HOME": (EnvironmentPaths.agent_dir).as_posix(),
        }

        setup_command = """
cat >"$CODEX_HOME/auth.json" <<EOF
{
  "OPENAI_API_KEY": "${OPENAI_API_KEY}"
}
EOF
                """

        if self._mcp_servers:
            setup_command += "\n" + self._build_mcp_registration_commands()

        return [
            ExecInput(command=setup_command, env=env),
            ExecInput(
                command=self._build_codex_command(model, escaped_instruction),
                env=env,
            ),
        ]


class SkillFlowCodexAgent(SkillInjectionMixin, AdaptedCodex):
    """Codex CLI agent that injects skills resolved from a configured source."""

    def _resolve_from_eval_results(self, task_name: str) -> list[Path]:
        """Resolve skill folders via the Codex module namespace."""
        assert self._eval_results
        resolved = resolve_eval_skill_folders(
            self._eval_results,
            self._tasks_dir or Path(),
            task_name,
            self._corpus_dir,
        )
        folders = [s.folder_path for s in resolved]
        top_k = getattr(self, "_eval_results_top_k", None)
        if top_k is not None:
            return folders[:top_k]
        return folders


__all__ = [
    "TASK_NAME_SEPARATOR",
    "AdaptedCodex",
    "McpServer",
    "SkillFlowCodexAgent",
    "get_project_root",
]

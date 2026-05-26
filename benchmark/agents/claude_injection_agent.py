"""Claude Code backend for SkillFlow skill injection."""

import re

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from benchmark.agents.skill_injection_mixin import SkillInjectionMixin
from benchmark.agents.skills import TarGzSkillInjector


class ClaudeUsageLimitError(RuntimeError):
    """Raised when Claude Code hits a subscription/API usage or rate limit."""


# Claude-specific error markers only -- deliberately NOT a bare "429", since a
# task's own output can legitimately mention rate limits.
_USAGE_LIMIT_RE = re.compile(
    r"rate_limit_error"
    r"|usage limit reached"
    r"|Claude AI usage limit"
    r"|API Error: Rate limit"
    r"|\d+-hour limit reached",
    re.IGNORECASE,
)


class SkillFlowClaudeAgent(SkillInjectionMixin, ClaudeCode):
    """Claude Code agent that injects skills resolved from a configured source."""

    # Claude Code discovers personal skills from ``~/.claude/skills`` and its
    # run step copies that tree into ``$CLAUDE_CONFIG_DIR/skills`` before launch.
    CLAUDE_SKILLS_DIR = "~/.claude/skills"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Inject skills, then mirror them into Claude's native skills dir."""
        await super().setup(environment)
        src = TarGzSkillInjector.CONTAINER_SKILLS_DIR
        await environment.exec(
            command=(
                f"if [ -d {src} ]; then mkdir -p {self.CLAUDE_SKILLS_DIR} && "
                f"cp -r {src}/. {self.CLAUDE_SKILLS_DIR}/; fi"
            )
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the agent, then fail loudly if Claude hit a usage/rate limit."""
        await super().run(instruction, environment, context)
        await self._raise_on_usage_limit(environment)

    async def _raise_on_usage_limit(self, environment: BaseEnvironment) -> None:
        """Scan the captured CLI output and raise on a usage/rate-limit marker."""
        result = await environment.exec(
            command="cat /logs/agent/claude-code.txt 2>/dev/null || true"
        )
        text = getattr(result, "stdout", "") or ""
        if _USAGE_LIMIT_RE.search(text):
            raise ClaudeUsageLimitError(
                "Claude Code hit a usage/rate limit during the run; flagging as "
                "an infra error so it is not scored as reward 0. Re-run this "
                "trial after the limit resets."
            )

"""Tests for SkillFlowClaudeAgent (Claude Code backend with skill injection)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.skillflow_injection_agent import (
    SkillFlowClaudeAgent,
    SkillInjectionMixin,
)
from benchmark.agents.skills import SkillManager, TarGzSkillInjector
from harbor.agents.installed.claude_code import ClaudeCode


def _make_claude_agent(
    logs_dir: Path,
    skills_dir: str | None = None,
    eval_results: str | None = None,
    selector_cache: str | None = None,
    tasks_dir: str | None = None,
) -> SkillFlowClaudeAgent:
    """Build a SkillFlowClaudeAgent bypassing the heavy installed-agent init."""
    agent = SkillFlowClaudeAgent.__new__(SkillFlowClaudeAgent)
    agent.logger = MagicMock()
    agent.logs_dir = logs_dir
    agent.model_name = "anthropic/claude-haiku-4-5-20251001"
    agent._eval_results = Path(eval_results) if eval_results else None
    agent._selector_cache = Path(selector_cache) if selector_cache else None
    agent._tasks_dir = Path(tasks_dir) if tasks_dir else None
    agent._corpus_dir = None
    agent._injector = TarGzSkillInjector(logger=agent.logger)
    if skills_dir:
        agent._skill_manager = SkillManager(
            source_dir=Path(skills_dir), logger=agent.logger
        )
    else:
        agent._skill_manager = None
    return agent


class TestSkillFlowClaudeAgentStructure:
    """Tests for the class wiring."""

    def test_inherits_mixin_and_claude_code(self) -> None:
        assert issubclass(SkillFlowClaudeAgent, SkillInjectionMixin)
        assert issubclass(SkillFlowClaudeAgent, ClaudeCode)

    def test_injection_mixin_precedes_claude_code_in_mro(self) -> None:
        # The mixin (skill injection) must run before ClaudeCode in the chain,
        # so the agent's setup -> super() reaches injection before the CLI base.
        mro = SkillFlowClaudeAgent.__mro__
        assert mro.index(SkillInjectionMixin) < mro.index(ClaudeCode)

    def test_name_is_claude_code(self) -> None:
        assert SkillFlowClaudeAgent.name() == "claude-code"


class TestSkillFlowClaudeAgentSetup:
    """Tests that injection runs and chains to the Claude Code base setup."""

    def test_setup_injects_skills_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("x")
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _make_claude_agent(
            logs_dir=logs_dir, skills_dir=str(tmp_path / "skills")
        )

        env = AsyncMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        env.upload_file = AsyncMock()

        # Patch the cooperating base (ClaudeCode) so super().setup() is a no-op.
        with patch.object(ClaudeCode, "setup", new_callable=AsyncMock) as mock_super:
            asyncio.run(agent.setup(env))
            mock_super.assert_called_once_with(env)

        assert env.exec.call_count > 0

    def test_setup_baseline_no_injection(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _make_claude_agent(logs_dir=logs_dir)

        env = AsyncMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        env.upload_file = AsyncMock()

        with patch.object(ClaudeCode, "setup", new_callable=AsyncMock):
            asyncio.run(agent.setup(env))

        # No skills uploaded in baseline mode...
        env.upload_file.assert_not_called()
        # ...but the (guarded, no-op) mirror command is still issued.
        cmds = [c.kwargs.get("command", "") for c in env.exec.call_args_list]
        assert any("~/.claude/skills" in c for c in cmds)

    def test_setup_mirrors_injected_skills_to_claude_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("x")
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _make_claude_agent(
            logs_dir=logs_dir, skills_dir=str(tmp_path / "skills")
        )

        env = AsyncMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        env.upload_file = AsyncMock()

        with patch.object(ClaudeCode, "setup", new_callable=AsyncMock):
            asyncio.run(agent.setup(env))

        cmds = [c.kwargs.get("command", "") for c in env.exec.call_args_list]
        mirror = [c for c in cmds if "~/.claude/skills" in c]
        assert mirror, "expected a mirror command targeting ~/.claude/skills"
        assert "/logs/agent/skills" in mirror[-1]

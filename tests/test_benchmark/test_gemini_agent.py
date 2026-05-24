"""Tests for SkillFlowGeminiAgent (Gemini CLI backend with skill injection)."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.skill_injection import SkillInjectionMixin
from benchmark.agents.skillflow_gemini_agent import SkillFlowGeminiAgent
from benchmark.agents.skills import SkillManager, TarGzSkillInjector
from harbor.agents.installed.gemini_cli import GeminiCli


def _make_gemini_agent(
    logs_dir: Path,
    skills_dir: str | None = None,
    eval_results: str | None = None,
    selector_cache: str | None = None,
    tasks_dir: str | None = None,
) -> SkillFlowGeminiAgent:
    """Build a SkillFlowGeminiAgent bypassing the heavy installed-agent init."""
    agent = SkillFlowGeminiAgent.__new__(SkillFlowGeminiAgent)
    agent.logger = MagicMock()
    agent.logs_dir = logs_dir
    agent.model_name = "google/gemini-2.5-flash"
    agent._version = None
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


class TestSkillFlowGeminiAgentStructure:
    """Tests for the class wiring."""

    def test_inherits_mixin_and_gemini_cli(self) -> None:
        assert issubclass(SkillFlowGeminiAgent, SkillInjectionMixin)
        assert issubclass(SkillFlowGeminiAgent, GeminiCli)

    def test_mro_resolves_setup_to_mixin(self) -> None:
        # The mixin's setup must take precedence over GeminiCli's.
        assert SkillFlowGeminiAgent.setup is SkillInjectionMixin.setup

    def test_name_is_gemini_cli(self) -> None:
        assert SkillFlowGeminiAgent.name() == "gemini-cli"


class TestSkillFlowGeminiAgentSetup:
    """Tests that injection runs and chains to the Gemini CLI base setup."""

    def test_setup_injects_skills_dir(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("x")
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _make_gemini_agent(
            logs_dir=logs_dir, skills_dir=str(tmp_path / "skills")
        )

        env = AsyncMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        env.upload_file = AsyncMock()

        # Patch the cooperating base (GeminiCli) so super().setup() is a no-op.
        with patch.object(GeminiCli, "setup", new_callable=AsyncMock) as mock_super:
            asyncio.run(agent.setup(env))
            mock_super.assert_called_once_with(env)

        assert env.exec.call_count > 0

    def test_setup_baseline_no_injection(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / "my-task__abc123" / "agent"
        logs_dir.mkdir(parents=True)
        agent = _make_gemini_agent(logs_dir=logs_dir)

        env = AsyncMock()
        env.exec = AsyncMock()

        with patch.object(GeminiCli, "setup", new_callable=AsyncMock):
            asyncio.run(agent.setup(env))

        env.exec.assert_not_called()


class TestSkillFlowGeminiAgentRunCommands:
    """Tests for the inherited Gemini CLI command builder."""

    def test_run_command_uses_gemini_model(self, tmp_path: Path) -> None:
        agent = _make_gemini_agent(logs_dir=tmp_path)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}, clear=False):
            commands = agent.create_run_agent_commands("do the task")

        assert len(commands) == 1
        assert "gemini -p" in commands[0].command
        assert "gemini-2.5-flash" in commands[0].command
        assert commands[0].env["GEMINI_API_KEY"] == "k"

"""Tests for SkillFlowGeminiAgent (Gemini CLI backend with skill injection)."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from benchmark.agents.skillflow_injection_agent import (
    SkillFlowGeminiAgent,
    SkillInjectionMixin,
)
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
    agent.model_name = "google/gemini-3.1-flash-lite"
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

    def test_injection_mixin_precedes_gemini_cli_in_mro(self) -> None:
        # The mixin (skill injection) must run before GeminiCli in the chain,
        # so the agent's setup -> super() reaches injection before the CLI base.
        mro = SkillFlowGeminiAgent.__mro__
        assert mro.index(SkillInjectionMixin) < mro.index(GeminiCli)

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
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        env.upload_file = AsyncMock()

        with patch.object(GeminiCli, "setup", new_callable=AsyncMock):
            asyncio.run(agent.setup(env))

        # No skills uploaded in baseline mode...
        env.upload_file.assert_not_called()
        # ...but the (guarded, no-op) mirror command is still issued.
        cmds = [c.kwargs.get("command", "") for c in env.exec.call_args_list]
        assert any("~/.gemini/skills" in c for c in cmds)

    def test_setup_mirrors_injected_skills_to_gemini_dir(self, tmp_path: Path) -> None:
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

        with patch.object(GeminiCli, "setup", new_callable=AsyncMock):
            asyncio.run(agent.setup(env))

        cmds = [c.kwargs.get("command", "") for c in env.exec.call_args_list]
        mirror = [c for c in cmds if "~/.gemini/skills" in c]
        assert mirror, "expected a mirror command targeting ~/.gemini/skills"
        assert "/logs/agent/skills" in mirror[-1]


class TestSkillFlowGeminiAgentRunCommands:
    """Tests for the inherited Gemini CLI command builder."""

    def test_run_command_uses_gemini_model(self, tmp_path: Path) -> None:
        agent = _make_gemini_agent(logs_dir=tmp_path)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}, clear=False):
            commands = agent.create_run_agent_commands("do the task")

        assert len(commands) == 1
        assert "gemini -p" in commands[0].command
        assert "gemini-3.1-flash-lite" in commands[0].command
        assert commands[0].env["GEMINI_API_KEY"] == "k"
        # Workspace trust must be granted or the CLI refuses to run headless.
        assert commands[0].env["GEMINI_CLI_TRUST_WORKSPACE"] == "true"


class TestSkillFlowGeminiAgentTrajectoryCapture:
    """Tests for capturing the JSONL session log + token counts."""

    def _jsonl_session(self) -> str:
        lines = [
            # Newer Gemini CLI stores content as a list of parts.
            {
                "type": "user",
                "content": [{"text": "do it"}],
                "timestamp": "2026-05-24T16:00:00.000Z",
                "sessionId": "sess-123",
            },
            {
                "type": "gemini",
                "content": "done",
                "timestamp": "2026-05-24T16:00:01.000Z",
                "model": "gemini-3.1-flash-lite",
                "tokens": {
                    "input": 100,
                    "output": 20,
                    "cached": 0,
                    "thoughts": 5,
                    "tool": 2,
                },
                "toolCalls": [],
            },
        ]
        return "\n".join(json.dumps(line) for line in lines)

    def test_capture_writes_trajectory_and_tokens(self, tmp_path: Path) -> None:
        agent = _make_gemini_agent(logs_dir=tmp_path)
        env = AsyncMock()
        env.exec = AsyncMock(
            return_value=MagicMock(stdout=self._jsonl_session(), return_code=0)
        )
        context = MagicMock()

        asyncio.run(agent._capture_session_trajectory(env, context))

        # Raw JSONL + reshaped single-object trajectory both persisted.
        assert (tmp_path / "gemini-cli.session.jsonl").exists()
        traj = json.loads((tmp_path / "gemini-cli.trajectory.json").read_text())
        assert traj["sessionId"] == "sess-123"
        assert len(traj["messages"]) == 2
        # List-of-parts content is flattened to a plain string.
        assert traj["messages"][0]["content"] == "do it"
        # Harbor's converter ran and produced ATIF trajectory + token counts.
        assert (tmp_path / "trajectory.json").exists()
        assert context.n_input_tokens == 100
        assert context.n_output_tokens == 27  # output + thoughts + tool

    def test_capture_noop_when_no_session(self, tmp_path: Path) -> None:
        agent = _make_gemini_agent(logs_dir=tmp_path)
        env = AsyncMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", return_code=0))
        context = MagicMock()

        asyncio.run(agent._capture_session_trajectory(env, context))

        assert not (tmp_path / "gemini-cli.trajectory.json").exists()

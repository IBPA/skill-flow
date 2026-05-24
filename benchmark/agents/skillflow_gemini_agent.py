"""Gemini CLI agent that injects skills into evaluation containers.

A different model family / CLI harness than the Codex backend, used to test
whether skill-augmentation results transfer across agent models. The agent
wraps Harbor's :class:`~harbor.agents.installed.gemini_cli.GeminiCli` (which
runs Google's open-source ``gemini`` CLI, authenticates via ``GEMINI_API_KEY``,
and parses Gemini trajectories) and adds the shared skill-injection behaviour.

The model ID is supplied via Harbor's ``--model`` flag and must be in
``provider/model`` form, e.g. ``google/gemini-2.5-flash`` (newest Gemini Flash).
Skill sources are identical to the Codex backend; see
:class:`SkillInjectionMixin`.
"""

from harbor.agents.installed.gemini_cli import GeminiCli

from benchmark.agents.skill_injection import SkillInjectionMixin


class SkillFlowGeminiAgent(SkillInjectionMixin, GeminiCli):
    """Gemini CLI agent that injects skills resolved from a configured source.

    See :class:`SkillInjectionMixin` for the supported skill sources. Unlike the
    Codex backend, the Gemini CLI does not accept a ``reasoning_effort`` setting,
    so callers should not pass one.
    """

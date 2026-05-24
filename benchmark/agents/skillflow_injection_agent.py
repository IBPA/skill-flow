"""Codex agent that injects skills into evaluation containers.

The skill-resolution and injection logic lives in :class:`SkillInjectionMixin`
(shared with the Gemini backend); this module only binds it to the Codex CLI
base class. Supports three mutually exclusive skill sources:

- ``skills_dir``: scan a local directory tree for SKILL.md files (golden injection)
- ``eval_results``: SkillFlow eval results JSON (from ``skill_flow.cli eval``)
- ``selector_cache``: selector cache JSON (from the selector stage)

If no source is provided, no skills are injected (baseline mode).
"""

from benchmark.agents.base import BaseCodexAgent
from benchmark.agents.skill_injection import SkillInjectionMixin


class SkillFlowInjectionAgent(SkillInjectionMixin, BaseCodexAgent):
    """Codex agent that injects skills resolved from a configured source.

    See :class:`SkillInjectionMixin` for the supported skill sources.
    """

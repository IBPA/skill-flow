"""Skill data models for the SkillFlow retrieval system."""

from typing import Any

from pydantic import BaseModel


class SkillRecord(BaseModel, frozen=True):
    """A skill record loaded from the corpus index.

    Stores metadata only — full SKILL.md content is loaded on demand.
    """

    key: str  # e.g. "skillsmp/infographic-creator"
    name: str
    description: str
    source: str  # e.g. "skillsmp"
    local_path: str  # relative path to skill folder from corpus root
    metadata: dict[str, Any] = {}

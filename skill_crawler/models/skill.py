"""Skill data models."""

from datetime import datetime

from pydantic import BaseModel, Field


class SkillMetadata(BaseModel):
    """Metadata for a skill from marketplace API."""

    id: str = Field(..., description="Unique identifier (slug)")
    name: str = Field(..., description="Display name")
    source: str = Field(..., description="Source marketplace (e.g. skillsmp)")
    url: str | None = Field(default=None, description="Source page URL")
    github_url: str | None = Field(default=None, description="GitHub repository URL")
    description: str | None = Field(default=None, description="Skill description")
    author: str | None = Field(default=None, description="Author name")
    tags: list[str] = Field(default_factory=list, description="Skill tags")
    stars: int | None = Field(default=None, description="Star/like count")
    rating: float | None = Field(default=None, description="Average rating")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    updated_at: datetime | None = Field(
        default=None, description="Last update timestamp"
    )
    content_hash: str | None = Field(
        default=None, description="Hash of SKILL.md content for deduplication"
    )
    local_path: str | None = Field(
        default=None, description="Local path after download"
    )

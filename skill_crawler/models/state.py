"""Sync state models for resume capability."""

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class SyncStatus(str, Enum):
    """Status of a sync operation."""

    IDLE = "idle"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class SourceState(BaseModel):
    """Sync state for a single source."""

    source: str = Field(..., description="Source name (e.g. skillsmp)")
    status: SyncStatus = Field(default=SyncStatus.IDLE, description="Current status")
    last_sync: datetime | None = Field(default=None, description="Last sync timestamp")
    cursor: str | None = Field(default=None, description="Pagination cursor for resume")
    page: int = Field(default=0, description="Current page number")
    total_found: int = Field(default=0, description="Total skills found")
    synced_count: int = Field(default=0, description="Successfully synced skills")
    failed_count: int = Field(default=0, description="Failed sync attempts")
    failed_skills: list[str] = Field(
        default_factory=list, description="List of failed skill IDs"
    )
    skipped_count: int = Field(
        default=0, description="Skipped skills (e.g., too large)"
    )
    skipped_skills: list[str] = Field(
        default_factory=list, description="List of skipped skill IDs with reasons"
    )
    error_message: str | None = Field(default=None, description="Last error message")


class SyncState(BaseModel):
    """Global sync state across all sources."""

    sources: dict[str, SourceState] = Field(
        default_factory=dict, description="Per-source sync states"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="Last state update"
    )

    def get_source_state(self, source: str) -> SourceState:
        """Get or create state for a source."""
        if source not in self.sources:
            self.sources[source] = SourceState(source=source)
        return self.sources[source]

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.utcnow()
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "SyncState":
        """Load state from JSON file."""
        if not path.exists():
            return cls()
        return cls.model_validate_json(path.read_text())

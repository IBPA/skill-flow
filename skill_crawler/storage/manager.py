"""Storage manager for downloaded skills."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from skill_crawler.models.skill import SkillMetadata
from skill_crawler.models.state import SyncState


class StorageManager:
    """Manage storage of downloaded skills and metadata."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize storage manager.

        Args:
            base_dir: Base directory for skill storage.
        """
        self.base_dir = base_dir
        self.metadata_dir = base_dir / "_metadata"
        self.index_path = self.metadata_dir / "index.json"
        self.state_path = self.metadata_dir / "sync_state.json"

    def ensure_directories(self) -> None:
        """Create required directories."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def get_skill_dir(self, source: str, skill_id: str) -> Path:
        """Get directory for a specific skill.

        Args:
            source: Source name (e.g. skillsmp).
            skill_id: Skill identifier.

        Returns:
            Path to skill directory.
        """
        return self.base_dir / source / skill_id

    def skill_exists(self, source: str, skill_id: str) -> bool:
        """Check if a skill is already indexed by its API id.

        Args:
            source: Source name.
            skill_id: Skill API identifier.

        Returns:
            True if skill exists in the index.
        """
        index = self.load_index()
        for skill_data in index.get("skills", {}).values():
            if skill_data.get("source") == source and skill_data.get("id") == skill_id:
                return True
        return False

    def save_skill_metadata(self, skill: SkillMetadata) -> None:
        """Save skill metadata to index.

        Args:
            skill: Skill to save.
        """
        index = self.load_index()

        # Use folder name from local_path as key
        folder_name = Path(skill.local_path).name if skill.local_path else skill.id
        key = f"{skill.source}/{folder_name}"

        # Serialize all fields from the model
        skill_data = skill.model_dump(mode="json")
        skill_data["indexed_at"] = datetime.utcnow().isoformat()

        index["skills"][key] = skill_data
        index["last_updated"] = datetime.utcnow().isoformat()
        self._save_index(index)

    def load_index(self) -> dict[str, Any]:
        """Load skill index from disk.

        Returns:
            Index dictionary.
        """
        if not self.index_path.exists():
            return {
                "version": "1.0",
                "skills": {},
                "last_updated": None,
            }
        return cast("dict[str, Any]", json.loads(self.index_path.read_text()))

    def _save_index(self, index: dict[str, Any]) -> None:
        """Save index to disk."""
        self.index_path.write_text(json.dumps(index, indent=2))

    def get_all_skills(self) -> list[dict[str, Any]]:
        """Get all skills from index.

        Returns:
            List of skill dictionaries.
        """
        index = self.load_index()
        return list(index.get("skills", {}).values())

    def get_skills_by_source(self, source: str) -> list[dict[str, Any]]:
        """Get skills from a specific source.

        Args:
            source: Source name.

        Returns:
            List of skill dictionaries.
        """
        return [s for s in self.get_all_skills() if s.get("source") == source]

    def get_stats(self) -> dict[str, int]:
        """Get storage statistics.

        Returns:
            Dictionary with counts per source.
        """
        skills = self.get_all_skills()
        stats: dict[str, int] = {"total": len(skills)}

        for skill in skills:
            source = skill.get("source", "unknown")
            stats[source] = stats.get(source, 0) + 1

        return stats

    def load_sync_state(self) -> SyncState:
        """Load sync state from disk.

        Returns:
            SyncState object.
        """
        return SyncState.load(self.state_path)

    def save_sync_state(self, state: SyncState) -> None:
        """Save sync state to disk.

        Args:
            state: SyncState to save.
        """
        state.save(self.state_path)

    def cleanup_failed_downloads(self, source: str) -> int:
        """Remove incomplete downloads.

        Args:
            source: Source to clean up.

        Returns:
            Number of directories removed.
        """
        source_dir = self.base_dir / source
        if not source_dir.exists():
            return 0

        removed = 0
        for skill_dir in source_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                shutil.rmtree(skill_dir)
                removed += 1

        return removed

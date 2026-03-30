"""Tests for storage manager."""

import tempfile
from pathlib import Path

from skill_crawler.models.skill import SkillMetadata
from skill_crawler.models.state import SyncState, SyncStatus
from skill_crawler.storage.manager import StorageManager


class TestStorageManagerQueries:
    """Test StorageManager query methods."""

    def _create_manager_with_skills(self, tmpdir: str) -> StorageManager:
        """Create a StorageManager with pre-populated skills."""
        manager = StorageManager(Path(tmpdir))
        manager.ensure_directories()

        for i, source in enumerate(["skillsmp", "skillsmp", "skillsmp"]):
            skill = SkillMetadata(
                id=f"skill-{i}",
                name=f"Skill {i}",
                source=source,
                local_path=str(Path(tmpdir) / source / f"skill-{i}"),
            )
            manager.save_skill_metadata(skill)
        return manager

    def test_get_all_skills_empty(self):
        """Test get_all_skills on empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            assert manager.get_all_skills() == []

    def test_get_all_skills_with_data(self):
        """Test get_all_skills returns all indexed skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._create_manager_with_skills(tmpdir)
            skills = manager.get_all_skills()
            assert len(skills) == 3

    def test_get_skills_by_source(self):
        """Test filtering skills by source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._create_manager_with_skills(tmpdir)
            skillsmp = manager.get_skills_by_source("skillsmp")
            assert len(skillsmp) == 3

    def test_get_skills_by_source_empty(self):
        """Test filtering by nonexistent source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._create_manager_with_skills(tmpdir)
            assert manager.get_skills_by_source("nonexistent") == []

    def test_get_stats(self):
        """Test storage statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = self._create_manager_with_skills(tmpdir)
            stats = manager.get_stats()
            assert stats["total"] == 3
            assert stats["skillsmp"] == 3

    def test_get_stats_empty(self):
        """Test stats on empty index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            stats = manager.get_stats()
            assert stats == {"total": 0}


class TestStorageManagerSyncState:
    """Test StorageManager sync state operations."""

    def test_load_sync_state_no_file(self):
        """Test load returns default state when no file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            state = manager.load_sync_state()
            assert isinstance(state, SyncState)
            assert len(state.sources) == 0

    def test_save_and_load_sync_state(self):
        """Test round-trip save and load of sync state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            manager.ensure_directories()

            state = SyncState()
            source_state = state.get_source_state("skillsmp")
            source_state.status = SyncStatus.COMPLETED
            source_state.synced_count = 42

            manager.save_sync_state(state)
            loaded = manager.load_sync_state()
            assert loaded.sources["skillsmp"].status == SyncStatus.COMPLETED
            assert loaded.sources["skillsmp"].synced_count == 42


class TestCleanupFailedDownloads:
    """Test cleanup of failed/incomplete downloads."""

    def test_cleanup_nonexistent_source(self):
        """Test cleanup returns 0 for nonexistent source dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            assert manager.cleanup_failed_downloads("nonexistent") == 0

    def test_cleanup_removes_incomplete(self):
        """Test cleanup removes dirs without SKILL.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            source_dir = Path(tmpdir) / "skillsmp"
            source_dir.mkdir()

            # Complete skill (has SKILL.md)
            complete = source_dir / "complete-skill"
            complete.mkdir()
            (complete / "SKILL.md").write_text("# Skill")

            # Incomplete skill (no SKILL.md)
            incomplete = source_dir / "incomplete-skill"
            incomplete.mkdir()
            (incomplete / "README.md").write_text("# WIP")

            removed = manager.cleanup_failed_downloads("skillsmp")
            assert removed == 1
            assert complete.exists()
            assert not incomplete.exists()

    def test_cleanup_skips_files(self):
        """Test cleanup ignores non-directory entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            source_dir = Path(tmpdir) / "skillsmp"
            source_dir.mkdir()

            (source_dir / "some_file.txt").write_text("not a dir")
            removed = manager.cleanup_failed_downloads("skillsmp")
            assert removed == 0

    def test_cleanup_all_complete(self):
        """Test cleanup removes nothing when all skills are complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            source_dir = Path(tmpdir) / "skillsmp"
            source_dir.mkdir()

            for name in ["skill-a", "skill-b"]:
                skill_dir = source_dir / name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text("# Skill")

            assert manager.cleanup_failed_downloads("skillsmp") == 0


class TestSaveSkillMetadata:
    """Test skill metadata save edge cases."""

    def test_save_without_local_path(self):
        """Test saving skill uses id as key when no local_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            manager.ensure_directories()

            skill = SkillMetadata(
                id="my-skill",
                name="My Skill",
                source="skillsmp",
            )
            manager.save_skill_metadata(skill)

            index = manager.load_index()
            assert "skillsmp/my-skill" in index["skills"]
            assert index["last_updated"] is not None

    def test_load_index_from_disk(self):
        """Test loading a persisted index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            manager.ensure_directories()

            skill = SkillMetadata(
                id="test",
                name="Test",
                source="skillsmp",
                local_path=str(Path(tmpdir) / "skillsmp" / "test"),
            )
            manager.save_skill_metadata(skill)

            # Create a new manager to test loading from disk
            manager2 = StorageManager(Path(tmpdir))
            index = manager2.load_index()
            assert "skillsmp/test" in index["skills"]
            assert index["skills"]["skillsmp/test"]["name"] == "Test"

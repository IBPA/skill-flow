"""Tests for skill crawler package."""

import tempfile
from pathlib import Path

from skill_crawler.config import Settings
from skill_crawler.models.skill import SkillMetadata
from skill_crawler.models.state import SourceState, SyncState, SyncStatus
from skill_crawler.storage.manager import StorageManager


class TestSettings:
    """Test configuration settings."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        assert settings.data_dir == Path("./data")
        assert settings.max_workers == 4
        assert settings.rate_limit_delay == 1.0

    def test_skills_dir_property(self):
        """Test skills_dir computed property."""
        settings = Settings()
        assert settings.skills_dir == Path("./data/skills")

    def test_skillsmp_dir_property(self):
        """Test skillsmp_dir computed property."""
        settings = Settings()
        assert settings.skillsmp_dir == Path("./data/skills/skillsmp")


class TestSkillModel:
    """Test Skill data model."""

    def test_skill_creation(self):
        """Test creating a skill."""
        skill = SkillMetadata(
            id="test-skill",
            name="Test Skill",
            source="skillsmp",
            description="A test skill",
        )
        assert skill.id == "test-skill"
        assert skill.name == "Test Skill"
        assert skill.source == "skillsmp"

    def test_skill_with_urls(self):
        """Test skill with URLs."""
        skill = SkillMetadata(
            id="test",
            name="Test",
            source="skillsmp",
            url="https://skillsmp.com/skills/test",
            github_url="https://github.com/user/test-skill",
        )
        assert str(skill.url) == "https://skillsmp.com/skills/test"
        assert str(skill.github_url) == "https://github.com/user/test-skill"


class TestSyncState:
    """Test sync state models."""

    def test_source_state_defaults(self):
        """Test default values for source state."""
        state = SourceState(source="skillsmp")
        assert state.status == SyncStatus.IDLE
        assert state.synced_count == 0
        assert state.failed_count == 0

    def test_sync_state_get_source_state(self):
        """Test getting or creating source state."""
        state = SyncState()
        source_state = state.get_source_state("skillsmp")
        assert source_state.source == "skillsmp"
        assert "skillsmp" in state.sources

    def test_sync_state_save_load(self):
        """Test saving and loading sync state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "state.json"

            state = SyncState()
            source_state = state.get_source_state("skillsmp")
            source_state.synced_count = 10
            state.save(path)

            loaded = SyncState.load(path)
            assert loaded.sources["skillsmp"].synced_count == 10


class TestStorageManager:
    """Test storage manager."""

    def test_skill_dir_path(self):
        """Test skill directory path generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            skill_dir = manager.get_skill_dir("skillsmp", "test-skill")
            assert skill_dir == Path(tmpdir) / "skillsmp" / "test-skill"

    def test_skill_exists_false(self):
        """Test skill_exists returns False for non-existent skill."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            assert not manager.skill_exists("skillsmp", "nonexistent")

    def test_skill_exists_true(self):
        """Test skill_exists returns True when skill is in index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager(Path(tmpdir))
            manager.ensure_directories()

            # Save a skill to the index
            skill = SkillMetadata(
                id="test-skill",
                name="Test Skill",
                source="skillsmp",
                local_path=str(Path(tmpdir) / "skillsmp" / "test-skill"),
            )
            manager.save_skill_metadata(skill)

            assert manager.skill_exists("skillsmp", "test-skill")

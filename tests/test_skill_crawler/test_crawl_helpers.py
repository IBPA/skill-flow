"""Tests for crawl command helper functions."""

import tempfile
from pathlib import Path

from skill_crawler.commands.crawl import (
    _create_timing_logger,
    _handle_download_result,
    _should_skip_failed,
)
from skill_crawler.models.skill import SkillMetadata
from skill_crawler.models.state import SourceState
from skill_crawler.storage.manager import StorageManager


class TestShouldSkipFailed:
    """Test _should_skip_failed helper."""

    def test_skip_client_error(self):
        """Test skips skill that previously failed with client error."""
        failed = ["skill-1: Client error 404"]
        assert _should_skip_failed("skill-1", failed) is True

    def test_no_skip_server_error(self):
        """Test does not skip skill that failed with server error."""
        failed = ["skill-1: Server error 500"]
        assert _should_skip_failed("skill-1", failed) is False

    def test_no_skip_not_in_list(self):
        """Test does not skip skill not in failed list."""
        assert _should_skip_failed("skill-99", []) is False

    def test_no_skip_different_skill(self):
        """Test does not skip unrelated skills."""
        failed = ["skill-1: Client error 404"]
        assert _should_skip_failed("skill-2", failed) is False


class TestHandleDownloadResult:
    """Test _handle_download_result helper."""

    def _make_fixtures(self, tmpdir: str):
        """Create test fixtures."""
        manager = StorageManager(Path(tmpdir))
        manager.ensure_directories()
        source_state = SourceState(source="skillsmp")
        skill = SkillMetadata(
            id="test-skill",
            name="Test Skill",
            source="skillsmp",
        )
        return manager, source_state, skill

    def test_successful_download(self):
        """Test handling successful download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, source_state, skill = self._make_fixtures(tmpdir)
            _handle_download_result(
                skill=skill,
                success=True,
                result="abc123hash",
                folder_name="test-skill",
                output_dir=Path(tmpdir) / "skillsmp",
                source_state=source_state,
                storage=manager,
            )
            assert source_state.synced_count == 1
            assert skill.content_hash == "abc123hash"
            assert skill.local_path is not None

    def test_skipped_download(self):
        """Test handling skipped download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, source_state, skill = self._make_fixtures(tmpdir)
            _handle_download_result(
                skill=skill,
                success=False,
                result="Skipped: repo too large",
                folder_name="test-skill",
                output_dir=Path(tmpdir),
                source_state=source_state,
                storage=manager,
            )
            assert source_state.skipped_count == 1
            assert len(source_state.skipped_skills) == 1

    def test_failed_download(self):
        """Test handling failed download."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager, source_state, skill = self._make_fixtures(tmpdir)
            _handle_download_result(
                skill=skill,
                success=False,
                result="Connection timeout",
                folder_name="test-skill",
                output_dir=Path(tmpdir),
                source_state=source_state,
                storage=manager,
            )
            assert source_state.failed_count == 1
            assert len(source_state.failed_skills) == 1


class TestCreateTimingLogger:
    """Test _create_timing_logger helper."""

    def test_returns_none_when_not_verbose(self):
        """Test returns None when verbose=False."""
        assert _create_timing_logger(False) is None

    def test_returns_callable_when_verbose(self):
        """Test returns a callable when verbose=True."""
        logger = _create_timing_logger(True)
        assert callable(logger)

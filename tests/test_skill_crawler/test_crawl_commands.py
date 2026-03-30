"""Tests for crawl command implementations."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from skill_crawler.commands.crawl import (
    _create_timing_logger,
    _handle_download_result,
    _should_skip_failed,
    crawl_async,
    search_async,
)
from skill_crawler.config import Settings
from skill_crawler.models.skill import SkillMetadata
from skill_crawler.models.state import SourceState


def _skill(**kwargs) -> SkillMetadata:
    defaults = {"id": "s1", "name": "Skill 1", "source": "skillsmp"}
    return SkillMetadata(**(defaults | kwargs))


class TestShouldSkipFailed:
    """Test client-error skip logic."""

    def test_skip_when_client_error(self):
        failed = ["s1: Client error 404"]
        assert _should_skip_failed("s1", failed) is True

    def test_no_skip_when_server_error(self):
        failed = ["s1: Server error 500"]
        assert _should_skip_failed("s1", failed) is False

    def test_no_skip_when_not_in_list(self):
        assert _should_skip_failed("s2", ["s1: Client error"]) is False


class TestHandleDownloadResult:
    """Test download result handling."""

    def test_success(self):
        skill = _skill()
        state = SourceState(source="skillsmp")
        storage = MagicMock()
        _handle_download_result(
            skill, True, "abc123", "folder", Path("/out"), state, storage
        )
        assert state.synced_count == 1
        assert skill.content_hash == "abc123"
        storage.save_skill_metadata.assert_called_once()

    def test_skipped(self):
        skill = _skill()
        state = SourceState(source="skillsmp")
        storage = MagicMock()
        _handle_download_result(
            skill, False, "Skipped: too big", "folder", Path("/out"), state, storage
        )
        assert state.skipped_count == 1
        assert len(state.skipped_skills) == 1

    def test_failed(self):
        skill = _skill()
        state = SourceState(source="skillsmp")
        storage = MagicMock()
        _handle_download_result(
            skill, False, "Connection refused", "folder", Path("/out"), state, storage
        )
        assert state.failed_count == 1
        assert len(state.failed_skills) == 1


class TestCreateTimingLogger:
    """Test timing logger factory."""

    def test_verbose_returns_callable(self):
        logger = _create_timing_logger(verbose=True)
        assert logger is not None
        logger("test", 1.5)  # Should not raise

    def test_not_verbose_returns_none(self):
        assert _create_timing_logger(verbose=False) is None


class TestCrawlAsync:
    """Test crawl_async orchestration."""

    @pytest.mark.asyncio
    async def test_verbose_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(data_dir=Path(tmpdir))
            with (
                patch(
                    "skill_crawler.commands.crawl.get_settings",
                    return_value=settings,
                ),
                patch(
                    "skill_crawler.commands.crawl._crawl_skillsmp",
                    new_callable=AsyncMock,
                ) as mock_crawl,
            ):
                await crawl_async("skillsmp", dry_run=False, resume=True, verbose=True)
            # Verify timing_callback was passed (not None)
            _, kwargs = mock_crawl.call_args
            assert kwargs.get("timing_callback") is not None or (
                mock_crawl.call_args[0][5] is not None
            )


class TestSearchAsync:
    """Test search_async function."""

    @pytest.mark.asyncio
    async def test_search_with_results(self):
        mock_crawler = AsyncMock()
        mock_crawler.search = AsyncMock(
            return_value=[
                _skill(description="A skill", github_url="https://github.com/a/b"),
            ]
        )
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "skill_crawler.commands.crawl.get_settings",
                return_value=Settings(),
            ),
            patch(
                "skill_crawler.commands.crawl.SkillsMPScraper",
                return_value=mock_crawler,
            ),
        ):
            await search_async("test query", limit=5)
        mock_crawler.search.assert_called_once_with("test query", limit=5)

    @pytest.mark.asyncio
    async def test_search_uses_api_when_token_set(self):
        mock_crawler = AsyncMock()
        mock_crawler.search = AsyncMock(return_value=[])
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=False)

        settings = Settings(skillsmp_api_token="sk_test")
        with (
            patch(
                "skill_crawler.commands.crawl.get_settings",
                return_value=settings,
            ),
            patch(
                "skill_crawler.commands.crawl.SkillsMPAPIClient",
                return_value=mock_crawler,
            ) as mock_cls,
        ):
            await search_async("query", limit=10)
        mock_cls.assert_called_once_with(api_token="sk_test")

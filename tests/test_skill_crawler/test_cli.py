"""Tests for CLI command logic.

Note: typer cannot be imported in the test environment (Python 3.14
compatibility), so we test the underlying async functions and helpers
from commands.crawl directly instead of invoking the CLI runner.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from skill_crawler.commands.crawl import crawl_async, search_async
from skill_crawler.config import Settings


class TestCrawlAsync:
    """Test crawl_async function."""

    @pytest.mark.asyncio
    async def test_crawl_unknown_source(self):
        """Test crawl with an unknown source logs error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = Settings(data_dir=Path(tmpdir))
            with patch(
                "skill_crawler.commands.crawl.get_settings",
                return_value=settings,
            ):
                # Unknown source should not raise, just print error
                await crawl_async("unknown_source", dry_run=True, resume=False)

    @pytest.mark.asyncio
    async def test_crawl_creates_directories(self):
        """Test crawl ensures directories exist."""
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
                ),
            ):
                await crawl_async("skillsmp", dry_run=True, resume=False)
            assert settings.skillsmp_dir.exists()
            assert settings.metadata_dir.exists()

    @pytest.mark.asyncio
    async def test_crawl_default_source(self):
        """Test crawl without source crawls skillsmp."""
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
                ) as mock_smp,
            ):
                await crawl_async(None, dry_run=True, resume=False)
            mock_smp.assert_called_once()


class TestSearchAsync:
    """Test search_async function."""

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """Test search with no results."""
        mock_crawler = AsyncMock()
        mock_crawler.search = AsyncMock(return_value=[])
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
            await search_async("nonexistent", limit=10)
        mock_crawler.search.assert_called_once_with("nonexistent", limit=10)

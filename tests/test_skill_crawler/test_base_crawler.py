"""Tests for BaseCrawler abstract class."""

from types import TracebackType

import pytest
from skill_crawler.crawlers.base import BaseCrawler
from skill_crawler.models.skill import SkillMetadata


class _StubCrawler(BaseCrawler):
    """Minimal concrete crawler for testing base class behaviour."""

    def __init__(self, pages: dict[str | None, list[SkillMetadata]]) -> None:
        self._pages = pages

    @property
    def source_name(self) -> str:
        return "stub"

    async def list_skills(
        self, cursor: str | None = None, limit: int = 100
    ) -> tuple[list[SkillMetadata], str | None]:
        skills = self._pages.get(cursor, [])
        # Return next cursor based on page order
        keys = list(self._pages.keys())
        idx = keys.index(cursor) if cursor in keys else 0
        next_cursor = keys[idx + 1] if idx + 1 < len(keys) else None
        return skills, next_cursor

    async def get_skill(self, skill_id: str) -> SkillMetadata | None:
        return None

    async def search(self, query: str, limit: int = 50) -> list[SkillMetadata]:
        return []

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


def _skill(name: str) -> SkillMetadata:
    return SkillMetadata(id=name, name=name, source="stub")


class TestIterAllSkills:
    """Test iter_all_skills pagination logic."""

    @pytest.mark.asyncio
    async def test_single_page(self):
        crawler = _StubCrawler({None: [_skill("a"), _skill("b")]})
        results = [s async for s in crawler.iter_all_skills()]
        assert [s.id for s in results] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_multi_page(self):
        crawler = _StubCrawler(
            {
                None: [_skill("a")],
                "page2": [_skill("b")],
            }
        )
        results = [s async for s in crawler.iter_all_skills()]
        assert [s.id for s in results] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_empty(self):
        crawler = _StubCrawler({None: []})
        results = [s async for s in crawler.iter_all_skills()]
        assert results == []

    @pytest.mark.asyncio
    async def test_resume_from_cursor(self):
        crawler = _StubCrawler(
            {
                None: [_skill("a")],
                "page2": [_skill("b")],
            }
        )
        results = [s async for s in crawler.iter_all_skills(cursor="page2")]
        assert [s.id for s in results] == ["b"]


class TestAsyncContext:
    """Test async context manager."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self):
        crawler = _StubCrawler({None: []})
        async with crawler as ctx:
            assert ctx is crawler

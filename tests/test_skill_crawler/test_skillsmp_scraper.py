"""Tests for SkillsMP web scraper."""

import warnings
from unittest.mock import AsyncMock, patch

import pytest
from bs4 import BeautifulSoup, Tag
from skill_crawler.crawlers.skillsmp.scraper import SkillsMPScraper, _get_attr_str
from skill_crawler.utils.http import RateLimitedClient


def _mock_html_response(html: str) -> AsyncMock:
    resp = AsyncMock()
    resp.text = html
    return resp


def _patch_get(**kwargs):
    """Shortcut for patching RateLimitedClient.get."""
    return patch.object(RateLimitedClient, "get", new_callable=AsyncMock, **kwargs)


def _tag(html: str) -> Tag:
    """Parse HTML and return the first tag (non-None)."""
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.find()
    assert isinstance(tag, Tag)
    return tag


class TestGetAttrStr:
    """Test the _get_attr_str helper."""

    def test_string_value(self):
        assert _get_attr_str(_tag('<a href="/test">link</a>'), "href") == "/test"

    def test_list_value(self):
        assert _get_attr_str(_tag('<div class="a b">x</div>'), "class") == "a"

    def test_missing_attr(self):
        assert _get_attr_str(_tag("<div>x</div>"), "missing", "default") == "default"

    def test_empty_list_returns_default(self):
        tag = _tag("<div>x</div>")
        tag.attrs["data"] = []  # type: ignore[assignment]
        assert _get_attr_str(tag, "data", "fallback") == "fallback"


class TestSkillsMPScraper:
    """Test SkillsMP scraper."""

    def test_source_name(self):
        scraper = SkillsMPScraper()
        assert scraper.source_name == "skillsmp"

    @pytest.mark.asyncio
    async def test_list_skills_parses_cards(self):
        html = """
        <html><body>
        <div class="skill-card" data-skill-id="test-skill">
            <a href="/skills/test-skill">
                <h3>Test Skill</h3>
            </a>
            <p>A test skill description</p>
            <span class="author">John</span>
            <a href="https://github.com/acme/test">GitHub</a>
            <span class="tag">python</span>
        </div>
        </body></html>
        """
        scraper = SkillsMPScraper()
        with (
            _patch_get(return_value=_mock_html_response(html)),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            skills, next_cursor = await scraper.list_skills()
        assert len(skills) == 1
        assert skills[0].id == "test-skill"
        assert skills[0].name == "Test Skill"
        assert skills[0].github_url == "https://github.com/acme/test"
        assert next_cursor is None

    @pytest.mark.asyncio
    async def test_list_skills_with_pagination(self):
        html = """
        <html><body>
        <div data-skill-id="s1"><h3>S1</h3></div>
        <nav aria-label="pagination">
            <a rel="next" href="?page=2">Next</a>
        </nav>
        </body></html>
        """
        scraper = SkillsMPScraper()
        with (
            _patch_get(return_value=_mock_html_response(html)),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            skills, next_cursor = await scraper.list_skills()
        assert len(skills) == 1
        assert next_cursor == "2"

    @pytest.mark.asyncio
    async def test_list_skills_empty(self):
        scraper = SkillsMPScraper()
        empty = _mock_html_response("<html><body></body></html>")
        with (
            _patch_get(return_value=empty),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", FutureWarning)
            skills, next_cursor = await scraper.list_skills()
        assert skills == []
        assert next_cursor is None

    @pytest.mark.asyncio
    async def test_get_skill_success(self):
        html = """
        <html><body>
        <h1>My Skill</h1>
        <div class="description">A great skill</div>
        <a class="author" href="/u/jane">jane</a>
        <a href="https://github.com/jane/skill">GitHub</a>
        <span class="tag">automation</span>
        <div class="stars"><span class="count">1,234</span></div>
        </body></html>
        """
        scraper = SkillsMPScraper()
        with _patch_get(return_value=_mock_html_response(html)):
            skill = await scraper.get_skill("my-skill")
        assert skill is not None
        assert skill.name == "My Skill"
        assert skill.stars == 1234

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self):
        scraper = SkillsMPScraper()
        with _patch_get(side_effect=RuntimeError("404")):
            result = await scraper.get_skill("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_search(self):
        html = """
        <html><body>
        <div data-skill-id="match1"><h3>Match 1</h3></div>
        <div data-skill-id="match2"><h3>Match 2</h3></div>
        </body></html>
        """
        scraper = SkillsMPScraper()
        with _patch_get(return_value=_mock_html_response(html)):
            results = await scraper.search("test", limit=10)
        assert len(results) == 2

    def test_parse_skill_card_no_id(self):
        """Card without a skill ID returns None."""
        scraper = SkillsMPScraper()
        card = _tag('<div class="skill-card"><p>No id here</p></div>')
        assert scraper._parse_skill_card(card) is None

    @pytest.mark.asyncio
    async def test_aexit_without_session(self):
        scraper = SkillsMPScraper()
        await scraper.__aexit__(None, None, None)

    def test_parse_skill_card_with_link(self):
        """Card with a link extracts skill ID from href."""
        scraper = SkillsMPScraper()
        card = _tag("""
        <div class="skill-card">
            <a href="/skills/from-link"><h3>Skill Name</h3></a>
        </div>
        """)
        skill = scraper._parse_skill_card(card)
        assert skill is not None
        assert skill.id == "from-link"

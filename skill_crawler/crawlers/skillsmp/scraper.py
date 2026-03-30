"""SkillsMP web scraper (fallback when API unavailable)."""

import re
from types import TracebackType
from typing import TYPE_CHECKING
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from skill_crawler.crawlers.base import BaseCrawler
from skill_crawler.models.skill import SkillMetadata
from skill_crawler.utils.http import RateLimitedClient

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager


def _get_attr_str(element: Tag, attr: str, default: str = "") -> str:
    """Get element attribute as string."""
    value = element.get(attr, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value if value else default


class SkillsMPScraper(BaseCrawler):
    """Web scraper for SkillsMP marketplace (fallback)."""

    BASE_URL = "https://skillsmp.com"

    def __init__(
        self,
        rate_limit: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        """Initialize scraper.

        Args:
            rate_limit: Request rate limit in seconds.
            timeout: Request timeout in seconds.
        """
        self._client = RateLimitedClient(
            rate_limit=rate_limit,
            timeout=timeout,
            headers={"User-Agent": "SkillCrawler/1.0"},
        )
        self._session_cm: AbstractAsyncContextManager[RateLimitedClient] | None = None

    @property
    def source_name(self) -> str:
        """Return source name."""
        return "skillsmp"

    async def __aenter__(self) -> "SkillsMPScraper":
        """Enter async context."""
        self._session_cm = self._client.session()
        await self._session_cm.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        if self._session_cm:
            await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
            self._session_cm = None

    async def list_skills(
        self,
        cursor: str | None = None,
        _limit: int = 100,
    ) -> tuple[list[SkillMetadata], str | None]:
        """List skills by scraping skill listing pages.

        Args:
            cursor: Page number as string.
            _limit: Not used (page size determined by site).

        Returns:
            Tuple of (skills, next page cursor).
        """
        page = int(cursor) if cursor else 1
        url = f"{self.BASE_URL}/skills?page={page}"

        response = await self._client.get(url)
        soup = BeautifulSoup(response.text, "lxml")

        skills = []
        skill_cards = soup.select(".skill-card, [data-skill-id], .skill-item")

        for card in skill_cards:
            skill = self._parse_skill_card(card)
            if skill:
                skills.append(skill)

        # Check for next page
        next_page = None
        pagination = soup.select_one(".pagination, nav[aria-label='pagination']")
        if pagination:
            next_link = pagination.select_one(
                "a[rel='next'], .next a, a:contains('Next')"
            )
            if next_link:
                next_page = str(page + 1)

        return skills, next_page

    async def get_skill(self, skill_id: str) -> SkillMetadata | None:
        """Get skill details by scraping skill page.

        Args:
            skill_id: SkillMetadata slug/ID.

        Returns:
            SkillMetadata or None if not found.
        """
        try:
            url = f"{self.BASE_URL}/skills/{skill_id}"
            response = await self._client.get(url)
            soup = BeautifulSoup(response.text, "lxml")

            return self._parse_skill_page(soup, skill_id, url)
        except Exception:
            return None

    async def search(self, query: str, limit: int = 50) -> list[SkillMetadata]:
        """Search skills by scraping search results.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            Matching skills.
        """
        url = f"{self.BASE_URL}/skills"
        params = {"q": query}

        response = await self._client.get(url, params=params)
        soup = BeautifulSoup(response.text, "lxml")

        skills = []
        skill_cards = soup.select(".skill-card, [data-skill-id], .skill-item")

        for card in skill_cards[:limit]:
            skill = self._parse_skill_card(card)
            if skill:
                skills.append(skill)

        return skills

    def _parse_skill_card(self, card: Tag) -> SkillMetadata | None:
        """Parse a skill card from listing page."""
        try:
            # Extract skill ID/slug from link or data attribute
            skill_id = _get_attr_str(card, "data-skill-id")
            link = card.select_one("a[href*='/skills/']")
            if link and isinstance(link, Tag):
                href = _get_attr_str(link, "href")
                match = re.search(r"/skills/([^/\?]+)", href)
                if match:
                    skill_id = match.group(1)

            if not skill_id:
                return None

            # Extract name
            name_elem = card.select_one("h3, h4, .skill-name, [class*='title']")
            name = name_elem.get_text(strip=True) if name_elem else skill_id

            # Extract description
            desc_elem = card.select_one("p, .skill-description, [class*='description']")
            description = desc_elem.get_text(strip=True) if desc_elem else None

            # Extract author
            author_elem = card.select_one(".author, [class*='author'], .creator")
            author = author_elem.get_text(strip=True) if author_elem else None

            # Extract tags
            tag_elems = card.select(".tag, .badge, [class*='tag']")
            tags = [t.get_text(strip=True) for t in tag_elems]

            # Extract GitHub URL
            github_link = card.select_one("a[href*='github.com']")
            github_url: str | None = None
            if github_link and isinstance(github_link, Tag):
                github_url = _get_attr_str(github_link, "href") or None

            return SkillMetadata(
                id=skill_id,
                name=name,
                source=self.source_name,
                url=urljoin(self.BASE_URL, f"/skills/{skill_id}"),
                github_url=github_url,
                description=description,
                author=author,
                tags=tags,
            )
        except Exception:
            return None

    def _parse_skill_page(
        self,
        soup: BeautifulSoup,
        skill_id: str,
        url: str,
    ) -> SkillMetadata | None:
        """Parse skill details from individual skill page."""
        try:
            # Name
            name_elem = soup.select_one("h1, .skill-title, [class*='title']")
            name = name_elem.get_text(strip=True) if name_elem else skill_id

            # Description
            desc_elem = soup.select_one(
                ".skill-description, .description, [class*='description'], article p"
            )
            description = desc_elem.get_text(strip=True) if desc_elem else None

            # Author
            author_elem = soup.select_one(".author, [class*='author'], .creator a")
            author = author_elem.get_text(strip=True) if author_elem else None

            # GitHub URL
            github_link = soup.select_one("a[href*='github.com']")
            github_url: str | None = None
            if github_link and isinstance(github_link, Tag):
                github_url = _get_attr_str(github_link, "href") or None

            # Tags
            tag_elems = soup.select(".tag, .badge, [class*='tag']")
            tags = [t.get_text(strip=True) for t in tag_elems]

            # Stars
            stars_elem = soup.select_one("[class*='star'] .count, .stars")
            stars = None
            if stars_elem:
                num_match = re.search(r"[\d,]+", stars_elem.get_text())
                if num_match:
                    stars = int(num_match.group().replace(",", ""))

            return SkillMetadata(
                id=skill_id,
                name=name,
                source=self.source_name,
                url=url,
                github_url=github_url,
                description=description,
                author=author,
                tags=tags,
                stars=stars,
            )
        except Exception:
            return None

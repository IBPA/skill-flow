"""SkillsMP API client (primary crawler)."""

from datetime import datetime
from types import TracebackType
from typing import TYPE_CHECKING

from skill_crawler.crawlers.base import BaseCrawler
from skill_crawler.models.skill import SkillMetadata
from skill_crawler.utils.http import RateLimitedClient

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager


class SkillsMPAPIClient(BaseCrawler):
    """API client for SkillsMP marketplace."""

    BASE_URL = "https://skillsmp.com/api/v1"

    def __init__(
        self,
        api_token: str | None = None,
        rate_limit: float = 1.0,
        timeout: float = 30.0,
    ) -> None:
        """Initialize API client.

        Args:
            api_token: SkillsMP API token (optional).
            rate_limit: Request rate limit in seconds.
            timeout: Request timeout in seconds.
        """
        self.api_token = api_token
        headers = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        self._client = RateLimitedClient(
            rate_limit=rate_limit,
            timeout=timeout,
            headers=headers,
        )
        self._session_cm: AbstractAsyncContextManager[RateLimitedClient] | None = None

    @property
    def source_name(self) -> str:
        """Return source name."""
        return "skillsmp"

    async def __aenter__(self) -> "SkillsMPAPIClient":
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
        limit: int = 100,
    ) -> tuple[list[SkillMetadata], str | None]:
        """List skills from SkillsMP API using search endpoint.

        Args:
            cursor: Page number as string (1-indexed), or None for first page.
            limit: Maximum results per page.

        Returns:
            Tuple of (skills, next_page as string or None).
        """
        page = int(cursor) if cursor else 1
        params: dict[str, str | int] = {"q": "*", "limit": limit, "page": page}

        response = await self._client.get(
            f"{self.BASE_URL}/skills/search", params=params
        )
        json_data = response.json()

        data = json_data.get("data", {})
        skills = [self._parse_skill(item) for item in data.get("skills", [])]

        pagination = data.get("pagination", {})
        next_cursor = str(page + 1) if pagination.get("hasNext") else None

        return skills, next_cursor

    async def get_skill(self, skill_id: str) -> SkillMetadata | None:
        """Get skill details by ID.

        Args:
            skill_id: SkillMetadata identifier.

        Returns:
            SkillMetadata or None if not found.
        """
        try:
            response = await self._client.get(f"{self.BASE_URL}/skills/{skill_id}")
            data = response.json()
            return self._parse_skill(data)
        except Exception:
            return None

    async def search(self, query: str, limit: int = 50) -> list[SkillMetadata]:
        """Search skills by query.

        Args:
            query: Search query.
            limit: Maximum results.

        Returns:
            Matching skills.
        """
        params: dict[str, str | int] = {"q": query, "limit": limit}
        response = await self._client.get(
            f"{self.BASE_URL}/skills/search", params=params
        )
        json_data = response.json()

        data = json_data.get("data", {})
        return [self._parse_skill(item) for item in data.get("skills", [])]

    def _parse_skill(self, data: dict) -> SkillMetadata:
        """Parse API response into SkillMetadata model."""
        # Handle updatedAt as Unix timestamp (seconds)
        updated_at = None
        if data.get("updatedAt"):
            updated_at = datetime.fromtimestamp(data["updatedAt"])

        return SkillMetadata(
            id=data.get("id", data.get("slug", "")),
            name=data.get("name", ""),
            source=self.source_name,
            url=data.get("skillUrl"),
            github_url=data.get("githubUrl"),
            description=data.get("description"),
            author=data.get("author"),
            tags=data.get("tags", []),
            stars=data.get("stars"),
            updated_at=updated_at,
        )

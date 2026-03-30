"""Abstract base class for skill crawlers."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from types import TracebackType

from skill_crawler.models.skill import SkillMetadata


class BaseCrawler(ABC):
    """Abstract base class for crawling skill marketplaces."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this source."""

    @abstractmethod
    async def list_skills(
        self,
        cursor: str | None = None,
        limit: int = 100,
    ) -> tuple[list[SkillMetadata], str | None]:
        """List skills from the source.

        Args:
            cursor: Pagination cursor for resume.
            limit: Maximum number of skills to return.

        Returns:
            Tuple of (skills list, next cursor or None if done).
        """

    @abstractmethod
    async def get_skill(self, skill_id: str) -> SkillMetadata | None:
        """Get details for a specific skill.

        Args:
            skill_id: Unique skill identifier.

        Returns:
            SkillMetadata details or None if not found.
        """

    @abstractmethod
    async def search(self, query: str, limit: int = 50) -> list[SkillMetadata]:
        """Search for skills matching a query.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching skills.
        """

    async def iter_all_skills(
        self,
        cursor: str | None = None,
    ) -> AsyncIterator[SkillMetadata]:
        """Iterate through all skills with automatic pagination.

        Args:
            cursor: Optional starting cursor for resume.

        Yields:
            Skills one at a time.
        """
        current_cursor = cursor
        while True:
            skills, next_cursor = await self.list_skills(cursor=current_cursor)
            for skill in skills:
                yield skill

            if next_cursor is None:
                break
            current_cursor = next_cursor

    async def __aenter__(self) -> "BaseCrawler":
        """Enter async context."""
        return self

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context - subclasses must implement for cleanup."""

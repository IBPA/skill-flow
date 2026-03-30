"""Searcher protocol — shared interface for all retriever backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from skill_flow.retriever.retriever import SearchResult


@runtime_checkable
class Searcher(Protocol):
    """Protocol that all retriever backends must satisfy."""

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]: ...

    def augment(self, keys: list[str], descriptions: list[str]) -> None: ...

    def add_descriptions(self, descriptions: dict[str, str]) -> None: ...

    def add_contents(self, contents: dict[str, str]) -> None: ...

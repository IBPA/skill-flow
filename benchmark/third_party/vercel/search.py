"""Search the skills.sh API for agent skills."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SKILLS_SH_API = "https://skills.sh/api/search"


def search_skills_sh(
    query: str,
    *,
    top_k: int = 1,
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Search skills.sh for skills matching *query*.

    Returns up to *top_k* non-duplicate skill objects, each containing at
    least ``id``, ``skillId``, ``name``, ``source``, and ``installs``.
    """
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(SKILLS_SH_API, params={"q": query})
        resp.raise_for_status()

    data = resp.json()
    skills: list[dict[str, Any]] = data.get("skills", [])

    # Filter out duplicates flagged by the API
    skills = [s for s in skills if not s.get("isDuplicate", False)]

    return skills[:top_k]

"""LLM-based skill selector for Stage 4 of the retrieval pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

if TYPE_CHECKING:
    from skill_flow.config import SelectorConfig
    from skill_flow.retriever.retriever import SearchResult

logger = logging.getLogger(__name__)


def _load_template(path: str) -> Template:
    """Load a Jinja2 template from a file path."""
    return Template(Path(path).read_text(encoding="utf-8"))


class Selector:
    """Filters search results by LLM relevance judgment with JSON caching."""

    def __init__(self, config: SelectorConfig) -> None:
        self._config = config
        load_dotenv()
        self._client = OpenAI()
        self._cache = self._load_cache()
        self._system_template = _load_template(config.system_instruction)
        self._user_template = _load_template(config.user_instruction)

    @property
    def _cache_path(self) -> Path:
        return Path(self._config.cache_path)

    def _load_cache(self) -> dict[str, list[str]]:
        path = self._cache_path
        if path.exists():
            data: dict[str, list[str]] = json.loads(
                path.read_text(encoding="utf-8")
            )
            logger.info(
                "Loaded %d cached selections from %s", len(data), path,
            )
            return data
        return {}

    def _save_cache(self) -> None:
        path = self._cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _render_system_prompt(self) -> str:
        return self._system_template.render()

    def _render_user_prompt(
        self, query: str, candidates: list[SearchResult],
    ) -> str:
        return self._user_template.render(
            query=query, candidates=candidates,
        )

    @staticmethod
    def _parse_response(
        content: str, index_to_key: dict[int, str],
    ) -> list[str]:
        """Extract a JSON array of candidate numbers from the LLM response.

        Maps 1-based indices back to real candidate keys.
        """
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        parsed: list[int] = json.loads(text)
        return [
            index_to_key[idx]
            for idx in parsed
            if idx in index_to_key
        ]

    def select(
        self,
        query: str,
        candidates: list[SearchResult],
        task_id: str = "",
    ) -> list[SearchResult]:
        """Select relevant skills from candidates via LLM judgment.

        Returns filtered list preserving original scores.
        Falls back to returning all candidates on parse failure.
        """
        if not candidates:
            return []

        trimmed = candidates[: self._config.top_k]
        cache_key = task_id or query

        if cache_key in self._cache:
            logger.debug("Cache hit for %s", cache_key)
            selected = set(self._cache[cache_key])
            return [c for c in trimmed if c.key in selected]

        # Build 1-based index mapping so real keys are never shown to the LLM
        index_to_key = {i: c.key for i, c in enumerate(trimmed, 1)}

        system_prompt = self._render_system_prompt()
        user_prompt = self._render_user_prompt(query, trimmed)
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        content = response.choices[0].message.content or ""
        try:
            selected_keys = self._parse_response(content, index_to_key)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Failed to parse selector response for %s, "
                "returning all candidates. Response: %s",
                cache_key,
                content[:200],
            )
            return list(trimmed)

        self._cache[cache_key] = selected_keys
        self._save_cache()
        logger.info(
            "Selected %d/%d skills for %s",
            len(selected_keys),
            len(trimmed),
            cache_key,
        )

        selected_set = set(selected_keys)
        return [c for c in trimmed if c.key in selected_set]

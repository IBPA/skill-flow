"""LLM-based query generation for cross-encoder reranking."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

if TYPE_CHECKING:
    from skill_flow.config import QueryGenConfig

logger = logging.getLogger(__name__)


class QueryGenerator:
    """Converts verbose task instructions into concise search queries via LLM."""

    def __init__(self, config: QueryGenConfig) -> None:
        self._config = config
        load_dotenv()
        self._client = OpenAI()
        self._cache = self._load_cache()

    @property
    def _cache_path(self) -> Path:
        return Path(self._config.cache_path)

    def _load_cache(self) -> dict[str, str]:
        path = self._cache_path
        if path.exists():
            data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
            logger.info("Loaded %d cached queries from %s", len(data), path)
            return data
        return {}

    def _save_cache(self) -> None:
        path = self._cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._cache, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def generate(self, task_id: str, instruction: str) -> str:
        """Generate a concise search query from a task instruction.

        Returns a cached result if available, otherwise calls the LLM
        and writes through to the cache file.
        """
        if task_id in self._cache:
            logger.debug("Cache hit for %s", task_id)
            return self._cache[task_id]

        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": instruction},
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        content = response.choices[0].message.content
        query = content.strip() if content else instruction

        self._cache[task_id] = query
        self._save_cache()
        logger.info("Generated query for %s: %s", task_id, query)
        return query

"""LLM-based query generation for retrieval and reranking stages."""

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
        self._multi_prompt_template = Path(config.multi_query_prompt).read_text(
            encoding="utf-8"
        )
        load_dotenv()
        self._client = OpenAI()
        self._cache: dict[str, str | list[str]] = self._load_cache()

    @property
    def _cache_path(self) -> Path:
        return Path(self._config.cache_path)

    def _load_cache(self) -> dict[str, str | list[str]]:
        path = self._cache_path
        if path.exists():
            data: dict[str, str | list[str]] = json.loads(
                path.read_text(encoding="utf-8")
            )
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

    @staticmethod
    def _normalize_cached(value: str | list[str]) -> list[str]:
        """Normalize a cached value to a list of strings (backward compat)."""
        if isinstance(value, str):
            return [value]
        return value

    @staticmethod
    def _clean_query(text: str) -> str:
        """Strip whitespace and surrounding quote characters from LLM output."""
        cleaned = text.strip()
        if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
            cleaned = cleaned[1:-1]
        return cleaned

    def generate(self, task_id: str, instruction: str) -> str:
        """Generate a concise search query from a task instruction.

        Returns a cached result if available, otherwise calls the LLM
        and writes through to the cache file.
        """
        if task_id in self._cache:
            logger.debug("Cache hit for %s", task_id)
            cached = self._cache[task_id]
            if isinstance(cached, list):
                return cached[0]
            return cached

        kwargs: dict[str, object] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": self._config.system_prompt},
                {"role": "user", "content": instruction},
            ],
            "max_tokens": self._config.max_tokens,
        }
        if not self._config.model.startswith("gpt-5"):
            kwargs["temperature"] = self._config.temperature
        response = self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]

        content = response.choices[0].message.content
        query = self._clean_query(content) if content else instruction

        self._cache[task_id] = query
        self._save_cache()
        logger.info("Generated query for %s: %s", task_id, query)
        return query

    _MAX_RETRIES = 2

    def generate_multi(self, task_id: str, instruction: str) -> list[str]:
        """Generate one or more search queries from a task instruction.

        When ``num_queries == 1`` delegates to :meth:`generate` and wraps
        the result in a list.  When ``num_queries > 1`` makes a single LLM
        call requesting a JSON array of queries.  Retries up to
        ``_MAX_RETRIES`` times if parsing fails.

        Cached values are normalised: legacy ``str`` entries become
        single-element lists so old caches keep working.
        """
        raw_num = self._config.num_queries
        if isinstance(raw_num, list):
            msg = "num_queries must be int for generate_multi, got list"
            raise TypeError(msg)
        num: int = raw_num

        if num <= 1:
            return [self.generate(task_id, instruction)]

        if task_id in self._cache:
            logger.debug("Cache hit (multi) for %s", task_id)
            return self._normalize_cached(self._cache[task_id])

        system_prompt = self._multi_prompt_template.format(num_queries=num)
        queries: list[str] | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            temp = self._config.temperature + 0.3 * (attempt - 1)
            kwargs_multi: dict[str, object] = {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ],
                "max_tokens": self._config.max_tokens,
            }
            if not self._config.model.startswith("gpt-5"):
                kwargs_multi["temperature"] = min(temp, 1.0)
            response = self._client.chat.completions.create(**kwargs_multi)  # type: ignore[call-overload]
            content = response.choices[0].message.content
            parsed = self._parse_multi_response(content, num)
            if parsed is not None:
                queries = parsed
                break
            logger.warning(
                "Retry %d/%d for %s: LLM returned invalid queries",
                attempt,
                self._MAX_RETRIES,
                task_id,
            )

        if queries is None:
            logger.warning(
                "All retries failed for %s, falling back to %d single-query calls",
                task_id,
                num,
            )
            queries = self._generate_n_single(task_id, instruction, num)

        self._cache[task_id] = queries
        self._save_cache()
        logger.info("Generated %d queries for %s: %s", len(queries), task_id, queries)
        return queries

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """Remove markdown code fences (```json ... ```) from LLM output."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            if first_nl != -1:
                stripped = stripped[first_nl + 1 :]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
        return stripped.strip()

    def _generate_n_single(self, task_id: str, instruction: str, num: int) -> list[str]:
        """Generate *num* diverse queries via repeated single-query LLM calls.

        Each call after the first includes previously generated queries in
        the prompt so the LLM avoids repeating itself.  Temperature is
        bumped by 0.3 per call to encourage diversity.
        """
        queries: list[str] = []
        for i in range(num):
            temp = self._config.temperature + 0.3 * i
            user_msg = instruction
            if queries:
                already = "\n".join(f"- {q}" for q in queries)
                user_msg = (
                    f"{instruction}\n\n"
                    f"Already generated (do NOT repeat these):\n{already}"
                )
            kwargs_single: dict[str, object] = {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": self._config.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": self._config.max_tokens,
            }
            if not self._config.model.startswith("gpt-5"):
                kwargs_single["temperature"] = min(temp, 1.0)
            response = self._client.chat.completions.create(**kwargs_single)  # type: ignore[call-overload]
            content = response.choices[0].message.content
            q = self._clean_query(content) if content else ""
            if q and q not in set(queries):
                queries.append(q)
        logger.info(
            "Single-query fallback for %s: %d / %d requested",
            task_id,
            len(queries),
            num,
        )
        return queries if queries else [instruction]

    _MAX_QUERY_LEN = 500

    @staticmethod
    def _parse_multi_response(
        content: str | None,
        num: int,
    ) -> list[str] | None:
        """Parse the LLM response as a JSON array of short query strings.

        Returns ``None`` when the response is unparseable or contains
        items that look like task output rather than search queries (e.g.
        JSON objects or strings longer than ``_MAX_QUERY_LEN``).
        """
        if not content:
            return None
        text = QueryGenerator._strip_code_fence(content)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse multi-query JSON")
            return None
        if not isinstance(parsed, list):
            logger.warning("Expected JSON array, got %s", type(parsed).__name__)
            return None
        queries: list[str] = []
        for item in parsed:
            if not isinstance(item, str):
                logger.warning(
                    "Non-string item in query array: %s", type(item).__name__
                )
                return None
            stripped = item.strip()
            if not stripped:
                continue
            if len(stripped) > QueryGenerator._MAX_QUERY_LEN:
                logger.warning(
                    "Query too long (%d chars), likely task output", len(stripped)
                )
                return None
            queries.append(stripped)
        return queries[:num] if queries else None

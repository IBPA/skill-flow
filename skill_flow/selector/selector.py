"""LLM-based skill selector for Stage 4 of the retrieval pipeline."""

from __future__ import annotations

import json
import logging
import threading
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


def _load_json_cache(path: Path) -> dict[str, list[str]]:
    """Load a JSON cache file, returning empty dict if missing."""
    if path.exists():
        data: dict[str, list[str]] = json.loads(
            path.read_text(encoding="utf-8"),
        )
        logger.info("Loaded %d cached entries from %s", len(data), path)
        return data
    return {}


def _save_json_cache(
    cache: dict[str, list[str]],
    path: Path,
) -> None:
    """Persist a JSON cache to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


class Selector:
    """Filters search results by LLM relevance judgment with JSON caching.

    Supports an optional two-step flow: relevancy → specificity.
    The specificity step runs only when ``specificity_instruction`` is
    configured and the relevancy step returns results.
    """

    def __init__(self, config: SelectorConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        load_dotenv()
        self._client = OpenAI()
        self._cache = _load_json_cache(self._cache_path)
        self._system_template = _load_template(config.system_instruction)
        self._user_template = _load_template(config.user_instruction)

        self._specificity_template: Template | None = None
        self._specificity_cache: dict[str, list[str]] = {}
        if config.specificity_instruction:
            self._specificity_template = _load_template(
                config.specificity_instruction,
            )
            self._specificity_cache = _load_json_cache(
                self._specificity_cache_path,
            )

    @property
    def _cache_path(self) -> Path:
        return Path(self._config.cache_path)

    @property
    def _specificity_cache_path(self) -> Path:
        return Path(self._config.specificity_cache_path)

    def _render_system_prompt(self) -> str:
        return self._system_template.render()

    def _render_user_prompt(
        self,
        query: str,
        candidates: list[SearchResult],
    ) -> str:
        return self._user_template.render(
            query=query,
            candidates=candidates,
        )

    @staticmethod
    def _parse_response(
        content: str,
        index_to_key: dict[int, str],
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

        raw = json.loads(text)
        # Support both plain array [1, 3] and object {"selected": [1, 3]}
        parsed: list[int] = raw["selected"] if isinstance(raw, dict) else raw
        return [index_to_key[idx] for idx in parsed if idx in index_to_key]

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Make an LLM chat completion call and return the content."""
        kwargs: dict[str, object] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": self._config.max_tokens,
        }
        if not self._config.model.startswith("gpt-5"):
            kwargs["temperature"] = self._config.temperature
        response = self._client.chat.completions.create(**kwargs)  # type: ignore[call-overload]
        return response.choices[0].message.content or ""

    def _run_specificity(
        self,
        query: str,
        candidates: list[SearchResult],
        cache_key: str,
    ) -> list[SearchResult]:
        """Run the specificity step on relevancy-filtered candidates.

        Returns candidates that pass specificity. On parse failure,
        returns an empty list (conservative — better to skip than inject).
        """
        if not self._specificity_template:
            return candidates

        with self._lock:
            if cache_key in self._specificity_cache:
                logger.debug("Specificity cache hit for %s", cache_key)
                selected = set(self._specificity_cache[cache_key])
                return [c for c in candidates if c.key in selected]

        index_to_key = {i: c.key for i, c in enumerate(candidates, 1)}
        system_prompt = self._specificity_template.render()
        user_prompt = self._render_user_prompt(query, candidates)
        content = self._call_llm(system_prompt, user_prompt)

        try:
            selected_keys = self._parse_response(content, index_to_key)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Failed to parse specificity response for %s, "
                "returning empty list (conservative). Response: %s",
                cache_key,
                content[:200],
            )
            selected_keys = []

        with self._lock:
            self._specificity_cache[cache_key] = selected_keys
            _save_json_cache(
                self._specificity_cache,
                self._specificity_cache_path,
            )
        logger.info(
            "Specificity: kept %d/%d skills for %s",
            len(selected_keys),
            len(candidates),
            cache_key,
        )

        selected_set = set(selected_keys)
        return [c for c in candidates if c.key in selected_set]

    def get_labels(self, cache_key: str) -> tuple[set[str], set[str]]:
        """Return (relevant_keys, specific_keys) from caches."""
        with self._lock:
            relevant = set(self._cache.get(cache_key, []))
            specific = set(self._specificity_cache.get(cache_key, []))
        return relevant, specific

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

        input_k = self._config.input_top_k
        if input_k > 0:
            candidates = candidates[:input_k]

        cache_key = task_id or query

        with self._lock:
            if cache_key in self._cache:
                logger.debug("Cache hit for %s", cache_key)
                selected = set(self._cache[cache_key])
                relevant = [c for c in candidates if c.key in selected]
                if self._specificity_template and relevant:
                    return self._run_specificity(query, relevant, cache_key)
                return relevant

        # Build 1-based index mapping so real keys are never shown to the LLM
        index_to_key = {i: c.key for i, c in enumerate(candidates, 1)}

        system_prompt = self._render_system_prompt()
        user_prompt = self._render_user_prompt(query, candidates)
        content = self._call_llm(system_prompt, user_prompt)

        try:
            selected_keys = self._parse_response(content, index_to_key)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                "Failed to parse selector response for %s, "
                "returning all candidates. Response: %s",
                cache_key,
                content[:200],
            )
            return list(candidates)

        max_sel = self._config.max_selected
        if max_sel > 0 and len(selected_keys) > max_sel:
            logger.info(
                "Truncating selection from %d to %d for %s",
                len(selected_keys),
                max_sel,
                cache_key,
            )
            selected_keys = selected_keys[:max_sel]

        with self._lock:
            self._cache[cache_key] = selected_keys
            _save_json_cache(self._cache, self._cache_path)
        logger.info(
            "Selected %d/%d skills for %s",
            len(selected_keys),
            len(candidates),
            cache_key,
        )

        selected_set = set(selected_keys)
        relevant = [c for c in candidates if c.key in selected_set]

        # Step 2: specificity filter (if configured and results non-empty)
        if self._specificity_template and relevant:
            return self._run_specificity(query, relevant, cache_key)

        return relevant

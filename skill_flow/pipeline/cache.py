"""Stage cache utilities for pipeline resume support."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from skill_flow.retriever.retriever import SearchResult

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_STAGE_FILES: list[tuple[int, str, str]] = [
    (4, "pipeline-stage4-selector.json", ""),
    (3, "pipeline-stage3-deep_reranker.json", "pipeline-stage3-deep-reranker.json"),
    (2, "pipeline-stage2-reranker.json", ""),
    (1, "pipeline-stage1-retriever.json", ""),
]


def load_content_maps(
    index_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Load description and content maps from the index directory."""
    desc_path = index_dir / "skill_descriptions.json"
    descriptions: dict[str, str] = (
        json.loads(desc_path.read_text(encoding="utf-8")) if desc_path.exists() else {}
    )
    content_path = index_dir / "skill_contents.json"
    contents: dict[str, str] = (
        json.loads(content_path.read_text(encoding="utf-8"))
        if content_path.exists()
        else {}
    )
    return descriptions, contents


def load_stage_cache(
    path: Path,
    task_ids: set[str],
    descriptions: dict[str, str],
    contents: dict[str, str],
) -> dict[str, list[SearchResult]] | None:
    """Load cached stage results if the file covers all requested tasks.

    Returns ``None`` when the cache is missing or incomplete so the caller
    knows to rerun the stage.
    """
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    cached_ids = {t["task_id"] for t in data["task_results"]}
    if not task_ids.issubset(cached_ids):
        logger.info(
            "Cache %s missing %d tasks — rerunning stage",
            path.name,
            len(task_ids - cached_ids),
        )
        return None

    candidates: dict[str, list[SearchResult]] = {}
    for task_result in data["task_results"]:
        tid = task_result["task_id"]
        if tid not in task_ids:
            continue
        candidates[tid] = [
            SearchResult(
                key=s["key"],
                score=0.0 if isinstance(s["score"], dict) else s["score"],
                description=descriptions.get(s["key"], ""),
                content=contents.get(s["key"], ""),
            )
            for s in task_result["skills"]
        ]

    logger.info("Resumed %d tasks from %s", len(candidates), path.name)
    return candidates


def find_latest_cache(
    output_dir: Path,
    task_ids: set[str],
    descriptions: dict[str, str],
    contents: dict[str, str],
    max_stage: int,
) -> tuple[dict[str, list[SearchResult]] | None, int]:
    """Walk stages in reverse to find the latest complete cache.

    Returns (task_candidates, last_completed_stage_number).
    Returns (None, 0) when no usable cache is found.
    """
    for stage_num, filename, alt_filename in _STAGE_FILES:
        if stage_num > max_stage:
            continue
        cached = load_stage_cache(
            output_dir / filename,
            task_ids,
            descriptions,
            contents,
        )
        if cached is None and alt_filename:
            cached = load_stage_cache(
                output_dir / alt_filename,
                task_ids,
                descriptions,
                contents,
            )
        if cached is not None:
            return cached, stage_num
    return None, 0

"""Run task directories through the SkillFlow pipeline.

Automatically detects ground-truth skills and enters eval mode when present.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from skill_flow.pipeline.cache import find_latest_cache, load_content_maps
from skill_flow.pipeline.ground_truth import load_ground_truth_context
from skill_flow.pipeline.models import (
    load_task_instructions,
    write_latency_summary,
    write_result_cache,
)
from skill_flow.pipeline.stages import (
    run_stage1,
    run_stage2,
    run_stage3,
    run_stage4,
)

if TYPE_CHECKING:
    from skill_flow.config import Config, PipelineConfig
    from skill_flow.retriever.retriever import SearchResult

logger = logging.getLogger(__name__)


def _max_enabled_stage(config: Config) -> int:
    """Determine the highest enabled pipeline stage."""
    if config.models.selector.enabled:
        return 4
    if config.models.reranker.enabled and config.models.deep_reranker.enabled:
        return 3
    if config.models.reranker.enabled:
        return 2
    return 1


def run_pipeline(
    config: Config,
    pipeline_config: PipelineConfig,
) -> dict[str, list[str]]:
    """Run the full SkillFlow pipeline and return task_id -> skill keys."""
    tasks_dir = Path(pipeline_config.tasks_dir)
    output_dir = Path(pipeline_config.output_dir)
    index_dir = Path(config.index.output_index_path)

    tasks = load_task_instructions(
        tasks_dir,
        max_query_chars=pipeline_config.max_query_chars,
        max_tasks=pipeline_config.max_tasks,
        include_tasks=pipeline_config.include_tasks or None,
    )
    if not tasks:
        logger.warning("No tasks found in %s", tasks_dir)
        return {}

    # Detect ground truth for eval mode
    gt_context = load_ground_truth_context(
        tasks_dir,
        index_dir,
        pipeline_config.max_query_chars,
    )

    task_ids = {t.task_id for t in tasks}
    max_stage = _max_enabled_stage(config)
    reranker_cfg = config.models.reranker
    deep_cfg = config.models.deep_reranker

    descriptions, contents = load_content_maps(index_dir)
    cached, last_stage = find_latest_cache(
        output_dir,
        task_ids,
        descriptions,
        contents,
        max_stage,
    )
    task_candidates: dict[str, list[SearchResult]] = cached or {}

    if last_stage < 1:
        run_stage1(
            config,
            index_dir,
            tasks,
            task_candidates,
            output_dir,
            gt_context=gt_context,
            tasks_dir=tasks_dir,
        )
    else:
        logger.info("Skipping Stage 1 (cached, last_stage=%d)", last_stage)

    if reranker_cfg.enabled and last_stage < 2:
        run_stage2(
            config,
            tasks,
            task_candidates,
            output_dir,
            gt_context=gt_context,
            tasks_dir=tasks_dir,
        )
    elif last_stage >= 2:
        logger.info("Skipping Stage 2 (cached, last_stage=%d)", last_stage)

    if reranker_cfg.enabled and deep_cfg.enabled and last_stage < 3:
        run_stage3(
            config,
            tasks,
            task_candidates,
            output_dir,
            gt_context=gt_context,
            tasks_dir=tasks_dir,
        )
    elif last_stage >= 3:
        logger.info("Skipping Stage 3 (cached, last_stage=%d)", last_stage)

    if config.models.selector.enabled and last_stage < 4:
        run_stage4(
            config,
            tasks,
            task_candidates,
            output_dir,
            gt_context=gt_context,
            tasks_dir=tasks_dir,
        )
    elif last_stage >= 4:
        logger.info("Skipping Stage 4 (cached, last_stage=%d)", last_stage)

    cache: dict[str, list[str]] = {
        task.task_id: [r.key for r in task_candidates[task.task_id]] for task in tasks
    }
    write_result_cache(cache, output_dir / "result_cache.json")
    write_latency_summary(output_dir)
    return cache

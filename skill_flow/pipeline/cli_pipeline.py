"""CLI helper for the pipeline command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from skill_flow.pipeline.runner import run_pipeline

if TYPE_CHECKING:
    import argparse

    from skill_flow.config import Config


def run_pipeline_command(args: argparse.Namespace, config: Config) -> None:
    """Bridge CLI args to PipelineConfig and run the pipeline."""
    overrides: dict[str, object] = {}
    if args.tasks_dir is not None:
        overrides["tasks_dir"] = args.tasks_dir
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.max_tasks:
        overrides["max_tasks"] = args.max_tasks
    if args.max_query_chars:
        overrides["max_query_chars"] = args.max_query_chars
    pipeline_config = config.pipeline.model_copy(update=overrides)

    cache = run_pipeline(config, pipeline_config)

    output_dir = Path(pipeline_config.output_dir)
    total_skills = sum(len(v) for v in cache.values())
    avg = total_skills / len(cache) if cache else 0

    print(f"\n{'=' * 50}")
    print(f"  Tasks: {len(cache)}")
    print(f"  Total skills retrieved: {total_skills}")
    print(f"  Avg skills/task: {avg:.1f}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 50}")

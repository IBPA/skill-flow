"""Pipeline data models and utility functions."""

from __future__ import annotations

import json
import logging
import statistics
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from skill_flow.retriever.retriever import SearchResult

logger = logging.getLogger(__name__)


class TaskInstruction(BaseModel, frozen=True):
    """A single task with its instruction query."""

    task_id: str
    query: str


class SkillResult(BaseModel, frozen=True):
    """A retrieved skill with its score and rank."""

    key: str
    score: float | dict[str, int]
    rank: int


class PipelineStageResult(BaseModel, frozen=True):
    """Results for a single task from one pipeline stage."""

    task_id: str
    query: str
    skills: list[SkillResult]
    elapsed_ms: float = 0.0


def load_task_instructions(
    tasks_dir: Path,
    max_query_chars: int = 0,
    max_tasks: int = 0,
    include_tasks: list[str] | None = None,
) -> list[TaskInstruction]:
    """Load task instructions from a directory of task folders."""
    task_dirs = sorted(d for d in tasks_dir.iterdir() if d.is_dir())

    if include_tasks:
        include = set(include_tasks)
        task_dirs = [d for d in task_dirs if d.name in include]

    instructions: list[TaskInstruction] = []
    for task_dir in task_dirs:
        instruction_file = task_dir / "instruction.md"
        if not instruction_file.exists():
            logger.warning("Skipping %s: no instruction.md", task_dir.name)
            continue

        query = instruction_file.read_text(encoding="utf-8").strip()
        if max_query_chars > 0:
            query = query[:max_query_chars]

        instructions.append(TaskInstruction(task_id=task_dir.name, query=query))

    if max_tasks > 0:
        instructions = instructions[:max_tasks]

    logger.info("Loaded %d task instructions from %s", len(instructions), tasks_dir)
    return instructions


def _to_stage_results(
    task: TaskInstruction,
    results: list[SearchResult],
    elapsed_ms: float = 0.0,
) -> PipelineStageResult:
    """Convert search results to stage output format."""
    return PipelineStageResult(
        task_id=task.task_id,
        query=task.query,
        skills=[
            SkillResult(key=r.key, score=r.score, rank=i)
            for i, r in enumerate(results, 1)
        ],
        elapsed_ms=elapsed_ms,
    )


def _to_selector_results(
    task: TaskInstruction,
    candidates: list[SearchResult],
    relevant_keys: set[str],
    specific_keys: set[str],
    elapsed_ms: float = 0.0,
) -> PipelineStageResult:
    """Convert all selector input candidates to stage output with labels."""
    return PipelineStageResult(
        task_id=task.task_id,
        query=task.query,
        skills=[
            SkillResult(
                key=c.key,
                score={
                    "relevancy": int(c.key in relevant_keys),
                    "specificity": int(c.key in specific_keys),
                },
                rank=i,
            )
            for i, c in enumerate(candidates, 1)
        ],
        elapsed_ms=elapsed_ms,
    )


def _write_stage_results(
    path: Path,
    stage_name: str,
    results: list[PipelineStageResult],
) -> None:
    """Write stage results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "stage": stage_name,
        "num_tasks": len(results),
        "task_results": [r.model_dump() for r in results],
    }
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Wrote %s results (%d tasks) to %s",
        stage_name,
        len(results),
        path,
    )


def write_result_cache(
    cache: dict[str, list[str]],
    output_path: Path,
) -> None:
    """Write the final result cache JSON (task_id -> skill keys)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Wrote result cache (%d tasks) to %s",
        len(cache),
        output_path,
    )


def _compute_latency_stats(values: list[float]) -> dict[str, float]:
    """Compute median, mean, p95 from a list of latency values."""
    if not values:
        return {"median_ms": 0.0, "mean_ms": 0.0, "p95_ms": 0.0}
    sorted_vals = sorted(values)
    p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
    return {
        "median_ms": round(statistics.median(sorted_vals), 2),
        "mean_ms": round(statistics.mean(sorted_vals), 2),
        "p95_ms": round(sorted_vals[p95_idx], 2),
    }


def write_latency_summary(output_dir: Path) -> None:
    """Read stage JSON files and write a latency_summary.json."""
    stage_files = sorted(output_dir.glob("pipeline-stage*.json"))
    if not stage_files:
        return

    summary: dict[str, object] = {"stages": {}}
    stages_dict: dict[str, object] = {}

    for stage_file in stage_files:
        data = json.loads(stage_file.read_text(encoding="utf-8"))
        stage_name = data.get("stage", stage_file.stem)
        per_task: dict[str, float] = {}
        values: list[float] = []
        for task_result in data.get("task_results", []):
            ms = task_result.get("elapsed_ms", 0.0)
            per_task[task_result["task_id"]] = ms
            values.append(ms)
        stages_dict[stage_name] = {
            "per_task_ms": per_task,
            "aggregate": _compute_latency_stats(values),
        }

    summary["stages"] = stages_dict
    out_path = output_dir / "latency_summary.json"
    out_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote latency summary to %s", out_path)

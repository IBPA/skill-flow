"""Selector comparison experiment runner with grid-search over input_top_k."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from skill_flow.config import SelectorConfig
from skill_flow.eval.runner_stages import run_selector_evaluation
from skill_flow.eval.utils.helpers import slug

if TYPE_CHECKING:
    from skill_flow.config import SelectorExperimentConfig, SelectorVariant
    from skill_flow.eval.models import EvalReport

logger = logging.getLogger(__name__)


def _get_input_top_k_list(variant: SelectorVariant) -> list[int]:
    k = variant.input_top_k
    return k if isinstance(k, list) else [k]


def _set_input_top_k(variant: SelectorVariant, k: int) -> SelectorVariant:
    return variant.model_copy(update={"input_top_k": k})


def _to_selector_config(
    variant: SelectorVariant,
    cache_dir: str = "",
    include_tasks: list[str] | None = None,
) -> SelectorConfig:
    k = variant.input_top_k
    if isinstance(k, list):
        msg = "input_top_k must be a single int; expand list variants first"
        raise TypeError(msg)
    label = _selector_label(variant, include_tasks)
    if cache_dir:
        cache_path = f"{cache_dir.rstrip('/')}/selector_cache_{label}.json"
    else:
        cache_path = f"outputs/selector_cache_{label}.json"
    return SelectorConfig(
        enabled=True,
        model=variant.model,
        system_instruction=variant.system_instruction,
        user_instruction=variant.user_instruction,
        max_tokens=variant.max_tokens,
        temperature=variant.temperature,
        input_top_k=k,
        cache_path=cache_path,
    )


def _selector_label(
    variant: SelectorVariant,
    include_tasks: list[str] | None = None,
) -> str:
    label = slug(variant.model)
    k = variant.input_top_k
    if isinstance(k, int) and k > 0:
        label = f"{label}-top{k}"
    if include_tasks:
        label = f"{label}-i{len(include_tasks)}"
    return label


def run_selector_experiment(
    config: SelectorExperimentConfig,
) -> list[tuple[str, EvalReport]]:
    """Run selector eval for each variant, return (label, report) pairs."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(config.input_report_path)
    index_dir = Path(config.index_dir)

    results: list[tuple[str, EvalReport]] = []
    for variant in config.selectors:
        if not variant.enabled:
            continue
        for k in _get_input_top_k_list(variant):
            expanded = _set_input_top_k(variant, k)
            label = _selector_label(expanded, config.include_tasks)
            out = output_dir / f"{label}.json"
            sel_cfg = _to_selector_config(
                expanded,
                config.cache_dir,
                config.include_tasks,
            )
            report = run_selector_evaluation(
                input_path,
                sel_cfg,
                Path(config.tasks_dir),
                index_dir,
                max_tasks=config.max_tasks,
                include_tasks=config.include_tasks or None,
                output_path=out,
            )
            results.append((label, report))
            logger.info("Evaluated %s", label)

    return results

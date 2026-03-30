#!/usr/bin/env python3
"""Re-run all retrieval experiments with timing enabled.

Outputs go to outputs/experiments/latency-*/ to avoid overwriting
existing results. Reports will include per-task elapsed_ms.

Usage::

    uv run python scripts/run-latency-experiments.py
    uv run python scripts/run-latency-experiments.py --only retriever
    uv run python scripts/run-latency-experiments.py --only reranker
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from skill_flow.config import RerankerExperimentConfig, RetrieverExperimentConfig
from skill_flow.eval.experiments import run_experiment, run_reranker_experiment

_CONFIGS: dict[str, tuple[str, str]] = {
    "retriever-comparison": (
        "skill_flow/config/experiments/latency/latency-retriever-comparison.json",
        "retriever",
    ),
    "retriever-querygen": (
        "skill_flow/config/experiments/latency/latency-retriever-querygen.json",
        "retriever",
    ),
    "retriever-union": (
        "skill_flow/config/experiments/latency/latency-retriever-union.json",
        "retriever",
    ),
    "reranker-comparison": (
        "skill_flow/config/experiments/latency/latency-reranker-comparison.json",
        "reranker",
    ),
    "deep-reranker": (
        "skill_flow/config/experiments/latency/latency-deep-reranker-comparison.json",
        "reranker",
    ),
}


def _run_one(label: str, config_path: str, exp_type: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"  Config:  {config_path}")
    print(f"{'=' * 60}\n")

    raw = json.loads(Path(config_path).read_text())

    if exp_type == "retriever":
        ret_cfg = RetrieverExperimentConfig.model_validate(raw)
        results = run_experiment(ret_cfg)
    else:
        rer_cfg = RerankerExperimentConfig.model_validate(raw)
        results = run_reranker_experiment(rer_cfg)

    print(f"\n  Completed {label}: {len(results)} reports written")
    for lbl, rpt in results:
        ms = rpt.summary.mean_elapsed_ms
        print(f"    {lbl}: mean_elapsed_ms={ms:.1f}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run latency experiments")
    parser.add_argument(
        "--only",
        choices=list(_CONFIGS.keys()),
        default=None,
        help="Run a single experiment by name (default: all)",
    )
    args = parser.parse_args()

    for label, (config_path, exp_type) in _CONFIGS.items():
        if args.only and label != args.only:
            continue
        _run_one(label, config_path, exp_type)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()

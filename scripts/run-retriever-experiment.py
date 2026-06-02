"""Run a Stage-1 retriever comparison experiment from a config file.

This is the config-driven entrypoint for ``RetrieverExperimentConfig`` (the
multi-variant retriever comparison). It loads the config, runs each retriever
variant over the tasks, and prints the recall/MRR comparison table. The runner
auto-builds any missing dense index and writes per-task report snapshots into
the config's ``output_dir``.

Usage:
    uv run python scripts/run-retriever-experiment.py \
        --config skill_flow/config/experiments/retriever-octen.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from skill_flow.config import RetrieverExperimentConfig
from skill_flow.eval.experiments import print_comparison, run_experiment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        required=True,
        help="Path to a RetrieverExperimentConfig JSON (a 'retrievers' list)",
    )
    args = ap.parse_args()

    cfg = RetrieverExperimentConfig(**json.loads(Path(args.config).read_text()))
    results = run_experiment(cfg)
    print_comparison(results)
    if cfg.output_dir:
        print(f"\nPer-task report snapshots written under: {cfg.output_dir}")


if __name__ == "__main__":
    main()

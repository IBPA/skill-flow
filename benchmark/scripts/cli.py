#!/usr/bin/env python3
"""CLI for evaluation scripts.

Usage:
    uv run python -m benchmark.scripts.cli run [--config PATH]

Examples:
    # Run with default config
    uv run python -m benchmark.scripts.cli run

    # Run with custom config
    uv run python -m benchmark.scripts.cli run --config benchmark/config/default.json
"""

import argparse
import sys
from pathlib import Path

from benchmark.core.config import load_config
from benchmark.core.runner import EvaluationRunner


def cmd_run(args: argparse.Namespace) -> int:
    """Handle 'run' subcommand."""
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    if args.job_name:
        config = config.model_copy(update={"job_name": args.job_name})

    if args.resume:
        retry = config.retry.model_copy(update={"resume": True})
        config = config.model_copy(update={"retry": retry})

    runner = EvaluationRunner(config, config_path=config_path)

    if config.num_runs > 1:
        return runner.run_multiple()
    return runner.run()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Run Harbor benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--config",
        type=str,
        help="Config file path (default: benchmark/config/default.json)",
    )
    run_parser.add_argument(
        "--job-name",
        type=str,
        help="Job name prefix (timestamp will be auto-appended)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous job (requires --job-name)",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

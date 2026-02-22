"""Tests for evaluation CLI."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from benchmark.core.config import (
    EnvironmentConfig,
    EvalConfig,
    RetryConfig,
    TaskConfig,
)
from benchmark.scripts.cli import cmd_run, create_parser


@pytest.fixture
def mock_eval_config() -> EvalConfig:
    """Create a mock eval config for testing."""
    return EvalConfig(
        jobs_dir=Path("outputs/evaluation"),
        model="openai/gpt-5-mini-2025-08-07",
        dataset="terminal-bench@2.0",
        num_runs=1,
        environment=EnvironmentConfig(use_daytona=True, n_concurrent=6),
        tasks=TaskConfig(
            include_tasks=[],
            exclude_tasks=["train-fasttext"],
        ),
        retry=RetryConfig(
            resume=False,
            retry_errors=False,
            retry_tasks=[],
            retry_error_types=["CancelledError"],
        ),
    )


class TestCreateParser:
    """Tests for CLI parser creation."""

    def test_parser_has_subcommands(self) -> None:
        """Test that parser has required subcommands."""
        parser = create_parser()

        # Test 'run' subcommand
        args = parser.parse_args(["run"])
        assert args.command == "run"
        assert args.config is None

    def test_run_with_config(self) -> None:
        """Test run subcommand with config option."""
        parser = create_parser()
        args = parser.parse_args(["run", "--config", "path/to/config.json"])
        assert args.command == "run"
        assert args.config == "path/to/config.json"


class TestCmdRun:
    """Tests for cmd_run function."""

    @patch("benchmark.scripts.cli.EvaluationRunner")
    @patch("benchmark.scripts.cli.load_config")
    def test_run_single(
        self,
        mock_load_config: MagicMock,
        mock_runner_class: MagicMock,
        mock_eval_config: EvalConfig,
    ) -> None:
        """Test running a single evaluation."""
        mock_load_config.return_value = mock_eval_config
        mock_runner = MagicMock()
        mock_runner.run.return_value = 0
        mock_runner_class.return_value = mock_runner

        parser = create_parser()
        args = parser.parse_args(["run"])

        result = cmd_run(args)

        assert result == 0
        mock_load_config.assert_called_once_with(None)
        mock_runner.run.assert_called_once()

    @patch("benchmark.scripts.cli.EvaluationRunner")
    @patch("benchmark.scripts.cli.load_config")
    def test_run_with_custom_config(
        self,
        mock_load_config: MagicMock,
        mock_runner_class: MagicMock,
        mock_eval_config: EvalConfig,
    ) -> None:
        """Test running with custom config file."""
        mock_load_config.return_value = mock_eval_config
        mock_runner = MagicMock()
        mock_runner.run.return_value = 0
        mock_runner_class.return_value = mock_runner

        parser = create_parser()
        args = parser.parse_args(["run", "--config", "custom/config.json"])

        result = cmd_run(args)

        assert result == 0
        mock_load_config.assert_called_once_with(Path("custom/config.json"))

    @patch("benchmark.scripts.cli.EvaluationRunner")
    @patch("benchmark.scripts.cli.load_config")
    def test_run_multiple(
        self,
        mock_load_config: MagicMock,
        mock_runner_class: MagicMock,
        mock_eval_config: EvalConfig,
    ) -> None:
        """Test running multiple evaluations."""
        # Set num_runs > 1 to trigger run_multiple
        mock_eval_config.num_runs = 3
        mock_load_config.return_value = mock_eval_config
        mock_runner = MagicMock()
        mock_runner.run_multiple.return_value = 0
        mock_runner_class.return_value = mock_runner

        parser = create_parser()
        args = parser.parse_args(["run"])

        result = cmd_run(args)

        assert result == 0
        mock_runner.run_multiple.assert_called_once()

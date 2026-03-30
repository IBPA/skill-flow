"""Tests for evaluation configuration models."""

import json
from pathlib import Path
from typing import Any

import pytest
from benchmark.core.config import (
    EnvironmentConfig,
    EvalConfig,
    EvalMode,
    RetryConfig,
    SkillsConfig,
    TaskConfig,
    load_config,
)
from pydantic import ValidationError


def _make_default_config(**overrides: Any) -> dict[str, Any]:
    """Build a config dict with sensible defaults, applying overrides."""
    defaults: dict[str, Any] = {
        "jobs_dir": Path("outputs/evaluation"),
        "model": "openai/gpt-5-mini-2025-08-07",
        "dataset": "terminal-bench@2.0",
        "num_runs": 1,
        "environment": EnvironmentConfig(use_daytona=True, n_concurrent=6),
        "tasks": TaskConfig(
            include_tasks=[],
            exclude_tasks=["train-fasttext"],
        ),
        "retry": RetryConfig(
            resume=False,
            retry_errors=False,
            retry_tasks=[],
            retry_error_types=["CancelledError"],
        ),
    }
    defaults.update(overrides)
    return defaults


class TestEvalMode:
    """Tests for EvalMode enum."""

    def test_mode_values(self) -> None:
        """Test that all expected modes exist."""
        assert EvalMode.BASELINE.value == "baseline"
        assert EvalMode.SKILLS.value == "skills"
        assert EvalMode.MCP.value == "mcp"
        assert EvalMode.SKILLFLOW_INJECTION.value == "skillflow_injection"
        assert EvalMode.SKILLFLOW_CACHED.value == "skillflow_cached"


class TestSkillsConfig:
    """Tests for SkillsConfig."""

    def test_minimal_config(self) -> None:
        """Test creating config with only required field."""
        config = SkillsConfig(skills_dir=Path("outputs/skills/downloaded"))
        assert config.skills_dir == Path("outputs/skills/downloaded")
        assert config.match_skill_to_task is False

    def test_custom_values(self) -> None:
        """Test setting custom values."""
        config = SkillsConfig(
            skills_dir=Path("/custom/path"),
            match_skill_to_task=True,
        )
        assert config.skills_dir == Path("/custom/path")
        assert config.match_skill_to_task is True


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_required_fields(self) -> None:
        """Test that all fields are required."""
        config = EnvironmentConfig(use_daytona=False, n_concurrent=6)
        assert config.use_daytona is False
        assert config.n_concurrent == 6

    def test_concurrent_bounds(self) -> None:
        """Test n_concurrent values."""
        config = EnvironmentConfig(use_daytona=True, n_concurrent=1)
        assert config.n_concurrent == 1

        config = EnvironmentConfig(use_daytona=True, n_concurrent=100)
        assert config.n_concurrent == 100


class TestTaskConfig:
    """Tests for TaskConfig."""

    def test_required_fields(self) -> None:
        """Test that all fields are required."""
        config = TaskConfig(
            include_tasks=[],
            exclude_tasks=["train-fasttext"],
        )
        assert config.include_tasks == []
        assert "train-fasttext" in config.exclude_tasks

    def test_get_all_tasks_from_names(self) -> None:
        """Test get_all_tasks with include_tasks only."""
        config = TaskConfig(
            include_tasks=["task1", "task2"],
            exclude_tasks=[],
        )
        assert config.get_all_tasks() == ["task1", "task2"]


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_required_fields(self) -> None:
        """Test that all fields are required."""
        config = RetryConfig(
            resume=False,
            retry_errors=False,
            retry_tasks=[],
            retry_error_types=["CancelledError"],
        )
        assert config.resume is False
        assert config.retry_errors is False
        assert config.retry_tasks == []
        assert "CancelledError" in config.retry_error_types


class TestEvalConfig:
    """Tests for EvalConfig."""

    def _make_config(self, **overrides: Any) -> EvalConfig:
        """Helper to create EvalConfig with defaults."""
        return EvalConfig(**_make_default_config(**overrides))

    def test_required_fields(self) -> None:
        """Test creating config with required fields."""
        config = self._make_config()
        assert config.mode == EvalMode.BASELINE
        assert config.job_name is None
        assert config.jobs_dir == Path("outputs/evaluation")
        assert config.num_runs == 1
        assert config.skills is None

    def test_mode_derived_baseline(self) -> None:
        """Test mode is BASELINE when no skills config."""
        config = self._make_config()
        assert config.mode == EvalMode.BASELINE

    def test_mode_derived_skills(self, tmp_path: Path) -> None:
        """Test mode is SKILLS when skills config has no peer URL."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = self._make_config(
            skills=SkillsConfig(skills_dir=skills_dir),
        )
        assert config.mode == EvalMode.SKILLS
        assert config.skills is not None
        assert config.skills.skills_dir == skills_dir

    def test_mode_derived_mcp(self) -> None:
        """Test mode is MCP when mcp_url is set."""
        config = self._make_config(mcp_url="http://172.17.0.1:8765/sse")
        assert config.mode == EvalMode.MCP

    def test_mode_derived_skillflow_injection_from_eval_results(self) -> None:
        """Test mode is SKILLFLOW_INJECTION when eval_results is set."""
        config = self._make_config(
            eval_results=Path("outputs/eval-selector-results.json"),
            tasks_dir_for_skills=Path("integration/skillsbench/tasks"),
        )
        assert config.mode == EvalMode.SKILLFLOW_INJECTION

    def test_mode_derived_skillflow_injection_from_selector_cache(self) -> None:
        """Test mode is SKILLFLOW_INJECTION when selector_cache set."""
        config = self._make_config(
            selector_cache=Path("outputs/selector-cache.json"),
            tasks_dir_for_skills=Path("integration/skillsbench/tasks"),
        )
        assert config.mode == EvalMode.SKILLFLOW_INJECTION

    def test_eval_results_takes_precedence(self) -> None:
        """Test eval_results takes precedence over mcp_url."""
        config = self._make_config(
            eval_results=Path("outputs/results.json"),
            mcp_url="http://172.17.0.1:8765/mcp",
        )
        assert config.mode == EvalMode.SKILLFLOW_INJECTION

    def test_mode_derived_skillflow_cached(self) -> None:
        """Test mode is SKILLFLOW_CACHED when mcp_url + cached_skillflow."""
        config = self._make_config(
            mcp_url="http://172.17.0.1:8765/mcp",
            cached_skillflow=True,
        )
        assert config.mode == EvalMode.SKILLFLOW_CACHED

    def test_cached_skillflow_without_mcp_url_is_baseline(self) -> None:
        """Test cached_skillflow alone does not trigger cached mode."""
        config = self._make_config(cached_skillflow=True)
        assert config.mode == EvalMode.BASELINE

    def test_mcp_url_takes_precedence(self, tmp_path: Path) -> None:
        """Test mcp_url takes precedence over skills config."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        config = self._make_config(
            mcp_url="http://172.17.0.1:8765/sse",
            skills=SkillsConfig(skills_dir=skills_dir),
        )
        assert config.mode == EvalMode.MCP

    def test_skills_mode_missing_dir(self) -> None:
        """Test validation error for missing skills directory."""
        with pytest.raises(ValidationError):
            self._make_config(
                skills=SkillsConfig(skills_dir=Path("/nonexistent/skills")),
            )

    def test_tasks_path_instead_of_dataset(self) -> None:
        """Test creating config with tasks_path instead of dataset."""
        config = self._make_config(
            dataset=None, tasks_path=Path("integration/skillbench/tasks")
        )
        assert config.tasks_path == Path("integration/skillbench/tasks")
        assert config.dataset is None

    def test_both_dataset_and_tasks_path_raises(self) -> None:
        """Test validation error when both dataset and tasks_path are set."""
        with pytest.raises(ValidationError, match="Only one of"):
            self._make_config(
                dataset="terminal-bench@2.0",
                tasks_path=Path("integration/skillbench/tasks"),
            )

    def test_neither_dataset_nor_tasks_path_raises(self) -> None:
        """Test validation error when neither dataset nor tasks_path is set."""
        with pytest.raises(ValidationError, match="Either"):
            self._make_config(dataset=None, tasks_path=None)

    def test_benchmark_source_dataset(self) -> None:
        """Test benchmark_source returns dataset name."""
        config = self._make_config()
        assert config.benchmark_source == "terminal-bench@2.0"

    def test_benchmark_source_tasks_path(self) -> None:
        """Test benchmark_source returns tasks_path."""
        config = self._make_config(
            dataset=None, tasks_path=Path("integration/skillbench/tasks")
        )
        assert config.benchmark_source == "integration/skillbench/tasks"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_json(self, tmp_path: Path) -> None:
        """Test loading config from JSON file."""
        config_data = {
            "jobs_dir": "outputs/evaluation",
            "model": "openai/gpt-5-mini-2025-08-07",
            "dataset": "terminal-bench@2.0",
            "num_runs": 1,
            "environment": {"use_daytona": True, "n_concurrent": 6},
            "tasks": {
                "include_tasks": [],
                "exclude_tasks": ["train-fasttext"],
            },
            "retry": {
                "resume": False,
                "retry_errors": False,
                "retry_tasks": [],
                "retry_error_types": ["CancelledError"],
            },
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = load_config(config_file)
        assert config.mode == EvalMode.BASELINE
        assert config.model == "openai/gpt-5-mini-2025-08-07"
        assert config.environment.n_concurrent == 6
        assert config.skills is None

    def test_load_config_with_legacy_mode_field(self, tmp_path: Path) -> None:
        """Test that load_config ignores legacy mode field."""
        config_data = {
            "mode": "baseline",
            "jobs_dir": "outputs/evaluation",
            "model": "openai/gpt-5-mini-2025-08-07",
            "dataset": "terminal-bench@2.0",
            "num_runs": 1,
            "environment": {"use_daytona": True, "n_concurrent": 6},
            "tasks": {
                "include_tasks": [],
                "exclude_tasks": [],
            },
            "retry": {
                "resume": False,
                "retry_errors": False,
                "retry_tasks": [],
                "retry_error_types": ["CancelledError"],
            },
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = load_config(config_file)
        assert config.mode == EvalMode.BASELINE

    def test_load_config_with_tasks_path(self, tmp_path: Path) -> None:
        """Test loading config with tasks_path instead of dataset."""
        config_data = {
            "jobs_dir": "outputs/evaluation",
            "model": "openai/gpt-5-mini-2025-08-07",
            "tasks_path": "integration/skillbench/tasks",
            "num_runs": 1,
            "environment": {"use_daytona": True, "n_concurrent": 6},
            "tasks": {
                "include_tasks": [],
                "exclude_tasks": [],
            },
            "retry": {
                "resume": False,
                "retry_errors": False,
                "retry_tasks": [],
                "retry_error_types": ["CancelledError"],
            },
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        config = load_config(config_file)
        assert config.tasks_path == Path("integration/skillbench/tasks")
        assert config.dataset is None

    def test_load_config_ignores_legacy_peer_key(self, tmp_path: Path) -> None:
        """Test that load_config ignores legacy peer key in JSON."""
        config_data = {
            "jobs_dir": "outputs/evaluation",
            "model": "openai/gpt-5-mini-2025-08-07",
            "dataset": "terminal-bench@2.0",
            "num_runs": 1,
            "environment": {"use_daytona": True, "n_concurrent": 6},
            "tasks": {
                "include_tasks": [],
                "exclude_tasks": ["train-fasttext"],
            },
            "retry": {
                "resume": False,
                "retry_errors": False,
                "retry_tasks": [],
                "retry_error_types": ["CancelledError"],
            },
            "peer": {
                "skills_dir": "outputs/skills/letta",
                "host": "0.0.0.0",
                "port": 8765,
            },
        }

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        # Should not raise even though legacy peer key exists
        config = load_config(config_file)
        assert config.mode == EvalMode.BASELINE

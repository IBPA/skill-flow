"""Configuration models for evaluation scripts."""

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, model_validator

from benchmark.core.paths import get_project_root


class EvalMode(str, Enum):
    """Evaluation mode."""

    BASELINE = "baseline"
    SKILLS = "skills"
    MCP = "mcp"
    SKILLFLOW_INJECTION = "skillflow_injection"
    SKILLFLOW_CACHED = "skillflow_cached"


class SkillsConfig(BaseModel):
    """Configuration for skills-based evaluation."""

    skills_dir: Path
    match_skill_to_task: bool = False


class EnvironmentConfig(BaseModel):
    """Configuration for evaluation environment."""

    use_daytona: bool
    n_concurrent: int


class TaskConfig(BaseModel):
    """Configuration for task selection."""

    include_tasks: list[str]
    exclude_tasks: list[str]

    def get_all_tasks(self) -> list[str]:
        """Get all included tasks."""
        return list(self.include_tasks)


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    resume: bool
    retry_errors: bool
    retry_tasks: list[str]
    retry_error_types: list[str]


class EvalConfig(BaseModel):
    """Main configuration for evaluation runs."""

    job_name: str | None = None
    jobs_dir: Path
    model: str
    reasoning_effort: str | None = None
    dataset: str | None = None
    tasks_path: Path | None = None
    num_runs: int
    mcp_url: str | None = None
    cached_skillflow: bool = False
    selector_cache: Path | None = None
    eval_results: Path | None = None
    tasks_dir_for_skills: Path | None = None
    corpus_dir: Path | None = None

    skills: SkillsConfig | None = None
    environment: EnvironmentConfig
    tasks: TaskConfig
    retry: RetryConfig

    @property
    def mode(self) -> EvalMode:
        """Derive evaluation mode from config shape."""
        _is_injection = self.eval_results is not None or (
            self.selector_cache is not None and not self.cached_skillflow
        )
        if _is_injection:
            return EvalMode.SKILLFLOW_INJECTION
        if self.mcp_url is not None and self.cached_skillflow:
            return EvalMode.SKILLFLOW_CACHED
        if self.mcp_url is not None:
            return EvalMode.MCP
        if self.skills is None:
            return EvalMode.BASELINE
        return EvalMode.SKILLS

    @model_validator(mode="after")
    def validate_config(self) -> "EvalConfig":
        """Validate configuration consistency."""
        if not self.dataset and not self.tasks_path:
            msg = "Either 'dataset' or 'tasks_path' must be provided"
            raise ValueError(msg)

        if self.dataset and self.tasks_path:
            msg = "Only one of 'dataset' or 'tasks_path' may be provided"
            raise ValueError(msg)

        if self.skills and not self.skills.skills_dir.exists():
            msg = f"Skills directory not found: {self.skills.skills_dir}"
            raise ValueError(msg)

        return self

    @property
    def benchmark_source(self) -> str:
        """Return a display-friendly name for the benchmark source."""
        if self.dataset:
            return self.dataset
        if self.tasks_path:
            return str(self.tasks_path)
        return "(unknown)"


def _get_default_config_path() -> Path:
    """Get the default config path."""
    return get_project_root() / "benchmark" / "config" / "default.json"


def load_config(config_path: Path | None = None) -> EvalConfig:
    """Load evaluation config from JSON file."""
    path = config_path or _get_default_config_path()
    with path.open() as f:
        data = json.load(f)
    # Remove keys not part of EvalConfig
    data.pop("peer", None)
    data.pop("mode", None)
    return EvalConfig.model_validate(data)

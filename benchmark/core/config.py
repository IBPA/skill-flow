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
    SKILLFLOW = "skillflow"
    MCP = "mcp"
    SKILLFLOW_EVAL = "skillflow_eval"


class SkillsConfig(BaseModel):
    """Configuration for skills-based evaluation."""

    skills_dir: Path
    skill_list_name: str | None = None
    match_skill_to_task: bool = False
    skillflow_peer_url: str | None = None


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
        if self.eval_results is not None:
            return EvalMode.SKILLFLOW_EVAL
        if self.mcp_url is not None:
            return EvalMode.MCP
        if self.skills is None:
            return EvalMode.BASELINE
        if self.skills.skillflow_peer_url is not None:
            return EvalMode.SKILLFLOW
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

        if self.skills and self.skills.skillflow_peer_url is None:
            if not self.skills.skills_dir.exists():
                msg = f"Skills directory not found: {self.skills.skills_dir}"
                raise ValueError(msg)

            if self.skills.skill_list_name:
                skills_list_file = self._get_skills_list_file()
                if not skills_list_file.exists():
                    msg = f"Skills list file not found: {skills_list_file}"
                    raise ValueError(msg)

        return self

    def _get_skills_list_file(self) -> Path:
        """Get the full path to the skills list file."""
        if not self.skills or not self.skills.skill_list_name:
            msg = "No skill list name provided"
            raise ValueError(msg)

        skillsets_dir = get_project_root() / "benchmark/agents/skillsets"
        return skillsets_dir / f"{self.skills.skill_list_name}.txt"

    def get_skills_list_file(self) -> Path | None:
        """Get skills list file path if configured."""
        if self.skills and self.skills.skill_list_name:
            return self._get_skills_list_file()
        return None

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

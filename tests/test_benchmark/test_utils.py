"""Tests for evaluation utilities."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from benchmark.core.config import (
    EnvironmentConfig,
    EvalConfig,
    RetryConfig,
    SkillsConfig,
    TaskConfig,
)
from benchmark.core.utils import (
    _get_short_model_name,
    count_skills,
    generate_job_name,
    get_project_root,
    load_tasks_from_file,
)


def make_config(
    skills: SkillsConfig | None = None,
    tasks: TaskConfig | None = None,
    dataset: str | None = "terminal-bench@2.0",
    tasks_path: Path | None = None,
) -> EvalConfig:
    """Helper to create EvalConfig with defaults."""
    return EvalConfig(
        jobs_dir=Path("outputs/evaluation"),
        model="openai/gpt-5-mini-2025-08-07",
        dataset=dataset,
        tasks_path=tasks_path,
        num_runs=1,
        skills=skills,
        environment=EnvironmentConfig(use_daytona=False, n_concurrent=6),
        tasks=tasks
        or TaskConfig(
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


class TestGetProjectRoot:
    """Tests for get_project_root."""

    def test_returns_path(self) -> None:
        """Test that it returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_contains_pyproject(self) -> None:
        """Test that project root contains pyproject.toml."""
        root = get_project_root()
        assert (root / "pyproject.toml").exists()


class TestLoadTasksFromFile:
    """Tests for load_tasks_from_file."""

    def test_load_simple_file(self, tmp_path: Path) -> None:
        """Test loading a simple task file."""
        task_file = tmp_path / "tasks.txt"
        task_file.write_text("task1\ntask2\ntask3\n")

        tasks = load_tasks_from_file(task_file)
        assert tasks == ["task1", "task2", "task3"]

    def test_skip_comments(self, tmp_path: Path) -> None:
        """Test that comments are skipped."""
        task_file = tmp_path / "tasks.txt"
        task_file.write_text("task1\n# this is a comment\ntask2\n")

        tasks = load_tasks_from_file(task_file)
        assert tasks == ["task1", "task2"]

    def test_skip_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""
        task_file = tmp_path / "tasks.txt"
        task_file.write_text("task1\n\n\ntask2\n")

        tasks = load_tasks_from_file(task_file)
        assert tasks == ["task1", "task2"]

    def test_inline_comments(self, tmp_path: Path) -> None:
        """Test that inline comments are stripped."""
        task_file = tmp_path / "tasks.txt"
        task_file.write_text("task1 # inline comment\ntask2\n")

        tasks = load_tasks_from_file(task_file)
        assert tasks == ["task1", "task2"]


class TestGetShortModelName:
    """Tests for _get_short_model_name."""

    def test_openai_with_date(self) -> None:
        assert _get_short_model_name("openai/gpt-5-nano-2025-08-07") == "gpt5nano"

    def test_openai_mini(self) -> None:
        assert _get_short_model_name("openai/gpt-5-mini-2025-08-07") == "gpt5mini"

    def test_anthropic_model(self) -> None:
        assert _get_short_model_name("anthropic/claude-sonnet-4-5") == "claudesonnet45"

    def test_no_provider(self) -> None:
        assert _get_short_model_name("codex-mini") == "codexmini"

    def test_single_part(self) -> None:
        assert _get_short_model_name("gpt4o") == "gpt4o"


class TestGenerateJobName:
    """Tests for generate_job_name."""

    @patch("benchmark.core.utils.datetime")
    def test_baseline_mode(self, mock_datetime: Any) -> None:
        """Test job name for baseline mode includes model."""
        mock_datetime.now.return_value.strftime.return_value = "20240101-120000"

        config = make_config()
        job_name = generate_job_name(config)

        assert job_name.startswith("tb-baseline-gpt5mini-")
        assert "20240101-120000" in job_name

    @patch("benchmark.core.utils.datetime")
    def test_skillflow_injection_mode(self, mock_datetime: Any) -> None:
        """Test job name for skillflow injection mode."""
        mock_datetime.now.return_value.strftime.return_value = "20240101-120000"

        config = make_config()
        config = config.model_copy(
            update={
                "eval_results": Path("outputs/eval-results.json"),
                "tasks_dir_for_skills": Path("integration/skillsbench/tasks"),
            }
        )
        job_name = generate_job_name(config)

        assert job_name.startswith("tb-skillflow-injection-gpt5mini-")

    @patch("benchmark.core.utils.datetime")
    def test_skills_mode_with_dir(self, mock_datetime: Any, tmp_path: Path) -> None:
        """Test job name for skills mode using skills dir basename."""
        mock_datetime.now.return_value.strftime.return_value = "20240101-120000"

        skills_dir = tmp_path / "improved"
        skills_dir.mkdir()

        config = make_config(
            skills=SkillsConfig(
                skills_dir=skills_dir,
            ),
        )
        job_name = generate_job_name(config)

        assert "improved" in job_name
        assert job_name.startswith("tb-improved-gpt5mini-")

    @patch("benchmark.core.utils.datetime")
    def test_skills_mode_with_matched(self, mock_datetime: Any, tmp_path: Path) -> None:
        """Test job name includes matched suffix."""
        mock_datetime.now.return_value.strftime.return_value = "20240101-120000"

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = make_config(
            skills=SkillsConfig(
                skills_dir=skills_dir,
                match_skill_to_task=True,
            ),
        )
        job_name = generate_job_name(config)

        assert "-matched-" in job_name

    @patch("benchmark.core.utils.datetime")
    def test_single_task_suffix(self, mock_datetime: Any) -> None:
        """Test job name includes single task name."""
        mock_datetime.now.return_value.strftime.return_value = "20240101-120000"

        config = make_config(
            tasks=TaskConfig(
                include_tasks=["configure-git-webserver"],
                exclude_tasks=[],
            ),
        )
        job_name = generate_job_name(config)

        assert job_name.endswith("-configure-git-webserver")

    @patch("benchmark.core.utils.datetime")
    def test_multiple_tasks_suffix(self, mock_datetime: Any) -> None:
        """Test job name includes task count for multiple tasks."""
        mock_datetime.now.return_value.strftime.return_value = "20240101-120000"

        config = make_config(
            tasks=TaskConfig(
                include_tasks=["task1", "task2", "task3"],
                exclude_tasks=[],
            ),
        )
        job_name = generate_job_name(config)

        assert job_name.endswith("-3tasks")

    @patch("benchmark.core.utils.datetime")
    def test_includes_reasoning_effort(self, mock_datetime: Any) -> None:
        """Test job name includes reasoning effort when set."""
        mock_datetime.now.return_value.strftime.return_value = "20240101-120000"

        config = make_config()
        config = config.model_copy(update={"reasoning_effort": "high"})
        job_name = generate_job_name(config)

        assert "gpt5mini-high-" in job_name

    @patch("benchmark.core.utils.datetime")
    def test_tasks_path_prefix(self, mock_datetime: Any) -> None:
        """Test job name uses tasks_path parent for prefix."""
        mock_datetime.now.return_value.strftime.return_value = "20240101-120000"

        config = make_config(
            dataset=None,
            tasks_path=Path("integration/skillbench/tasks"),
        )
        job_name = generate_job_name(config)

        assert job_name.startswith("sk-baseline-gpt5mini-")


class TestCountSkills:
    """Tests for count_skills."""

    def test_count_from_list_file(self, tmp_path: Path) -> None:
        """Test counting skills from a list file."""
        skills_list = tmp_path / "skills.txt"
        skills_list.write_text("skill1\nskill2\nskill3\n")

        count = count_skills(tmp_path, skills_list)
        assert count == 3

    def test_count_from_directory(self, tmp_path: Path) -> None:
        """Test counting skills from directory."""
        # Create skill directories with SKILL.md
        for i in range(5):
            skill_dir = tmp_path / f"skill{i}"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(f"# Skill {i}")

        count = count_skills(tmp_path)
        assert count == 5

    def test_count_empty_directory(self, tmp_path: Path) -> None:
        """Test counting skills in empty directory."""
        count = count_skills(tmp_path)
        assert count == 0

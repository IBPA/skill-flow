"""Tests for evaluation runner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmark.core.commands import (
    build_harbor_run_command,
    build_mode_args,
    build_resume_command,
    build_retry_errors_command,
    build_task_args,
)
from benchmark.core.config import (
    EnvironmentConfig,
    EvalConfig,
    RetryConfig,
    SkillsConfig,
    TaskConfig,
)
from benchmark.core.display import print_config, print_multi_config
from benchmark.core.runner import EvaluationRunner


def make_config(
    job_name: str | None = None,
    jobs_dir: Path | None = None,
    skills: SkillsConfig | None = None,
    environment: EnvironmentConfig | None = None,
    tasks: TaskConfig | None = None,
    retry: RetryConfig | None = None,
    num_runs: int = 1,
    dataset: str | None = "terminal-bench@2.0",
    tasks_path: Path | None = None,
    mcp_url: str | None = None,
    cached_skillflow: bool = False,
    selector_cache: Path | None = None,
    eval_results: Path | None = None,
    tasks_dir_for_skills: Path | None = None,
    corpus_dir: Path | None = None,
) -> EvalConfig:
    """Helper to create EvalConfig with defaults."""
    return EvalConfig(
        job_name=job_name,
        jobs_dir=jobs_dir or Path("outputs/evaluation"),
        model="openai/gpt-5-mini-2025-08-07",
        dataset=dataset,
        tasks_path=tasks_path,
        num_runs=num_runs,
        mcp_url=mcp_url,
        cached_skillflow=cached_skillflow,
        selector_cache=selector_cache,
        eval_results=eval_results,
        tasks_dir_for_skills=tasks_dir_for_skills,
        corpus_dir=corpus_dir,
        skills=skills,
        environment=environment or EnvironmentConfig(use_daytona=False, n_concurrent=6),
        tasks=tasks
        or TaskConfig(
            include_tasks=[],
            exclude_tasks=["train-fasttext"],
        ),
        retry=retry
        or RetryConfig(
            resume=False,
            retry_errors=False,
            retry_tasks=[],
            retry_error_types=["CancelledError"],
        ),
    )


class TestEvaluationRunner:
    """Tests for EvaluationRunner class."""

    def test_init(self) -> None:
        """Test runner initialization."""
        config = make_config()
        runner = EvaluationRunner(config)
        assert runner.config == config
        assert runner.project_root.exists()

    @patch("benchmark.core.runner.EvaluationRunner._execute_command")
    @patch("benchmark.core.display.print_config")
    def test_run_new_baseline(
        self, mock_print: MagicMock, mock_execute: MagicMock
    ) -> None:
        """Test running a new baseline evaluation."""
        mock_execute.return_value = 0

        config = make_config(job_name="test-job")
        runner = EvaluationRunner(config)
        result = runner.run()

        assert result == 0
        mock_execute.assert_called_once()
        cmd = mock_execute.call_args[0][0]
        assert "harbor" in cmd
        assert "--agent-import-path" in cmd
        assert "SkillFlowInjectionAgent" in " ".join(cmd)

    @patch("benchmark.core.runner.EvaluationRunner._execute_command")
    @patch("benchmark.core.display.print_config")
    def test_run_new_skills(
        self, mock_print: MagicMock, mock_execute: MagicMock, tmp_path: Path
    ) -> None:
        """Test running a new skills evaluation."""
        mock_execute.return_value = 0

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = make_config(
            job_name="test-skills",
            skills=SkillsConfig(
                skills_dir=skills_dir,
            ),
        )
        runner = EvaluationRunner(config)
        result = runner.run()

        assert result == 0
        cmd = mock_execute.call_args[0][0]
        assert "--agent-import-path" in cmd
        assert "SkillFlowInjectionAgent" in " ".join(cmd)

    @patch("benchmark.core.display.print_config")
    def test_run_resume_missing_dir(
        self, mock_print: MagicMock, tmp_path: Path
    ) -> None:
        """Test resume fails when job directory doesn't exist."""
        config = make_config(
            job_name="nonexistent-job",
            jobs_dir=tmp_path,
            retry=RetryConfig(
                resume=True,
                retry_errors=False,
                retry_tasks=[],
                retry_error_types=["CancelledError"],
            ),
        )
        runner = EvaluationRunner(config)
        result = runner.run()

        assert result == 1

    @patch("benchmark.core.runner.EvaluationRunner._execute_command")
    @patch("benchmark.core.runner.fix_docker_file_ownership")
    @patch("benchmark.core.display.print_config")
    def test_run_resume_existing_dir(
        self,
        mock_print: MagicMock,
        mock_fix_ownership: MagicMock,
        mock_execute: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test resume works when job directory exists."""
        mock_execute.return_value = 0

        job_dir = tmp_path / "test-job"
        job_dir.mkdir()

        config = make_config(
            job_name="test-job",
            jobs_dir=tmp_path,
            retry=RetryConfig(
                resume=True,
                retry_errors=False,
                retry_tasks=[],
                retry_error_types=["CancelledError"],
            ),
        )
        runner = EvaluationRunner(config)
        result = runner.run()

        assert result == 0
        mock_fix_ownership.assert_called_once()
        cmd = mock_execute.call_args[0][0]
        assert "resume" in cmd

    @patch("benchmark.core.display.print_config")
    def test_run_retry_errors_missing_dir(
        self, mock_print: MagicMock, tmp_path: Path
    ) -> None:
        """Test retry errors fails when job directory doesn't exist."""
        config = make_config(
            job_name="nonexistent-job",
            jobs_dir=tmp_path,
            retry=RetryConfig(
                resume=False,
                retry_errors=True,
                retry_tasks=[],
                retry_error_types=["CancelledError"],
            ),
        )
        runner = EvaluationRunner(config)
        result = runner.run()

        assert result == 1

    @patch("benchmark.core.runner.EvaluationRunner._execute_command")
    @patch("benchmark.core.runner.fix_docker_file_ownership")
    @patch("benchmark.core.display.print_config")
    def test_run_retry_errors_existing_dir(
        self,
        mock_print: MagicMock,
        mock_fix_ownership: MagicMock,
        mock_execute: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test retry errors works when job directory exists."""
        mock_execute.return_value = 0

        job_dir = tmp_path / "test-job"
        job_dir.mkdir()

        config = make_config(
            job_name="test-job",
            jobs_dir=tmp_path,
            retry=RetryConfig(
                resume=False,
                retry_errors=True,
                retry_tasks=[],
                retry_error_types=["CancelledError"],
            ),
        )
        runner = EvaluationRunner(config)
        result = runner.run()

        assert result == 0
        cmd = mock_execute.call_args[0][0]
        assert "--filter-error-type" in cmd

    @patch("benchmark.core.display.print_config")
    def test_run_retry_tasks_missing_dir(
        self, mock_print: MagicMock, tmp_path: Path
    ) -> None:
        """Test retry tasks fails when job directory doesn't exist."""
        config = make_config(
            job_name="nonexistent-job",
            jobs_dir=tmp_path,
            retry=RetryConfig(
                resume=False,
                retry_errors=False,
                retry_tasks=["task1"],
                retry_error_types=["CancelledError"],
            ),
        )
        runner = EvaluationRunner(config)
        result = runner.run()

        assert result == 1

    @patch("benchmark.core.runner.EvaluationRunner._execute_command")
    @patch("benchmark.core.runner.fix_docker_file_ownership")
    @patch("benchmark.core.display.print_config")
    def test_run_retry_tasks_existing_dir(
        self,
        mock_print: MagicMock,
        mock_fix_ownership: MagicMock,
        mock_execute: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test retry tasks works and deletes trial directories."""
        mock_execute.return_value = 0

        job_dir = tmp_path / "test-job"
        job_dir.mkdir()

        # Create trial directories
        trial_dir = job_dir / "task1__abc123"
        trial_dir.mkdir()

        config = make_config(
            job_name="test-job",
            jobs_dir=tmp_path,
            retry=RetryConfig(
                resume=False,
                retry_errors=False,
                retry_tasks=["task1"],
                retry_error_types=["CancelledError"],
            ),
        )
        runner = EvaluationRunner(config)
        result = runner.run()

        assert result == 0
        assert not trial_dir.exists()  # Trial dir should be deleted

    def test_generate_job_prefix_baseline(self) -> None:
        """Test generating job prefix for baseline includes model."""
        config = make_config()
        runner = EvaluationRunner(config)
        prefix = runner._generate_job_prefix()

        assert prefix == "tb-baseline-gpt5mini"

    def test_generate_job_prefix_skills_with_options(self, tmp_path: Path) -> None:
        """Test job prefix with skill options."""
        skills_dir = tmp_path / "my_skills"
        skills_dir.mkdir()

        config = make_config(
            skills=SkillsConfig(
                skills_dir=skills_dir,
                match_skill_to_task=True,
            ),
        )
        runner = EvaluationRunner(config)
        prefix = runner._generate_job_prefix()

        assert "my_skills" in prefix
        assert "matched" in prefix
        assert "gpt5mini" in prefix

    def test_generate_job_prefix_with_single_task(self) -> None:
        """Test job prefix with single task."""
        config = make_config(
            tasks=TaskConfig(
                include_tasks=["single-task"],
                exclude_tasks=[],
            ),
        )
        runner = EvaluationRunner(config)
        prefix = runner._generate_job_prefix()

        assert "single-task" in prefix

    def test_generate_job_prefix_with_multiple_tasks(self) -> None:
        """Test job prefix with multiple tasks."""
        config = make_config(
            tasks=TaskConfig(
                include_tasks=["task1", "task2", "task3"],
                exclude_tasks=[],
            ),
        )
        runner = EvaluationRunner(config)
        prefix = runner._generate_job_prefix()

        assert "3tasks" in prefix

    def test_generate_job_prefix_with_reasoning_effort(self) -> None:
        """Test job prefix includes reasoning effort."""
        config = make_config()
        config = config.model_copy(update={"reasoning_effort": "high"})
        runner = EvaluationRunner(config)
        prefix = runner._generate_job_prefix()

        assert prefix == "tb-baseline-gpt5mini-high"

    @patch("benchmark.core.runner.EvaluationRunner._execute_command")
    def test_run_multiple_success(self, mock_execute: MagicMock) -> None:
        """Test running multiple evaluations successfully."""
        mock_execute.return_value = 0

        config = make_config(num_runs=2)
        runner = EvaluationRunner(config)
        result = runner.run_multiple()

        assert result == 0
        assert mock_execute.call_count == 2

    @patch("benchmark.core.runner.EvaluationRunner._execute_command")
    def test_run_multiple_failure(self, mock_execute: MagicMock) -> None:
        """Test running multiple evaluations with failure."""
        mock_execute.side_effect = [0, 1]  # First succeeds, second fails

        config = make_config(num_runs=3)
        runner = EvaluationRunner(config)
        result = runner.run_multiple()

        assert result == 1
        assert mock_execute.call_count == 2  # Stops after failure


class TestBuildCommands:
    """Tests for command building functions."""

    def test_build_task_args_with_tasks(self) -> None:
        """Test building task arguments."""
        config = make_config(
            tasks=TaskConfig(
                include_tasks=["task1", "task2"],
                exclude_tasks=[],
            ),
        )
        args = build_task_args(config)

        assert "--task-name" in args
        assert "task1" in args
        assert "task2" in args

    def test_build_task_args_with_exclude(self) -> None:
        """Test building task arguments with exclude."""
        config = make_config(
            tasks=TaskConfig(
                include_tasks=[],
                exclude_tasks=["slow-task"],
            ),
        )
        args = build_task_args(config)

        assert "--exclude-task-name" in args
        assert "slow-task" in args

    def test_build_harbor_run_command_with_daytona(self) -> None:
        """Test building harbor command with daytona."""
        config = make_config(
            environment=EnvironmentConfig(use_daytona=True, n_concurrent=6),
        )
        cmd = build_harbor_run_command(config, "test-job")

        assert "--env" in cmd
        assert "daytona" in cmd

    def test_build_harbor_run_command_with_dataset(self) -> None:
        """Test building harbor command uses --dataset for dataset configs."""
        config = make_config()
        cmd = build_harbor_run_command(config, "test-job")

        assert "--dataset" in cmd
        assert "terminal-bench@2.0" in cmd
        assert "-p" not in cmd

    def test_build_harbor_run_command_with_tasks_path(self) -> None:
        """Test building harbor command uses -p for tasks_path configs."""
        config = make_config(
            dataset=None,
            tasks_path=Path("integration/skillbench/tasks"),
        )
        cmd = build_harbor_run_command(config, "test-job")

        assert "-p" in cmd
        assert "integration/skillbench/tasks" in cmd
        assert "--dataset" not in cmd

    def test_build_mode_args_skills_with_options(self, tmp_path: Path) -> None:
        """Test building mode args for skills with all options."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = make_config(
            skills=SkillsConfig(
                skills_dir=skills_dir,
                match_skill_to_task=True,
            ),
        )
        args = build_mode_args(config)

        assert "match_skill_to_task=True" in " ".join(args)

    def test_build_mode_args_skillflow_injection_eval_results(self) -> None:
        """Test building mode args for SkillFlow injection from eval results."""
        config = make_config(
            eval_results=Path("outputs/eval-results.json"),
            tasks_dir_for_skills=Path("integration/skillsbench/tasks"),
            corpus_dir=Path("data/skills"),
        )
        args = build_mode_args(config)

        joined = " ".join(args)
        assert "skillflow_injection_agent:SkillFlowInjectionAgent" in joined
        assert "eval_results=outputs/eval-results.json" in joined
        assert "tasks_dir=integration/skillsbench/tasks" in joined
        assert "corpus_dir=data/skills" in joined

    def test_build_mode_args_skillflow_injection_selector_cache(self) -> None:
        """Test building mode args for SkillFlow injection from selector cache."""
        config = make_config(
            selector_cache=Path("outputs/cache.json"),
            tasks_dir_for_skills=Path("integration/skillsbench/tasks"),
        )
        args = build_mode_args(config)

        joined = " ".join(args)
        assert "skillflow_injection_agent:SkillFlowInjectionAgent" in joined
        assert "selector_cache=outputs/cache.json" in joined
        assert "tasks_dir=integration/skillsbench/tasks" in joined

    def test_build_mode_args_skillflow_injection_no_tasks_dir(self) -> None:
        """Test injection mode works without tasks_dir_for_skills."""
        config = make_config(
            selector_cache=Path("outputs/batch/terminal-bench/selector_cache.json"),
            corpus_dir=Path("data/skills"),
        )
        args = build_mode_args(config)

        joined = " ".join(args)
        assert "SkillFlowInjectionAgent" in joined
        assert (
            "selector_cache=outputs/batch/terminal-bench/selector_cache.json" in joined
        )
        assert "corpus_dir=data/skills" in joined
        assert "tasks_dir=" not in joined

    def test_build_mode_args_skillflow_cached(self, tmp_path: Path) -> None:
        """Test building mode args for cached SkillFlow mode."""
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("{}")
        config = make_config(
            mcp_url="https://x.ngrok-free.dev/mcp",
            cached_skillflow=True,
            selector_cache=cache_file,
        )
        args = build_mode_args(config)

        joined = " ".join(args)
        assert "skillflow_mcp_cached_agent:SkillFlowMCPCachedAgent" in joined
        assert "mcp_url=https://x.ngrok-free.dev/mcp" in joined
        assert f"selector_cache={cache_file}" in joined

    def test_build_resume_command(self, tmp_path: Path) -> None:
        """Test building resume command."""
        job_path = tmp_path / "test-job"
        cmd = build_resume_command(job_path)

        assert "resume" in cmd
        assert str(job_path) in cmd

    def test_build_retry_errors_command(self, tmp_path: Path) -> None:
        """Test building retry errors command."""
        job_path = tmp_path / "test-job"
        cmd = build_retry_errors_command(job_path, ["CancelledError", "TimeoutError"])

        assert "resume" in cmd
        assert "--filter-error-type" in cmd
        assert "CancelledError" in cmd
        assert "TimeoutError" in cmd


class TestDisplayFunctions:
    """Tests for display functions."""

    def test_print_config_baseline(self) -> None:
        """Test printing config for baseline mode."""
        config = make_config()
        print_config(config, "test-job")

    def test_print_config_skills(self, tmp_path: Path) -> None:
        """Test printing config for skills mode."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = make_config(
            skills=SkillsConfig(
                skills_dir=skills_dir,
            ),
        )
        print_config(config, "test-job")

    def test_print_config_skills_with_matched(self, tmp_path: Path) -> None:
        """Test printing config for skills mode with matched skills."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = make_config(
            skills=SkillsConfig(
                skills_dir=skills_dir,
                match_skill_to_task=True,
            ),
        )
        print_config(config, "test-job")

    def test_print_config_with_daytona(self) -> None:
        """Test printing config with daytona environment."""
        config = make_config(
            environment=EnvironmentConfig(use_daytona=True, n_concurrent=6),
        )
        print_config(config, "test-job")

    def test_print_config_with_tasks(self) -> None:
        """Test printing config with task filter."""
        config = make_config(
            tasks=TaskConfig(
                include_tasks=["task1", "task2"],
                exclude_tasks=[],
            ),
        )
        print_config(config, "test-job")

    def test_print_multi_config_baseline(self) -> None:
        """Test printing multi-run config for baseline."""
        config = make_config()
        print_multi_config(config)

    def test_print_multi_config_skills(self, tmp_path: Path) -> None:
        """Test printing multi-run config for skills mode."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        config = make_config(
            skills=SkillsConfig(
                skills_dir=skills_dir,
            ),
            tasks=TaskConfig(
                include_tasks=["task1"],
                exclude_tasks=[],
            ),
        )
        print_multi_config(config)

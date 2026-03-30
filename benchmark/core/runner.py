"""Evaluation runner for Harbor benchmarks."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

from benchmark.core.commands import (
    build_harbor_run_command,
    build_resume_command,
    build_retry_errors_command,
)
from benchmark.core.config import EvalConfig, EvalMode
from benchmark.core.display import print_config, print_multi_config
from benchmark.core.paths import get_project_root
from benchmark.core.utils import (
    _get_benchmark_prefix,
    _get_short_model_name,
    fix_docker_file_ownership,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class EvaluationRunner:
    """Runner for Harbor benchmark evaluations."""

    def __init__(self, config: EvalConfig, config_path: Path | None = None) -> None:
        """Initialize the runner.

        Args:
            config: Evaluation configuration
            config_path: Path to config file (used for job name derivation)
        """
        self.config = config
        self.config_path = config_path
        self.project_root = get_project_root()

    def run(self) -> int:
        """Run a single evaluation.

        Returns:
            Exit code from harbor command
        """
        is_retry = (
            self.config.retry.retry_tasks
            or self.config.retry.retry_errors
            or self.config.retry.resume
        )

        if is_retry:
            if not self.config.job_name:
                print("Error: job_name is required for resume/retry")
                return 1
            job_name = self.config.job_name
        else:
            job_name = self._generate_new_job_name()

        print_config(self.config, job_name)

        if self.config.retry.retry_tasks:
            return self._run_retry_tasks(job_name)
        if self.config.retry.retry_errors:
            return self._run_retry_errors(job_name)
        if self.config.retry.resume:
            return self._run_resume(job_name)

        return self._run_new(job_name)

    def _generate_new_job_name(self) -> str:
        """Generate a job name for a new run (always includes timestamp)."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if self.config.job_name:
            return f"{self.config.job_name}-{timestamp}"
        prefix = self._generate_job_prefix()
        return f"{prefix}-{timestamp}"

    def run_multiple(self) -> int:
        """Run evaluation multiple times sequentially.

        Returns:
            Exit code (0 if all runs succeeded)
        """
        print("=" * 40)
        print(
            f"Running {self.config.mode.value} evaluation {self.config.num_runs} times"
        )
        print("=" * 40)
        print_multi_config(self.config)

        for i in range(1, self.config.num_runs + 1):
            print()
            print("=" * 40)
            print(f"Starting run {i} of {self.config.num_runs}")
            print("=" * 40)

            # Generate unique job name for each run
            job_name = self._generate_new_job_name()

            print(f"Job name: {job_name}")
            print()

            exit_code = self._run_new(job_name)

            if exit_code != 0:
                print(f"Run {i} failed with exit code {exit_code}")
                return exit_code

            print()
            print(f"Run {i} completed: {job_name}")

        print()
        print("=" * 40)
        print(f"All {self.config.num_runs} runs completed!")
        print("=" * 40)

        return 0

    def _generate_job_prefix(self) -> str:
        """Generate job name prefix based on config.

        Uses the config file stem (e.g. "skillsbench-inject" from
        "skillsbench-inject.json") when available, falling back to
        mode-based naming.
        """
        bp = _get_benchmark_prefix(self.config)

        if self.config_path is not None:
            config_stem = self.config_path.stem
            prefix = f"{bp}-{config_stem}"
        elif self.config.mode == EvalMode.BASELINE:
            prefix = f"{bp}-baseline"
        elif self.config.mode == EvalMode.MCP:
            prefix = f"{bp}-mcp"
        elif self.config.mode == EvalMode.SKILLFLOW_INJECTION:
            prefix = f"{bp}-skillflow-injection"
        else:
            # Skills mode
            assert self.config.skills
            skill_part = self.config.skills.skills_dir.name
            prefix = f"{bp}-{skill_part}"

            if self.config.skills.match_skill_to_task:
                prefix = f"{prefix}-matched"

        # Append model and reasoning effort
        short_model = _get_short_model_name(self.config.model)
        prefix = f"{prefix}-{short_model}"
        if self.config.reasoning_effort:
            prefix = f"{prefix}-{self.config.reasoning_effort}"

        # Add task suffix
        all_tasks = self.config.tasks.get_all_tasks()
        if len(all_tasks) == 1:
            prefix = f"{prefix}-{all_tasks[0]}"
        elif len(all_tasks) > 1:
            prefix = f"{prefix}-{len(all_tasks)}tasks"

        return prefix

    def _run_new(self, job_name: str) -> int:
        """Run a new evaluation job."""
        cmd = build_harbor_run_command(self.config, job_name)
        return self._execute_command(cmd)

    def _run_resume(self, job_name: str) -> int:
        """Resume an existing job."""
        job_path = self.config.jobs_dir / job_name
        if not job_path.exists():
            print(f"Error: Job directory not found at {job_path}")
            print("Cannot resume a job that doesn't exist. Run without --resume first.")
            return 1

        print(f"Resuming job from: {job_path}")

        if not self.config.environment.use_daytona:
            fix_docker_file_ownership(job_path)

        cmd = build_resume_command(job_path)
        return self._execute_command(cmd)

    def _run_retry_errors(self, job_name: str) -> int:
        """Retry transient errors in an existing job."""
        job_path = self.config.jobs_dir / job_name
        if not job_path.exists():
            print(f"Error: Job directory not found at {job_path}")
            print("Cannot retry errors for a job that doesn't exist.")
            return 1

        print(f"Retrying transient errors in: {job_path}")

        if not self.config.environment.use_daytona:
            fix_docker_file_ownership(job_path)

        error_types = " ".join(self.config.retry.retry_error_types)
        print(f"Retrying error types: {error_types}")

        cmd = build_retry_errors_command(job_path, self.config.retry.retry_error_types)
        return self._execute_command(cmd)

    def _run_retry_tasks(self, job_name: str) -> int:
        """Retry specific tasks by deleting their trial directories."""
        job_path = self.config.jobs_dir / job_name
        if not job_path.exists():
            print(f"Error: Job directory not found at {job_path}")
            print("Cannot retry tasks for a job that doesn't exist.")
            return 1

        print(f"Retrying specific tasks in: {job_path}")

        if not self.config.environment.use_daytona:
            fix_docker_file_ownership(job_path)

        # Delete trial directories for specified tasks
        for raw_task in self.config.retry.retry_tasks:
            task = raw_task.strip()
            for trial_dir in job_path.glob(f"{task}__*"):
                if trial_dir.is_dir():
                    print(f"  Removing trial: {trial_dir.name}")
                    shutil.rmtree(trial_dir)

        print("Resuming job to re-run deleted trials...")
        cmd = build_resume_command(job_path)
        return self._execute_command(cmd)

    def _execute_command(self, cmd: Sequence[str]) -> int:
        """Execute a command and return exit code."""
        result = subprocess.run(cmd, cwd=self.project_root, check=False)
        return result.returncode

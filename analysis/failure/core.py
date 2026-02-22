"""Core utilities for failure analysis.

This module contains data models, loaders, and formatters for analyzing
agent evaluation trajectories.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskData:
    """Data extracted from a single task's evaluation results."""

    task_name: str
    trial_name: str
    reward: float
    n_input_tokens: int
    n_output_tokens: int
    n_steps: int
    trajectory_steps: list[dict]
    test_results: dict
    task_prompt: str


def load_task_data(task_dir: Path) -> TaskData | None:
    """Load all relevant data for a single task.

    Args:
        task_dir: Path to task directory (e.g., feal-linear-cryptanalysis__94NPxfi)

    Returns:
        TaskData object or None if required files are missing
    """
    result_file = task_dir / "result.json"
    trajectory_file = task_dir / "agent" / "trajectory.json"
    ctrf_file = task_dir / "verifier" / "ctrf.json"

    if not result_file.exists():
        return None

    with result_file.open() as f:
        result = json.load(f)

    trajectory_steps: list[dict] = []
    task_prompt = ""
    if trajectory_file.exists():
        with trajectory_file.open() as f:
            trajectory = json.load(f)
            trajectory_steps = trajectory.get("steps", [])
            for step in trajectory_steps:
                if step.get("source") == "user" and step.get("step_id", 0) >= 3:
                    msg = step.get("message", "")
                    if not msg.startswith("<") and not msg.startswith("#"):
                        task_prompt = msg
                        break

    test_results: dict = {}
    if ctrf_file.exists():
        with ctrf_file.open() as f:
            test_results = json.load(f)

    agent_result = result.get("agent_result", {})
    verifier_result = result.get("verifier_result", {})

    return TaskData(
        task_name=result.get("task_name", task_dir.name.split("__")[0]),
        trial_name=result.get("trial_name", task_dir.name),
        reward=verifier_result.get("rewards", {}).get("reward", 0.0),
        n_input_tokens=agent_result.get("n_input_tokens", 0),
        n_output_tokens=agent_result.get("n_output_tokens", 0),
        n_steps=len([s for s in trajectory_steps if s.get("source") == "agent"]),
        trajectory_steps=trajectory_steps,
        test_results=test_results,
        task_prompt=task_prompt,
    )


def list_task_dirs(job_dir: Path) -> list[Path]:
    """List all task directories in a job directory.

    Args:
        job_dir: Path to job directory

    Returns:
        List of task directory paths
    """
    return [
        d
        for d in job_dir.iterdir()
        if d.is_dir() and "__" in d.name and (d / "result.json").exists()
    ]


def compress_trajectory(steps: list[dict], max_steps: int = 50) -> str:
    """Compress trajectory into a readable summary.

    Args:
        steps: List of trajectory steps
        max_steps: Maximum number of steps to include

    Returns:
        Compressed string representation of the trajectory
    """
    lines = []
    step_count = 0

    for step in steps:
        source = step.get("source", "")
        if source == "agent":
            step_count += 1
            if step_count > max_steps:
                lines.append(f"... ({len(steps) - max_steps} more steps)")
                break

            message = step.get("message", "")[:500]
            tool_calls = step.get("tool_calls", [])
            observation = step.get("observation", {})

            lines.append(f"[Step {step_count}] {message}")

            for tc in tool_calls:
                fn_name = tc.get("function_name", "unknown")
                args = tc.get("arguments", {})
                if fn_name == "shell":
                    cmd = args.get("command", [])
                    if isinstance(cmd, list) and len(cmd) >= 3:
                        lines.append(f"  -> shell: {cmd[-1][:200]}")
                    else:
                        lines.append(f"  -> shell: {cmd}")
                else:
                    lines.append(f"  -> {fn_name}")

            if observation and observation.get("results"):
                for result in observation["results"][:1]:
                    content = result.get("content", "")[:300]
                    if content:
                        lines.append(f"  <- {content}...")

    return "\n".join(lines)


def format_test_results(test_results: dict) -> str:
    """Format CTRF test results into a readable summary.

    Args:
        test_results: CTRF JSON data

    Returns:
        Formatted string with test summary and failures
    """
    results = test_results.get("results", {})
    summary = results.get("summary", {})
    tests = results.get("tests", [])

    lines = [
        f"Tests: {summary.get('tests', 0)} total, "
        f"{summary.get('passed', 0)} passed, "
        f"{summary.get('failed', 0)} failed"
    ]

    for test in tests:
        if test.get("status") == "failed":
            name = test.get("name", "unknown")
            trace = test.get("trace", "")[:1500]
            lines.append(f"\nFailed: {name}")
            lines.append(f"Trace:\n{trace}")

    return "\n".join(lines)


def extract_one_liner(summary: str) -> str:
    """Extract a one-line description from the summary.

    Args:
        summary: Full summary text

    Returns:
        One-line description
    """
    for line in summary.split("\n"):
        if line.strip() and not line.startswith("#") and not line.startswith("*"):
            return line.strip()[:100]
    return "No description available"


def extract_amenability(summary: str) -> str:
    """Extract skill amenability level from summary.

    Args:
        summary: Full summary text

    Returns:
        HIGH, MEDIUM, or LOW
    """
    lower = summary.lower()
    if "### skill amenability" in lower:
        idx = lower.find("### skill amenability")
        section = summary[idx : idx + 200]
        if "HIGH" in section.upper():
            return "HIGH"
        if "MEDIUM" in section.upper():
            return "MEDIUM"
    return "LOW"

"""Configuration display utilities for evaluation runs."""

from __future__ import annotations

from benchmark.core.config import EvalConfig, EvalMode


def print_config(config: EvalConfig, job_name: str) -> None:
    """Print configuration summary for a single run.

    Args:
        config: Evaluation configuration
        job_name: Name of the evaluation job
    """
    print()
    print(f"Job name: {job_name}")
    print(f"Job path: {config.jobs_dir}")
    print(f"Benchmark: {config.benchmark_source}")
    print(f"Model: {config.model}")
    print(f"Reasoning effort: {config.reasoning_effort or '(none)'}")
    env_name = "Daytona" if config.environment.use_daytona else "Docker"
    print(f"Environment: {env_name}")
    print(f"Concurrency: {config.environment.n_concurrent}")
    all_tasks = config.tasks.get_all_tasks()
    if all_tasks:
        print(f"Task filter: {len(all_tasks)} specific task(s)")

    print_mode_config(config)
    print()


def print_multi_config(config: EvalConfig) -> None:
    """Print multi-run configuration summary.

    Args:
        config: Evaluation configuration
    """
    print(f"Model: {config.model}")
    print(f"Reasoning effort: {config.reasoning_effort or '(none)'}")
    print(f"Concurrency: {config.environment.n_concurrent}")
    if config.mode == EvalMode.SKILLS and config.skills:
        print(f"Skills dir: {config.skills.skills_dir}")
        if config.skills.skill_list_name:
            print(f"Skillset: {config.skills.skill_list_name}")
        else:
            print("Skillset: (none - using all skills in directory)")

    all_tasks = config.tasks.get_all_tasks()
    if all_tasks:
        print(f"Tasks: {len(all_tasks)} task(s)")


def print_mode_config(config: EvalConfig) -> None:
    """Print mode-specific configuration details.

    Args:
        config: Evaluation configuration
    """
    if config.mode == EvalMode.BASELINE:
        print("Mode: BASELINE (no skills)")
    elif config.mode == EvalMode.MCP:
        print("Mode: MCP (tool description validation)")
        print(f"MCP URL: {config.mcp_url}")
    elif config.mode == EvalMode.SKILLFLOW:
        print("Mode: SKILLFLOW (dynamic skill discovery)")
        if config.skills:
            print(f"SkillFlow peer: {config.skills.skillflow_peer_url}")
    elif config.mode == EvalMode.SKILLFLOW_EVAL:
        print("Mode: SKILLFLOW_EVAL (retrieved skills injection + MCP)")
        print(f"Eval results: {config.eval_results}")
        print(f"Tasks dir for skills: {config.tasks_dir_for_skills}")
        if config.mcp_url:
            print(f"MCP URL: {config.mcp_url}")
    else:
        _print_skills_mode_config(config)


def _print_skills_mode_config(config: EvalConfig) -> None:
    """Print skills mode configuration details."""
    assert config.skills
    if config.skills.match_skill_to_task:
        print("Mode: MATCHED SKILLS (1:1 task-to-skill mapping)")
    elif config.skills.skill_list_name:
        print("Mode: CURATED SKILLS (filtered by skill list)")
    else:
        print("Mode: ALL SKILLS (using all skills in directory)")
    print(f"Skills dir: {config.skills.skills_dir}")
    if config.skills.skill_list_name:
        skills_list = config.get_skills_list_file()
        print(f"Skills list: {skills_list}")
    else:
        print("Skills list: (none - using all skills)")

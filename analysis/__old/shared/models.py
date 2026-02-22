"""Shared data models for analysis scripts."""

from dataclasses import dataclass


@dataclass
class TaskMetrics:
    """Metrics for a single task."""

    task_name: str
    reward: float
    input_tokens: int
    output_tokens: int
    cache_tokens: int
    execution_time_sec: float  # Agent execution time only
    n_steps: int  # Number of tool/function calls


@dataclass
class SkillReadRecord:
    """Record of a verified skill read with content."""

    skill_name: str
    command: str
    line_num: int
    content_preview: str  # First N chars of skill content
    task_name: str = ""  # Set after creation

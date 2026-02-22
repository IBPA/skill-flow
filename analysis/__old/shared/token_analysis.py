"""Token usage analysis for skill-related content using tiktoken."""

import json
import re
from functools import lru_cache
from pathlib import Path

import tiktoken

from .skill_detection import (
    _contains_skill_content,
    _extract_skill_content,
    _extract_skill_name_from_cmd,
    _is_skill_read_command,
    find_rollout_files,
)


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    """Get cached tiktoken encoder."""
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def analyze_skill_tokens(rollout_path: Path) -> dict:
    """Analyze token usage for skill-related content in a rollout file.

    Returns:
        - task_name: str
        - metadata_tokens: int - tokens in initial skill listing
        - content_tokens: int - tokens from reading SKILL.md files
        - skills_read: list of (skill_name, tokens)
    """
    # Extract task name from path
    task_name = _extract_task_name(rollout_path)

    lines = _load_jsonl(rollout_path)
    if not lines:
        return _empty_result(task_name)

    metadata_tokens = _count_metadata_tokens(lines)
    content_tokens, skills_read = _count_content_tokens(lines)

    return {
        "task_name": task_name,
        "metadata_tokens": metadata_tokens,
        "content_tokens": content_tokens,
        "skills_read": skills_read,
    }


def _extract_task_name(rollout_path: Path) -> str:
    """Extract task name from rollout file path."""
    parts = rollout_path.parts
    for i, part in enumerate(parts):
        if part == "agent" and i > 0:
            return parts[i - 1]
    return "unknown"


def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, skipping malformed lines."""
    lines = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return lines


def _empty_result(task_name: str) -> dict:
    """Return empty result structure."""
    return {
        "task_name": task_name,
        "metadata_tokens": 0,
        "content_tokens": 0,
        "skills_read": [],
    }


def _count_metadata_tokens(lines: list[dict]) -> int:
    """Count tokens in skill metadata from session initialization."""
    for line in lines[:5]:
        if line.get("type") == "session_meta":
            meta_str = json.dumps(line)
            if "SKILL.md" in meta_str or "/skills/" in meta_str:
                return _extract_skill_listing_tokens(meta_str)
    return 0


def _extract_skill_listing_tokens(meta_str: str) -> int:
    """Extract and count tokens for skill listing in metadata."""
    # Match skill-related content patterns
    skill_refs = re.findall(
        r"(?:Available skills[^}]*|"
        r"\(file: /logs/agent/skills/[^)]+\)|"
        r"\$CODEX_HOME/skills/[^\s]+|"
        r"/skills/[a-zA-Z0-9_-]+/SKILL\.md[^}]*)",
        meta_str,
        re.IGNORECASE,
    )
    return count_tokens("".join(skill_refs)) if skill_refs else 0


def _count_content_tokens(lines: list[dict]) -> tuple[int, list[tuple[str, int]]]:
    """Count tokens from skill content reads."""
    content_tokens = 0
    skills_read: list[tuple[str, int]] = []
    pending: dict[str, str] = {}  # call_id -> skill_name

    for line in lines:
        if line.get("type") == "session_meta":
            continue

        payload = line.get("payload", {})

        # Track shell commands that read skills
        if _is_shell_call(payload):
            cmd = _extract_command(payload)
            if _is_skill_read_command(cmd):
                skill_name = _extract_skill_name_from_cmd(cmd)
                call_id = payload.get("call_id", "")
                if skill_name and call_id:
                    pending[call_id] = skill_name

        # Check function output for skill content
        if payload.get("type") == "function_call_output":
            call_id = payload.get("call_id", "")
            if call_id in pending:
                output = payload.get("output", "")
                if _contains_skill_content(output):
                    content = _extract_skill_content(output)
                    tokens = count_tokens(content)
                    content_tokens += tokens
                    skills_read.append((pending[call_id], tokens))
                del pending[call_id]

        # Also check command_execution items
        item = line.get("item", {})
        if item.get("type") == "command_execution":
            cmd = item.get("command", "")
            if _is_skill_read_command(cmd):
                skill_name = _extract_skill_name_from_cmd(cmd)
                output = item.get("aggregated_output", "")
                if skill_name and _contains_skill_content(output):
                    content = _extract_skill_content(output)
                    tokens = count_tokens(content)
                    content_tokens += tokens
                    skills_read.append((skill_name, tokens))

    return content_tokens, skills_read


def _is_shell_call(payload: dict) -> bool:
    """Check if payload is a shell function call."""
    return payload.get("type") == "function_call" and payload.get("name") == "shell"


def _extract_command(payload: dict) -> str:
    """Extract command string from shell payload."""
    args = payload.get("arguments", "")
    if isinstance(args, str):
        try:
            args_obj = json.loads(args)
            return " ".join(args_obj.get("command", []))
        except (json.JSONDecodeError, TypeError):
            return args
    return str(args)


def analyze_skill_tokens_aggregate(eval_dirs: list[Path]) -> dict:
    """Analyze skill token usage across multiple evaluation runs.

    Returns:
        - total_metadata_tokens: int
        - total_content_tokens: int
        - per_task: dict of task_name -> {avg_metadata, avg_content, unique_skills}
        - avg_metadata_per_task: float
        - avg_content_per_task: float
    """
    per_task: dict[str, dict] = {}
    total_metadata = 0
    total_content = 0
    n_tasks = 0

    for eval_dir in eval_dirs:
        for rollout_path in find_rollout_files(eval_dir):
            result = analyze_skill_tokens(rollout_path)
            task_name = result["task_name"].split("__")[0]  # Remove hash suffix

            if task_name not in per_task:
                per_task[task_name] = {
                    "metadata_tokens": [],
                    "content_tokens": [],
                    "skills_read": [],
                }

            per_task[task_name]["metadata_tokens"].append(result["metadata_tokens"])
            per_task[task_name]["content_tokens"].append(result["content_tokens"])
            per_task[task_name]["skills_read"].extend(result["skills_read"])

            total_metadata += result["metadata_tokens"]
            total_content += result["content_tokens"]
            n_tasks += 1

    # Compute per-task summaries
    task_summary = {}
    for task_name, data in per_task.items():
        n = len(data["metadata_tokens"])
        task_summary[task_name] = {
            "avg_metadata_tokens": sum(data["metadata_tokens"]) / n if n else 0,
            "avg_content_tokens": sum(data["content_tokens"]) / n if n else 0,
            "total_skills_read": len(data["skills_read"]),
            "unique_skills": list({s[0] for s in data["skills_read"]}),
        }

    return {
        "total_metadata_tokens": total_metadata,
        "total_content_tokens": total_content,
        "n_tasks_analyzed": n_tasks,
        "avg_metadata_per_task": total_metadata / n_tasks if n_tasks else 0,
        "avg_content_per_task": total_content / n_tasks if n_tasks else 0,
        "per_task": task_summary,
    }

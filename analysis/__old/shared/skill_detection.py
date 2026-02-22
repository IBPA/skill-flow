"""Skill detection and analysis utilities."""

import json
import re
from collections import defaultdict
from pathlib import Path

from .models import SkillReadRecord


def find_rollout_files(evaluation_dir: Path) -> list[Path]:
    """Find all rollout JSONL files in the evaluation directory."""
    # Try multiple patterns to handle different directory structures
    patterns = [
        "*/agent/sessions/*/*/*rollout*.jsonl",  # task_id/agent/sessions/YYYY/MM/DD/
        "*/agent/sessions/*/*rollout*.jsonl",  # task_id/agent/sessions/YYYY/MM/
        "**/rollout*.jsonl",  # fallback: any rollout file
    ]
    for pattern in patterns:
        files = list(evaluation_dir.glob(pattern))
        if files:
            return files
    return []


def extract_skill_names_from_instructions(content: str) -> set[str]:
    """Extract skill names from the initial instructions."""
    skills = set()

    # Match patterns like: (file: /logs/agent/skills/author__skill_name/SKILL.md)
    pattern1 = r"/logs/agent/skills/([^/]+)/SKILL\.md"
    matches = re.findall(pattern1, content)
    skills.update(matches)

    # Match patterns like: $CODEX_HOME/skills/task-name/SKILL.md
    pattern2 = r"\$CODEX_HOME/skills/([^/]+)/SKILL\.md"
    matches = re.findall(pattern2, content)
    skills.update(matches)

    return skills


def _contains_skill_content(output: str) -> bool:
    """Check if output contains actual skill file content.

    Valid skill content starts with either:
    1. YAML frontmatter: ---\nname: xxx
    2. Markdown header: # Skill Name
    """
    if not output:
        return False

    # Handle JSON-wrapped output (from function_call_output)
    if output.startswith('{"output":'):
        try:
            parsed = json.loads(output)
            output = parsed.get("output", "")
        except json.JSONDecodeError:
            pass

    # Check for YAML frontmatter
    if "---" in output[:50] and "name:" in output[:200]:
        return True

    # Check for markdown header (skill title)
    return bool(output.lstrip().startswith("# "))


def _is_skill_read_command(cmd: str) -> bool:
    """Check if a command attempts to read a SKILL.md file."""
    if not cmd or "SKILL.md" not in cmd:
        return False

    # Check for read-like commands (not write commands)
    write_commands = ["mkdir", "rm ", "touch ", "mv ", "cp ", "echo "]
    return not any(wc in cmd for wc in write_commands)


def _extract_skill_name_from_cmd(cmd: str) -> str | None:
    """Extract skill name from a command that reads a skill file."""
    # Match patterns like:
    # - /logs/agent/skills/author__skill_name/SKILL.md (skillsmp format)
    # - $CODEX_HOME/skills/task-name/SKILL.md (letta format)
    # - /skills/task-name/SKILL.md (generic)

    # First try author__skill_name format (with double underscore)
    pattern1 = r"/(?:logs/agent/)?skills/([a-zA-Z0-9_-]+__[a-zA-Z0-9_-]+)"
    match = re.search(pattern1, cmd)
    if match:
        return match.group(1)

    # Then try task-name format (Letta-style, no double underscore)
    # Match /skills/name/ or $CODEX_HOME/skills/name/
    pattern2 = r"(?:\$CODEX_HOME)?/skills/([a-zA-Z0-9_-]+)/SKILL\.md"
    match = re.search(pattern2, cmd)
    if match:
        return match.group(1)

    return None


def _extract_skill_content(output: str) -> str:
    """Extract actual skill content from command output."""
    if not output:
        return ""

    # Handle JSON-wrapped output (from function_call_output)
    if output.startswith('{"output":'):
        try:
            parsed = json.loads(output)
            output = parsed.get("output", "")
        except json.JSONDecodeError:
            pass

    return output


def _process_shell_call(
    payload: dict, line_num: int, pending_reads: dict[str, tuple[str, str, int]]
) -> None:
    """Process a shell function_call and track pending skill reads."""
    args = payload.get("arguments", "")
    if isinstance(args, str):
        try:
            args_obj = json.loads(args)
            cmd = " ".join(args_obj.get("command", []))
        except (json.JSONDecodeError, TypeError):
            cmd = args
    else:
        cmd = str(args)

    if _is_skill_read_command(cmd):
        skill_name = _extract_skill_name_from_cmd(cmd)
        call_id = payload.get("call_id", "")
        if skill_name and call_id:
            pending_reads[call_id] = (skill_name, cmd, line_num)


def _process_call_output(
    payload: dict,
    line_num: int,
    pending_reads: dict[str, tuple[str, str, int]],
    skill_reads: dict[str, list[str]],
    skill_records: list[SkillReadRecord],
    content_limit: int,
) -> None:
    """Process function_call_output and verify skill content."""
    call_id = payload.get("call_id", "")
    if call_id not in pending_reads:
        return

    output = payload.get("output", "")
    skill_name, cmd, cmd_line = pending_reads[call_id]

    if _contains_skill_content(output):
        msg = f"verified (cmd L{cmd_line}, output L{line_num})"
        skill_reads[skill_name].append(msg)
        content = _extract_skill_content(output)
        skill_records.append(
            SkillReadRecord(
                skill_name=skill_name,
                command=cmd[:200],
                line_num=line_num,
                content_preview=content[:content_limit],
            )
        )

    del pending_reads[call_id]


def _process_command_execution(
    item: dict,
    line_num: int,
    skill_reads: dict[str, list[str]],
    skill_records: list[SkillReadRecord],
    content_limit: int,
) -> None:
    """Process command_execution items (alternative format)."""
    cmd = item.get("command", "")
    if not _is_skill_read_command(cmd):
        return

    skill_name = _extract_skill_name_from_cmd(cmd)
    output = item.get("aggregated_output", "")

    if skill_name and _contains_skill_content(output):
        skill_reads[skill_name].append(f"verified (line {line_num})")
        content = _extract_skill_content(output)
        skill_records.append(
            SkillReadRecord(
                skill_name=skill_name,
                command=cmd[:200],
                line_num=line_num,
                content_preview=content[:content_limit],
            )
        )


def detect_skill_reads(
    lines: list[dict], content_limit: int = 500
) -> tuple[dict[str, list[str]], list[SkillReadRecord]]:
    """
    Detect when skills are actually read/accessed during execution.

    Only counts VERIFIED reads where:
    1. Agent issued a command to read a skill file (cat, sed, head, etc.)
    2. The command output contains actual skill content (YAML frontmatter or markdown)

    Args:
        lines: Parsed JSONL lines from rollout file.
        content_limit: Max characters to capture in content preview.

    Returns:
        Tuple of (skill_reads dict, list of SkillReadRecord with content).
    """
    skill_reads: dict[str, list[str]] = defaultdict(list)
    skill_records: list[SkillReadRecord] = []
    pending_skill_reads: dict[str, tuple[str, str, int]] = {}

    for i, line in enumerate(lines):
        line_num = i + 1

        if line.get("type") == "session_meta":
            continue

        payload = line.get("payload", {})
        payload_type = payload.get("type")

        if payload_type == "function_call" and payload.get("name") == "shell":
            _process_shell_call(payload, line_num, pending_skill_reads)
        elif payload_type == "function_call_output":
            _process_call_output(
                payload,
                line_num,
                pending_skill_reads,
                skill_reads,
                skill_records,
                content_limit,
            )

        item = line.get("item", {})
        if item.get("type") == "command_execution":
            _process_command_execution(
                item, line_num, skill_reads, skill_records, content_limit
            )

    return dict(skill_reads), skill_records


def analyze_rollout(
    rollout_path: Path, content_limit: int = 500
) -> tuple[dict, list[SkillReadRecord]]:
    """Analyze a single rollout file for skill usage.

    Returns:
        Tuple of (result dict, list of SkillReadRecord with content).
    """
    # Extract task name from path: .../task_name/agent/sessions/YYYY/MM/DD/rollout.jsonl
    # Find "agent" in path and take the part before it
    parts = rollout_path.parts
    task_name = "unknown"
    for i, part in enumerate(parts):
        if part == "agent" and i > 0:
            task_name = parts[i - 1]
            break

    lines = []
    with rollout_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not lines:
        return {"task": task_name, "error": "Empty rollout file"}, []

    # Get skills from initial instructions (first 2 lines)
    initial_content = json.dumps(lines[:2])
    available_skills = extract_skill_names_from_instructions(initial_content)

    # Detect skill reads in the rest of the rollout
    skill_reads, skill_records = detect_skill_reads(lines, content_limit)

    # Add task name to each record for context
    for record in skill_records:
        record.task_name = task_name

    return {
        "task": task_name,
        "rollout_file": str(rollout_path),
        "total_lines": len(lines),
        "available_skills": sorted(available_skills),
        "skills_used": skill_reads,
        "num_available": len(available_skills),
        "num_used": len(skill_reads),
    }, skill_records

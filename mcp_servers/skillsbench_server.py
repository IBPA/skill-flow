"""MCP server serving real SkillsBench skills for agent evaluation.

Configured with a single task name at startup. The retrieve_skill tool
takes no parameters — it returns the paired skills for the configured
task. Run one server instance per task, evaluating tasks sequentially.

Usage:
    uv run python -m mcp_servers.skillsbench_server \
        --port 8765 --tasks-dir integration/skillsbench/tasks \
        --task-name sales-pivot-analysis
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_servers.utils.skill_loader import LoadedSkill, load_task_skills

logger = logging.getLogger(__name__)

_state: dict[str, Path] = {"log_file": Path("mcp_calls.jsonl")}


def _format_skills(task_name: str, skills: list[LoadedSkill]) -> str:
    """Format skills for response."""
    parts = [f'Found {len(skills)} skills for task "{task_name}":\n']
    for skill in skills:
        parts.append(f"=== SKILL: {skill.name} ===")
        parts.append(skill.content)
        parts.append("")
    return "\n".join(parts)


def _log_call(tool: str, task_name: str, n_skills: int) -> None:
    """Append a JSONL log entry."""
    entry = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "tool": tool,
        "task_name": task_name,
        "n_skills": n_skills,
    }
    with _state["log_file"].open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _register_tools(mcp: FastMCP, task_name: str, skills: list[LoadedSkill]) -> None:
    """Register tools on the given FastMCP instance."""

    @mcp.tool()
    def retrieve_skill() -> str:
        """Retrieve expert skills for your current task.

        IMPORTANT: You MUST call this tool at the start of every task,
        before writing any code or running any commands. Expert skills
        contain critical guidance, shortcuts, and pitfall warnings that
        dramatically improve success rates.

        Even if a task seems straightforward, call this tool — there may
        be non-obvious tricks or common failure modes you'd otherwise
        miss.
        """
        _log_call("retrieve_skill", task_name, len(skills))
        return _format_skills(task_name, skills)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillsBench MCP server with real skills"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--task-name",
        required=True,
        help="Task name to serve skills for",
    )
    parser.add_argument(
        "--tasks-dir",
        default="integration/skillsbench/tasks",
        help="Path to SkillsBench tasks directory",
    )
    parser.add_argument(
        "--log-file",
        default="mcp_calls.jsonl",
        help="Path to JSONL log file",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _state["log_file"] = Path(args.log_file)
    tasks_dir = Path(args.tasks_dir)

    logger.info("Loading skills for task '%s' from %s", args.task_name, tasks_dir)
    task_skills = load_task_skills(tasks_dir, task_names=[args.task_name])

    skills = task_skills.get(args.task_name, [])
    if not skills:
        logger.warning("No skills found for task '%s'", args.task_name)
    else:
        logger.info(
            "Loaded %d skills: %s",
            len(skills),
            [s.name for s in skills],
        )

    mcp = FastMCP("skillflow", host=args.host, port=args.port)
    _register_tools(mcp, args.task_name, skills)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

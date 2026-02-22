"""MCP server serving SkillFlow-retrieved skill *paths* for evaluation.

Unlike skillsbench_server.py (which returns full SKILL.md content for
ground-truth skills), this server returns the container paths where
skills have been injected by SkillFlowEvalAgent. The agent calls
``retrieve_skill()`` to discover which skills are available and where
to find them.

Usage:
    uv run python -m mcp_servers.skillflow_eval_server \
        --port 8765 --task-name sales-pivot-analysis \
        --eval-results outputs/eval-selector-results.json \
        --tasks-dir integration/skillsbench/tasks
"""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_servers.skill_loader import ResolvedSkill, resolve_eval_skill_folders

logger = logging.getLogger(__name__)

CONTAINER_SKILLS_DIR = "/logs/agent/skills"

_state: dict[str, Path | str] = {
    "log_file": Path("mcp_calls.jsonl"),
    "eval_results": "",
}


def _format_skill_paths(skills: list[ResolvedSkill]) -> str:
    """Format resolved skills as container path references."""
    if not skills:
        return "No skills available for this task."

    lines = ["Use the following skills:"]
    for skill in skills:
        path = f"{CONTAINER_SKILLS_DIR}/{skill.name}"
        desc = f" — {skill.description}" if skill.description else ""
        lines.append(f"- {path}{desc}")
    return "\n".join(lines)


def _log_call(tool: str, task_name: str, n_skills: int) -> None:
    """Append a JSONL log entry."""
    entry = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "tool": tool,
        "task_name": task_name,
        "n_skills": n_skills,
        "eval_results": str(_state["eval_results"]),
    }
    log_file = _state["log_file"]
    assert isinstance(log_file, Path)
    with log_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _register_tools(mcp: FastMCP, task_name: str, skills: list[ResolvedSkill]) -> None:
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
        return _format_skill_paths(skills)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SkillFlow eval MCP server (returns skill paths)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--task-name",
        required=True,
        help="Task name to serve skills for",
    )
    parser.add_argument(
        "--eval-results",
        required=True,
        help="Path to eval results JSON (e.g. outputs/eval-selector-results.json)",
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
    _state["eval_results"] = args.eval_results

    eval_results_path = Path(args.eval_results)
    tasks_dir = Path(args.tasks_dir)

    logger.info(
        "Resolving skills for task '%s' from %s",
        args.task_name,
        eval_results_path,
    )
    skills = resolve_eval_skill_folders(eval_results_path, tasks_dir, args.task_name)

    if not skills:
        logger.warning("No skills resolved for task '%s'", args.task_name)
    else:
        logger.info(
            "Resolved %d skills: %s",
            len(skills),
            [s.name for s in skills],
        )

    mcp = FastMCP("skillflow", host=args.host, port=args.port)
    _register_tools(mcp, args.task_name, skills)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

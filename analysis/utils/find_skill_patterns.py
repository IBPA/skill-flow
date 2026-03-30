"""Scan trajectories for SKILL.md references to understand loading patterns."""

import json
import re
import sys
from pathlib import Path


def extract_tool_call_texts(steps: list[dict]) -> list[tuple[int, str, str]]:
    """Extract all tool call argument strings from trajectory steps.

    Returns list of (step_id, function_name, raw_text).
    """
    results = []
    for step in steps:
        step_id = step.get("step_id", 0)
        tool_calls = step.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function_name", "")
            args = tc.get("arguments", {})
            # Flatten all argument values into text
            for _key, val in args.items():
                if isinstance(val, str):
                    results.append((step_id, fn, val))
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            results.append((step_id, fn, item))
            # Also check raw_arguments in extra
        extra = step.get("extra", {})
        raw = extra.get("raw_arguments", "")
        if raw:
            results.append((step_id, "raw", raw))
    return results


def extract_observation_texts(steps: list[dict]) -> list[tuple[int, str]]:
    """Extract observation/result texts from steps.

    Returns list of (step_id, text).
    """
    results = []
    for step in steps:
        step_id = step.get("step_id", 0)
        obs = step.get("observation", {})
        if isinstance(obs, dict):
            for r in obs.get("results", []):
                content = r.get("content", "")
                if content:
                    results.append((step_id, content))
    return results


def scan_trajectory(traj_path: Path) -> list[dict]:
    """Scan a single trajectory for SKILL.md patterns."""
    with traj_path.open() as f:
        data = json.load(f)

    steps = data.get("steps", [])
    findings = []

    # Search tool call arguments
    for step_id, fn, text in extract_tool_call_texts(steps):
        # Find SKILL.md references (excluding the instructions listing)
        for match in re.finditer(r"SKILL\.md", text):
            # Get surrounding context (100 chars before and after)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            # Skip if this is just the available skills listing
            if "### Available skills" in text and len(text) > 1000:
                continue
            findings.append(
                {
                    "step_id": step_id,
                    "source": "tool_call",
                    "function": fn,
                    "context": context.replace("\n", "\\n"),
                }
            )

    # Search observations (agent output/results)
    for step_id, text in extract_observation_texts(steps):
        for match in re.finditer(r"SKILL\.md", text):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end]
            findings.append(
                {
                    "step_id": step_id,
                    "source": "observation",
                    "function": "",
                    "context": context.replace("\n", "\\n"),
                }
            )

    # Also search agent messages
    for step in steps:
        step_id = step.get("step_id", 0)
        msg = step.get("message", "")
        if isinstance(msg, str) and "SKILL.md" in msg:
            # Skip the AGENTS.md instruction messages
            if "### Available skills" in msg or "AGENTS.md instructions" in msg:
                continue
            for match in re.finditer(r"SKILL\.md", msg):
                start = max(0, match.start() - 100)
                end = min(len(msg), match.end() + 100)
                context = msg[start:end]
                findings.append(
                    {
                        "step_id": step_id,
                        "source": "agent_message",
                        "function": "",
                        "context": context.replace("\n", "\\n"),
                    }
                )

    return findings


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python analysis/find_skill_patterns.py <job_dir> [--verbose]")
        example = "outputs/evaluation/sk-baseline-20260212-140626"
        print(f"Example: python analysis/find_skill_patterns.py {example}")
        sys.exit(1)

    job_dir = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv

    if not job_dir.exists():
        print(f"Directory not found: {job_dir}")
        sys.exit(1)

    traj_files = sorted(job_dir.glob("*/agent/trajectory.json"))
    print(f"Found {len(traj_files)} trajectories in {job_dir.name}\n")

    total_with_refs = 0
    all_patterns: dict[str, int] = {}

    for traj_path in traj_files:
        task_dir = traj_path.parent.parent.name
        task_name = task_dir.split("__")[0]
        findings = scan_trajectory(traj_path)

        if findings:
            total_with_refs += 1
            print(f"{'=' * 80}")
            print(f"Task: {task_name} ({len(findings)} SKILL.md references)")
            print(f"{'=' * 80}")
            for f in findings:
                source_label = f"[step {f['step_id']}] {f['source']}"
                if f["function"]:
                    source_label += f" ({f['function']})"
                print(f"  {source_label}")
                if verbose:
                    print(f"    Context: ...{f['context']}...")
                print()

                # Extract skill path patterns
                for m in re.finditer(
                    r"(/[^\s\"']+/skills/[^\s\"'/]+/SKILL\.md)", f["context"]
                ):
                    pattern = m.group(1)
                    # Normalize: replace specific skill name with <name>
                    normalized = re.sub(
                        r"/skills/([^/]+)/SKILL\.md",
                        "/skills/<name>/SKILL.md",
                        pattern,
                    )
                    all_patterns[normalized] = all_patterns.get(normalized, 0) + 1

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total trajectories: {len(traj_files)}")
    print(f"Trajectories with SKILL.md references: {total_with_refs}")
    print(
        f"Load rate: {total_with_refs}/{len(traj_files)} "
        f"({100 * total_with_refs / max(1, len(traj_files)):.1f}%)"
    )

    if all_patterns:
        print("\nPath patterns found:")
        for pattern, count in sorted(all_patterns.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x  {pattern}")


if __name__ == "__main__":
    main()

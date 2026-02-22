#!/usr/bin/env python3
"""Convert agent rollout JSONL files to human-readable text format.

This script takes a task directory (or JSONL file) and outputs a formatted
text file showing the conversation flow between user, assistant, and tools.

Usage:
    poetry run python analysis/rollout_to_text.py <task_dir_or_jsonl>
    poetry run python analysis/rollout_to_text.py \
        outputs/evaluation/run-name/task__hash
    poetry run python analysis/rollout_to_text.py \
        outputs/evaluation/run-name/task__hash -o output.txt
"""

import argparse
import json
import sys
from pathlib import Path


def find_rollout_file(input_path: Path) -> Path:
    """Find the rollout JSONL file from a task directory or file path."""
    if input_path.is_file() and input_path.suffix == ".jsonl":
        return input_path

    if input_path.is_dir():
        patterns = [
            "agent/sessions/*/*/*rollout*.jsonl",
            "agent/sessions/*/*rollout*.jsonl",
            "**/rollout*.jsonl",
        ]
        for pattern in patterns:
            files = list(input_path.glob(pattern))
            if files:
                return sorted(files)[-1]

    raise FileNotFoundError(f"No rollout JSONL file found in {input_path}")


def parse_rollout(path: Path) -> list[dict]:
    """Read and parse a JSONL rollout file."""
    lines = []
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return lines


def format_session_meta(payload: dict) -> str:
    """Format session metadata as header."""
    session_id = payload.get("id", "unknown")[:8]
    cwd = payload.get("cwd", "unknown")
    model = payload.get("model_provider", "unknown")
    cli_version = payload.get("cli_version", "")

    lines = [
        "=" * 70,
        f"SESSION: {session_id}",
        "=" * 70,
        f"Model: {model}",
        f"Working Directory: {cwd}",
    ]
    if cli_version:
        lines.append(f"CLI Version: {cli_version}")
    lines.append("")
    return "\n".join(lines)


def format_message(payload: dict) -> str:
    """Format a user or assistant message."""
    role = payload.get("role", "unknown").upper()
    content_list = payload.get("content", [])

    text_parts = []
    for item in content_list:
        if isinstance(item, dict):
            if (
                item.get("type") in ("input_text", "output_text")
                or item.get("type") == "text"
            ):
                text_parts.append(item.get("text", ""))
        elif isinstance(item, str):
            text_parts.append(item)

    text = "\n".join(text_parts).strip()
    if not text:
        return ""

    return f"[{role}]\n{text}\n"


def format_function_call(payload: dict) -> str:
    """Format a function/tool call."""
    name = payload.get("name", "unknown")
    args_raw = payload.get("arguments", "")

    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {"raw": args_raw}
    else:
        args = args_raw

    lines = [f"[TOOL: {name}]"]

    if name == "shell":
        cmd = args.get("command", [])
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        lines.append(f"$ {cmd}")
    elif name == "apply_patch":
        patch = args.get("patch", "")
        lines.append(patch[:2000] + ("..." if len(patch) > 2000 else ""))
    else:
        args_str = json.dumps(args, indent=2)
        if len(args_str) > 2000:
            args_str = args_str[:2000] + "..."
        lines.append(args_str)

    return "\n".join(lines)


def format_function_output(payload: dict, _pending_calls: dict) -> str:
    """Format function call output."""
    output = payload.get("output", "")
    metadata = payload.get("metadata", {})
    exit_code = metadata.get("exit_code")

    lines = []

    if exit_code is not None:
        lines.append(f"Exit: {exit_code}")

    if output:
        if len(output) > 5000:
            output = output[:5000] + f"\n... (truncated, {len(output)} chars total)"
        lines.append(output)

    return "\n".join(lines) if lines else ""


def _format_response_item(payload: dict, pending_calls: dict[str, dict]) -> list[str]:
    """Format a response item and return output parts."""
    output_parts: list[str] = []
    item_type = payload.get("type", "")

    if item_type == "message":
        formatted = format_message(payload)
        if formatted:
            output_parts.append(formatted)

    elif item_type == "function_call":
        call_id = payload.get("call_id", "")
        pending_calls[call_id] = payload
        formatted = format_function_call(payload)
        if formatted:
            output_parts.append(formatted)

    elif item_type == "function_call_output":
        formatted = format_function_output(payload, pending_calls)
        if formatted:
            output_parts.append(formatted)
        output_parts.append("")

    elif item_type == "reasoning":
        output_parts.extend(_format_reasoning(payload))

    return output_parts


def _format_reasoning(payload: dict) -> list[str]:
    """Format reasoning summary."""
    output_parts: list[str] = []
    summary = payload.get("summary", [])
    if summary:
        output_parts.append("[REASONING]")
        for s in summary:
            if isinstance(s, dict):
                output_parts.append(s.get("text", str(s)))
            else:
                output_parts.append(str(s))
        output_parts.append("")
    return output_parts


def convert_to_text(lines: list[dict]) -> str:
    """Convert parsed JSONL lines to formatted text."""
    output_parts: list[str] = []
    pending_calls: dict[str, dict] = {}
    turn_num = 0

    for entry in lines:
        entry_type = entry.get("type", "")
        payload = entry.get("payload", {})

        if entry_type == "session_meta":
            output_parts.append(format_session_meta(payload))
        elif entry_type == "turn_context":
            turn_num += 1
            model = payload.get("model", "")
            output_parts.append(f"\n{'─' * 50}")
            output_parts.append(f"Turn {turn_num}" + (f" [{model}]" if model else ""))
            output_parts.append("─" * 50 + "\n")
        elif entry_type == "response_item":
            output_parts.extend(_format_response_item(payload, pending_calls))

    return "\n".join(output_parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert agent rollout JSONL to readable text"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to task directory or rollout JSONL file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output text file path (default: rollout.txt in same directory)",
    )
    args = parser.parse_args()

    try:
        rollout_path = find_rollout_file(args.input_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"Reading: {rollout_path}", file=sys.stderr)

    lines = parse_rollout(rollout_path)
    if not lines:
        print("Error: Empty rollout file")
        return 1

    text = convert_to_text(lines)

    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(text)
        print(f"Written: {args.output}", file=sys.stderr)
        print(f"  {len(lines)} entries -> {len(text)} chars", file=sys.stderr)
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())

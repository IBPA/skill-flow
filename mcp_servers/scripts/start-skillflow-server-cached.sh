#!/usr/bin/env bash
# Start the SkillFlow MCP server in cached mode (no GPU).
# Skill keys are pushed per task by the benchmark agent via /set-task.
#
# Usage:
#   ./mcp_servers/scripts/start-skillflow-cached.sh

set -euo pipefail

PORT=8765
NGROK_DOMAIN="uncontended-unconsumptively-cletus.ngrok-free.dev"
BASE_URL="https://${NGROK_DOMAIN}"
CONFIG="skill_flow/config/default.json"
TASKS_DIR="integration/skillsbench/tasks"
LOG_FILE="log.jsonl"

exec uv run python -m mcp_servers.skillflow_server \
  --port "$PORT" \
  --config "$CONFIG" \
  --base-url "$BASE_URL" \
  --tasks-dir "$TASKS_DIR" \
  --log-file "$LOG_FILE" \
  --cached \
  "$@"

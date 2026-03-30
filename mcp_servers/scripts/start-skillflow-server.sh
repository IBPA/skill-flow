#!/usr/bin/env bash
# Start the SkillFlow retriever MCP server (live — requires GPU).
#
# Usage:
#   ./mcp_servers/scripts/start-skillflow-server.sh

set -euo pipefail

PORT=8765
NGROK_DOMAIN="uncontended-unconsumptively-cletus.ngrok-free.dev"
BASE_URL="https://${NGROK_DOMAIN}"
CONFIG="skill_flow/config/default.json"
LOG_FILE="log.jsonl"

exec uv run python -m mcp_servers.skillflow_server \
  --port "$PORT" \
  --config "$CONFIG" \
  --base-url "$BASE_URL" \
  --log-file "$LOG_FILE" \
  "$@"

#!/usr/bin/env bash
# Start ngrok tunnel for the MCP server.
#
# Usage:
#   ./mcp_servers/scripts/start-ngrok.sh
#   ./mcp_servers/scripts/start-ngrok.sh --port 9000

set -euo pipefail

PORT=8765
NGROK_DOMAIN="uncontended-unconsumptively-cletus.ngrok-free.dev"

exec npx ngrok http "$PORT" --domain="$NGROK_DOMAIN" "$@"

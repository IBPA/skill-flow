#!/usr/bin/env bash
# Run SkillFlow live retriever MCP experiment.
#
# Starts a single SkillFlow retriever server (all 4 stages), tunnels it
# via ngrok, and runs the benchmark evaluation. Unlike run_mcp_eval.sh,
# the server handles all tasks without restart.
#
# Usage:
#   ./scripts/run_skillflow_mcp_eval.sh

set -euo pipefail

PORT=8765
NGROK_DOMAIN="uncontended-unconsumptively-cletus.ngrok-free.dev"
BASE_URL="https://${NGROK_DOMAIN}"
CONFIG="skill_flow/config/default.json"
EVAL_CONFIG="benchmark/config/experiments/skillflow-mcp-10.json"
LOG_FILE="log.jsonl"

MCP_PID=""
NGROK_PID=""

cleanup() {
  echo ""
  echo "Cleaning up..."
  [[ -n "$MCP_PID" ]] && kill "$MCP_PID" 2>/dev/null && wait "$MCP_PID" 2>/dev/null || true
  [[ -n "$NGROK_PID" ]] && kill "$NGROK_PID" 2>/dev/null && wait "$NGROK_PID" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# 1. Start SkillFlow retriever server
echo "Starting SkillFlow retriever server on port $PORT..."
echo "(Model loading may take ~30 seconds)"
uv run python -m mcp_servers.skillflow_retriever_server \
  --port "$PORT" \
  --config "$CONFIG" \
  --base-url "$BASE_URL" \
  --log-file "$LOG_FILE" &
MCP_PID=$!

# Wait for server to initialize (model loading)
echo "Waiting for server to initialize..."
sleep 30

# Verify server is running
if ! kill -0 "$MCP_PID" 2>/dev/null; then
  echo "ERROR: Server failed to start. Check logs."
  exit 1
fi
echo "Server ready (PID $MCP_PID)"

# 2. Start ngrok tunnel
echo "Starting ngrok on port $PORT (domain: $NGROK_DOMAIN)..."
ngrok http "$PORT" --domain="$NGROK_DOMAIN" --log=stdout > /dev/null 2>&1 &
NGROK_PID=$!
sleep 3

echo "ngrok ready (PID $NGROK_PID)"
echo "========================================"
echo "MCP endpoint: ${BASE_URL}/mcp"
echo "Download endpoint: ${BASE_URL}/download/{key}"
echo "========================================"

# 3. Run benchmark evaluation
echo "Running benchmark evaluation..."
uv run python -m benchmark.scripts.cli run --config "$EVAL_CONFIG" || true

echo ""
echo "========================================"
echo "Evaluation complete."
echo "Query log: $LOG_FILE"

#!/usr/bin/env bash
# Run MCP skills experiment: one task at a time with paired skills.
#
# For each task: starts MCP server → starts ngrok → runs eval → cleans up.
# Ctrl+C kills everything.
#
# Usage:
#   ./benchmark/scripts/mcp-golden.sh
#   ./benchmark/scripts/mcp-golden.sh sales-pivot-analysis   # single task

set -euo pipefail

PORT=8765
NGROK_DOMAIN="uncontended-unconsumptively-cletus.ngrok-free.dev"
TASKS_DIR="integration/skillsbench/tasks"
CONFIG_TEMPLATE="benchmark/config/skillsbench/4-mcp-golden.json"
LOG_FILE="mcp_calls.jsonl"

ALL_TASKS=(
  sales-pivot-analysis
  mario-coin-counting
  flood-risk-analysis
  protein-expression-analysis
  sec-financial-report
  offer-letter-generator
  earthquake-plate-calculation
  manufacturing-fjsp-optimization
  dapt-intrusion-detection
  software-dependency-audit
)

# Allow running a single task via CLI arg
if [[ $# -gt 0 ]]; then
  TASKS=("$@")
else
  TASKS=("${ALL_TASKS[@]}")
fi

NGROK_PID=""
MCP_PID=""
TMP_CONFIG=""

cleanup() {
  echo ""
  echo "Cleaning up..."
  [[ -n "$MCP_PID" ]] && kill "$MCP_PID" 2>/dev/null && wait "$MCP_PID" 2>/dev/null || true
  [[ -n "$NGROK_PID" ]] && kill "$NGROK_PID" 2>/dev/null && wait "$NGROK_PID" 2>/dev/null || true
  [[ -n "$TMP_CONFIG" && -f "$TMP_CONFIG" ]] && rm -f "$TMP_CONFIG"
  echo "Done."
}
trap cleanup EXIT INT TERM

# Start ngrok (runs for entire session)
echo "Starting ngrok on port $PORT (domain: $NGROK_DOMAIN)..."
ngrok http "$PORT" --domain="$NGROK_DOMAIN" --log=stdout > /dev/null 2>&1 &
NGROK_PID=$!
sleep 3

echo "ngrok ready (PID $NGROK_PID)"
echo "========================================"

for task in "${TASKS[@]}"; do
  echo ""
  echo "=== Task: $task ==="

  # Generate temp config with just this one task
  TMP_CONFIG=$(mktemp /tmp/mcp-config-XXXXXX.json)
  python3 -c "
import json, sys
with open('$CONFIG_TEMPLATE') as f:
    cfg = json.load(f)
cfg['tasks']['include_tasks'] = ['$task']
cfg['job_name'] = 'sb-mcp-$task'
with open('$TMP_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)
"

  # Start MCP server for this task
  echo "Starting MCP server for '$task'..."
  uv run python -m mcp_servers.skillsbench_server \
    --port "$PORT" \
    --task-name "$task" \
    --tasks-dir "$TASKS_DIR" \
    --log-file "mcp_calls_${task}.jsonl" &
  MCP_PID=$!
  sleep 2

  # Run evaluation
  echo "Running evaluation..."
  uv run python -m benchmark.scripts.cli run --config "$TMP_CONFIG" \
    --job-name "sb-mcp-$task" || true

  # Stop MCP server
  echo "Stopping MCP server (PID $MCP_PID)..."
  kill "$MCP_PID" 2>/dev/null && wait "$MCP_PID" 2>/dev/null || true
  MCP_PID=""

  # Clean up temp config
  rm -f "$TMP_CONFIG"
  TMP_CONFIG=""

  echo "=== Done: $task ==="
done

echo ""
echo "========================================"
echo "All tasks complete."

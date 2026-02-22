#!/usr/bin/env bash
# Inspect the direct children (files/folders) inside each skill folder
# under integration/skillsbench/tasks/**/environment/skills/*/
#
# Usage:
#   ./scripts/inspect_skill_contents.sh [tasks_dir]
#
# Default tasks_dir: integration/skillsbench/tasks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TASKS_DIR="${1:-$REPO_ROOT/integration/skillsbench/tasks}"

if [ ! -d "$TASKS_DIR" ]; then
  echo "Error: tasks directory not found: $TASKS_DIR" >&2
  exit 1
fi

skill_count=0
task_count=0

# Collect entry<TAB>skill pairs into a temp file
tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT

while IFS= read -r skills_dir; do
  task_count=$((task_count + 1))
  task_name=$(echo "$skills_dir" | sed "s|$TASKS_DIR/||" | cut -d/ -f1)
  for skill in "$skills_dir"/*/; do
    [ -d "$skill" ] || continue
    skill_count=$((skill_count + 1))
    skill_name=$(basename "$skill")
    for entry in "$skill"*; do
      [ -e "$entry" ] || continue
      name=$(basename "$entry")
      if [ -d "$entry" ]; then
        printf '%s/\t%s/%s\n' "$name" "$task_name" "$skill_name"
      else
        printf '%s\t%s/%s\n' "$name" "$task_name" "$skill_name"
      fi
    done
  done
done < <(find "$TASKS_DIR" -maxdepth 3 -path "*/environment/skills" -type d) > "$tmpfile"

echo "Tasks with skills: $task_count"
echo "Total skill folders: $skill_count"
echo ""

# ---- Frequency table ----
echo "=== Frequency Table ==="
echo ""
echo "Freq  Entry"
echo "----  -----"
cut -f1 "$tmpfile" | sort | uniq -c | sort -rn
echo ""

# ---- Per-entry skill list (skip SKILL.md since it's in every skill) ----
echo "=== Skills containing each entry (excluding SKILL.md) ==="
echo ""

cut -f1 "$tmpfile" | sort -u | while IFS= read -r entry; do
  [ "$entry" = "SKILL.md" ] && continue
  count=$(grep -cP "^$(printf '%s' "$entry" | sed 's/[.[\/*+?(){}|^$]/\\&/g')\t" "$tmpfile")
  skills=$(grep -P "^$(printf '%s' "$entry" | sed 's/[.[\/*+?(){}|^$]/\\&/g')\t" "$tmpfile" | cut -f2 | sort | paste -sd ', ')
  echo "[$entry] ($count)"
  echo "  $skills"
  echo ""
done

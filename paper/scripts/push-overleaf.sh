#!/bin/bash
# One-way push of paper/ contents to Overleaf.
# The contents of paper/ become the root of Overleaf's master branch.
#
# Clones Overleaf, replaces contents with paper/, and pushes.
# Uses --force only when --reset is passed (for when Overleaf history diverges).
#
# Usage (run from inside paper/):
#   bash scripts/push-overleaf.sh [--reset]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"

if [ "$PWD" != "$PAPER_DIR" ]; then
    echo "Error: must be run from $PAPER_DIR (current: $PWD)"
    exit 1
fi

# Load .env: prefer paper/.env, fall back to project-root .env
if [ -f .env ]; then
    set -a; source .env; set +a
elif [ -f ../.env ]; then
    set -a; source ../.env; set +a
fi

if [ -z "${OVERLEAF_API_KEY:-}" ]; then
    echo "Error: OVERLEAF_API_KEY not set. Add it to .env or export it."
    exit 1
fi

if [ -z "${OVERLEAF_REPO_URL:-}" ]; then
    echo "Error: OVERLEAF_REPO_URL not set. Add it to .env or export it."
    exit 1
fi

# Require at least one tracked-ish file in paper/ (not just submissions/)
if [ -z "$(find . -maxdepth 1 -mindepth 1 ! -name submissions ! -name .git -print -quit)" ]; then
    echo "Error: paper/ is empty. Nothing to push."
    exit 1
fi

OVERLEAF_URL="https://git:${OVERLEAF_API_KEY}@${OVERLEAF_REPO_URL}"
FORCE_FLAG=""
if [ "${1:-}" = "--reset" ]; then
    FORCE_FLAG="--force"
    echo "Pushing paper/ to Overleaf (force reset)..."
else
    echo "Pushing paper/ to Overleaf..."
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

git clone "$OVERLEAF_URL" "$WORK_DIR"
# Wipe everything except .git
find "$WORK_DIR" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +

# Copy paper/ contents to Overleaf root, excluding submissions/ and .git
for item in * .[!.]*; do
    [ -e "$item" ] || continue
    [ "$item" = "submissions" ] && continue
    [ "$item" = ".git" ] && continue
    cp -r "$item" "$WORK_DIR"/
done

cd "$WORK_DIR"
git add -A

if git diff --cached --quiet; then
    echo "No changes to push. Overleaf is already up to date."
    exit 0
fi

git commit -m "Sync paper/ from project repo"
git push $FORCE_FLAG origin master
echo "Done."

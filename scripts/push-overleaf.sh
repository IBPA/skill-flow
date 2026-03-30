#!/bin/bash
# One-way push of paper/ directory to Overleaf.
# Only the contents of paper/ are pushed as the root of Overleaf's master branch.
#
# Clones Overleaf, replaces contents with paper/, and pushes.
# Uses --force only when --reset is passed (for when Overleaf history diverges).
#
# Usage: bash scripts/push-overleaf.sh [--reset]

set -euo pipefail

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${OVERLEAF_API_KEY:-}" ]; then
    echo "Error: OVERLEAF_API_KEY not set. Add it to .env or export it."
    exit 1
fi

if [ -z "${OVERLEAF_REPO_URL:-}" ]; then
    echo "Error: OVERLEAF_REPO_URL not set. Add it to .env or export it."
    exit 1
fi

if [ -z "$(ls -A paper/)" ]; then
    echo "Error: paper/ directory is empty. Nothing to push."
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

# Work in a temp directory to avoid touching the main worktree
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

git clone "$OVERLEAF_URL" "$WORK_DIR"
# Remove all existing files (except .git)
find "$WORK_DIR" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
# Copy paper/ contents as root
cp -r paper/* "$WORK_DIR"/
# Include dotfiles like .gitignore if they exist
for f in paper/.*; do
    [ -f "$f" ] && cp "$f" "$WORK_DIR"/
done
cd "$WORK_DIR"
git add -A

if git diff --cached --quiet; then
    echo "No changes to push. Overleaf is already up to date."
    exit 0
fi

git commit -m "Sync paper/ from skill-flow"
git push $FORCE_FLAG origin master
echo "Done."

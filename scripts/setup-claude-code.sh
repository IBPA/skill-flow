#!/usr/bin/env bash
#
# Setup script for .claude directory
# Clones the dot-claude repo and copies its contents into .claude/
# Preserves any existing files not present in the repo
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

REPO_URL="https://github.com/fangzhouli/dot-claude"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Cloning dot-claude into temp directory..."
git clone --depth 1 "$REPO_URL" "$TMPDIR/dot-claude"

# Remove .git so .claude is a plain directory
rm -rf "$TMPDIR/dot-claude/.git"

# Ensure .claude directory exists
mkdir -p .claude

# Copy repo contents into .claude, overwriting same-named files but preserving extras
echo "Copying files into .claude/..."
cp -a "$TMPDIR/dot-claude/." .claude/

# Move CLAUDE.md to project root
if [ -f ".claude/CLAUDE.md" ]; then
    echo "Moving CLAUDE.md to project root..."
    mv .claude/CLAUDE.md "$PROJECT_ROOT/CLAUDE.md"
fi

# Make hook scripts executable
if [ -d ".claude/hooks" ]; then
    echo "Making hook scripts executable..."
    chmod +x .claude/hooks/*.sh 2>/dev/null || true
fi

echo "Done! .claude directory is ready."

#!/bin/bash
# Bootstrap a private GitHub repo for venue-specific paper submissions.
# Initializes git in paper/submissions/, creates a private GitHub repo,
# and pushes the initial commit.
#
# Works for both empty paper/submissions/ (new project) and existing content
# (one-time migration of pre-existing submissions).
#
# Usage: bash paper/scripts/init-submissions.sh [<repo-name>]
#
# Default repo name: <project-dir>_submissions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(cd "$PAPER_DIR/.." && pwd)"
SUBMISSIONS_DIR="$PAPER_DIR/submissions"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
REPO_NAME="${1:-${PROJECT_NAME}_submissions}"

if ! command -v gh >/dev/null 2>&1; then
    echo "Error: gh CLI not installed. See https://cli.github.com/."
    exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
    echo "Error: gh not authenticated. Run 'gh auth login' first."
    exit 1
fi

GH_USER="$(gh api user -q .login)"

if gh repo view "$GH_USER/$REPO_NAME" >/dev/null 2>&1; then
    echo "Error: GitHub repo '$GH_USER/$REPO_NAME' already exists."
    exit 1
fi

if [ -d "$SUBMISSIONS_DIR/.git" ]; then
    echo "Error: $SUBMISSIONS_DIR is already a git repository."
    exit 1
fi

mkdir -p "$SUBMISSIONS_DIR"

if [ ! -f "$SUBMISSIONS_DIR/README.md" ]; then
    cat > "$SUBMISSIONS_DIR/README.md" <<EOF
# ${REPO_NAME}

Private repository for venue-specific submissions of the \`${PROJECT_NAME}\` paper.

## Layout

Each top-level directory is a venue submission, e.g.:

\`\`\`
.
├── ACM_CAIS/
├── COLM/
└── NeurIPS/
\`\`\`

This repo is cloned into the parent project's \`paper/submissions/\`, which is
gitignored in the public project repo.
EOF
fi

if [ ! -f "$SUBMISSIONS_DIR/.gitignore" ]; then
    cat > "$SUBMISSIONS_DIR/.gitignore" <<'EOF'
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.synctex(busy)
*.dvi
*.cut
EOF
fi

cd "$SUBMISSIONS_DIR"
git init -q -b main
git add .
git commit -q -m "Initial commit"

gh repo create "$REPO_NAME" \
    --private \
    --description "Venue submissions for ${PROJECT_NAME}" \
    --source=. \
    --remote=origin \
    --push

echo "Done. Private repo created and pushed:"
echo "  https://github.com/${GH_USER}/${REPO_NAME}"

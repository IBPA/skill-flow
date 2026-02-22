---
allowed-tools: Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(git status:*), Bash(git diff:*), Bash(git checkout:*), Bash(git log:*), Bash(git branch:*), Read, Edit
description: Create a git commit on a feature branch and push to remote
---

## Context

- Current status: !`git status`
- Staged changes: !`git diff --cached --stat`

## Task

Create a git commit and push to a feature branch on the remote origin.

1. **Check staging status**:
   - If there are already staged changes, commit only those (do NOT run `git add`)
   - If there are no staged changes but there are unstaged changes, stage all with `git add -A` then commit
   - If there are no changes at all, inform the user and stop

2. **Ensure feature branch**:
   - NEVER push directly to `main` or `dev` branches
   - If currently on `main` or `dev`, create a new feature branch first:
     - Derive the branch name from the changes (e.g. `feat/add-tasks-path-support`)
     - Use `git checkout -b <branch-name>`
   - If already on a feature branch, stay on it

3. **Commit**: Create a commit with an appropriate message based on the changes
   - Pre-commit hooks run automatically (ruff, mypy, bandit, formatting)
   - If hooks fail, fix the errors, re-stage files, and retry commit

4. **Push**: Push the feature branch to the remote origin
   - Use `git push -u origin <branch-name>`
   - Pre-push hooks run automatically (pytest)
   - If tests fail, fix the errors, amend the commit, and retry push

## Notes

- Respect the user's intent: pre-staged files indicate deliberate selection
- Use `git diff --cached` to see what will actually be committed
- Keep retrying until commit and push succeed
- Do NOT add "Co-Authored-By" lines or "Generated with Claude Code" footers to the commit message

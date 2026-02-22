---
allowed-tools: Bash(gh pr create:*), Bash(gh pr view:*), Bash(git log:*), Bash(git branch:*), Bash(gh repo view:*)
description: Create a pull request for the current branch
---

## Context

- Current branch: !`git branch --show-current`
- Existing PR: !`gh pr view 2>/dev/null || echo 'No existing PR'`

## Task

Create a pull request for the current branch using the GitHub CLI (`gh pr create`).

1. **Pre-flight checks**:
   - Get the default branch with `gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name'`
   - If already on the default branch, stop and inform the user to create a feature branch first
   - If a PR already exists for this branch, show its details and ask if the user wants to edit it instead
   - Check for unpushed commits and warn the user to push first if needed

2. **Gather context**:
   - List commits on this branch not in the default branch using `git log`

3. **Create the PR**:
   - Generate a clear, descriptive title starting with `feat:`, `fix:`, or `refactor:` prefix
   - Write a detailed description based on the commits, including:
     - Summary of what changed and why
     - Key implementation details if relevant
   - Target the default branch

4. **Confirm**: Display the created PR URL

## Notes

- If not authenticated with `gh`, inform the user to run `gh auth login`
- Do NOT add "Co-Authored-By" lines or "Generated with Claude Code" footers to the PR description

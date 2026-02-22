# Claude Configs

Shared Claude Code configuration files for use as a git submodule across projects.

## Contents

- **settings.json** - Claude Code settings with post-tool-use hooks
- **hooks/** - Shell scripts for linting, type checking, and security scanning
- **commands/** - Custom slash commands
- **skills/** - Custom skills

## Usage as a Submodule

### Adding to a new project

```bash
git submodule add <repo-url> .claude
git commit -m "Add claude-configs submodule"
```

### Cloning a project with this submodule

```bash
git clone --recurse-submodules <project-url>
```

Or if already cloned:

```bash
git submodule update --init --recursive
```

### Updating the submodule

```bash
git submodule update --remote .claude
git commit -m "Update claude-configs submodule"
```

## Hooks

The hooks run automatically after `Edit` or `Write` tool calls:

- **check-ruff.sh** - Python linting with ruff
- **check-types.sh** - Type checking
- **check-bandit.sh** - Security scanning with bandit

Ensure the required tools are installed in your project environment.

## Skills

- **skill-creator** - Helps create and validate new Claude Code skills

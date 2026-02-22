# SkillFlow Evaluation Framework

This evaluation framework uses Harbor to benchmark agent performance on SWE-bench-verified tasks, comparing baseline Codex agents against skill-augmented agents to measure the impact of skill learning.

## Quick Start

### Baseline Evaluation (No Skills)

```bash
./benchmark/scripts/run-harbor-baseline.sh django django
```

### Skill-Augmented Evaluation

```bash
./benchmark/scripts/run-harbor-with-skills.sh django django
```

This will load skills from `outputs/skills/django/` and inject them into the Codex agent before task execution.

## Overview

The evaluation framework consists of:

1. **Baseline Agent** - Standard Harbor Codex agent without skill augmentation
2. **Skill-Augmented Agent** - `CodexWithSkills` agent that injects pre-existing skills at setup
3. **Evaluation Scripts** - Wrapper scripts for running Harbor benchmarks
4. **Skill Repository** - Pre-existing skills stored in `outputs/skills/{repository}/`

## Running Evaluations

### Baseline Evaluation

Run standard Codex agent without skills:

```bash
./benchmark/scripts/run-harbor-baseline.sh <job_name> <task_prefix> [--resume]
```

**Arguments**:
- `job_name`: Name for this baseline job (e.g., "django", "pylint", "all")
- `task_prefix`: Task name prefix filter (e.g., "django", "pylint-dev", "*" for all)
- `--resume`: Optional flag to resume previously started job

**Examples**:
```bash
# Django tasks baseline
./benchmark/scripts/run-harbor-baseline.sh django django

# Pylint tasks baseline
./benchmark/scripts/run-harbor-baseline.sh pylint pylint-dev

# All tasks baseline
./benchmark/scripts/run-harbor-baseline.sh all "*"

# Resume failed tasks
./benchmark/scripts/run-harbor-baseline.sh django django --resume
```

**Configuration**:
- Agent: `codex` (standard Harbor Codex agent)
- Model: `openai/gpt-5-mini`
- Dataset: `swebench-verified@1.0`
- Concurrency: 10 concurrent tasks
- Output: `outputs/harbor/swebench-{job_name}-baseline/`

### Skill-Augmented Evaluation

Run Codex agent with pre-loaded skills:

```bash
./benchmark/scripts/run-harbor-with-skills.sh <skills_repo> <task_prefix> [--resume]
```

**Arguments**:
- `skills_repo`: Skill repository to use (e.g., "django", "pylint", "all")
- `task_prefix`: Task name prefix filter (e.g., "django", "pylint-dev", "*" for all)
- `--resume`: Optional flag to resume previously started job

**Examples**:
```bash
# Django skills for django tasks
./benchmark/scripts/run-harbor-with-skills.sh django django

# Pylint skills for pylint tasks
./benchmark/scripts/run-harbor-with-skills.sh pylint pylint-dev

# All skills for all tasks
./benchmark/scripts/run-harbor-with-skills.sh all "*"

# Resume failed tasks
./benchmark/scripts/run-harbor-with-skills.sh django django --resume
```

**Configuration**:
- Agent: `CodexWithSkills` (custom agent in `benchmark.agents.codex_with_skills`)
- Model: `openai/gpt-5-mini`
- Dataset: `swebench-verified@1.0`
- Concurrency: 10 concurrent tasks
- Skills Source: `outputs/skills/{skills_repo}/`
- Output: `outputs/harbor/swebench-{skills_repo}-skillflow/`

## Skill Repository

Skills are stored in `outputs/skills/{repository}/{skill-name}/SKILL.md`:

```
outputs/skills/
├── django/
│   ├── django-orm/
│   │   └── SKILL.md
│   ├── django-migrations/
│   │   └── SKILL.md
│   ├── django-forms/
│   │   └── SKILL.md
│   └── django-admin/
│       └── SKILL.md
└── pylint/
    ├── pylint-checkers/
    │   └── SKILL.md
    ├── pylint-configuration/
    │   └── SKILL.md
    └── pylint-lint-execution/
        └── SKILL.md
```

### Skill Format

Skills use YAML frontmatter with markdown content:

```markdown
---
name: skill-name
description: Brief description of when to use this skill
---

# Skill Title

## Key Directories
Relevant directories in the codebase

## Investigation Workflow
Step-by-step debugging approach

## Common Fix Patterns
Code patterns with examples

## Common Pitfalls
Mistakes to avoid
```

## How Skills Are Injected

The `CodexWithSkills` agent extends the standard Harbor Codex agent to inject skills at setup time:

1. **Agent Initialization**: `CodexWithSkills.setup()` is called
2. **Standard Setup**: Runs normal Codex installation (Node.js, npm, codex CLI)
3. **Skill Discovery**: Recursively finds all directories containing `SKILL.md` in the specified skills source directory
4. **Upload to Container**: Uploads entire skill folders to `$CODEX_HOME/skills/` in the Docker container
5. **Logging**: Copies skills to job logs directory for visibility
6. **Task Execution**: Agent can reference skills during problem-solving

Skills are injected **before** the agent starts working on tasks, allowing the agent to reference relevant patterns and approaches from the beginning.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ Harbor Job                                          │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Docker Container                                │ │
│ │ ┌─────────────────────────────────────────────┐ │ │
│ │ │ Codex Agent                                 │ │ │
│ │ │ CODEX_HOME=/logs/agent                      │ │ │
│ │ │ ┌─────────────────────────────────────────┐ │ │ │
│ │ │ │ skills/                                 │ │ │ │
│ │ │ │   ├── django-orm/       ← injected      │ │ │ │
│ │ │ │   │   └── SKILL.md                      │ │ │ │
│ │ │ │   ├── django-migrations/                │ │ │ │
│ │ │ │   │   └── SKILL.md                      │ │ │ │
│ │ │ │   └── (other skills)                    │ │ │ │
│ │ │ └─────────────────────────────────────────┘ │ │ │
│ │ └─────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ Persistent Output:                                  │
│ outputs/harbor/swebench-django-skillflow/           │
│   └── {task-id}/                                    │
│       ├── trajectory.json                           │
│       ├── logs.txt                                  │
│       ├── patch.diff                                │
│       └── agent/                                    │
│           └── skills/  ← copied here for inspection │
│               ├── django-orm/                       │
│               └── ...                               │
└─────────────────────────────────────────────────────┘
```

## Output Structure

```
outputs/harbor/{job_name}/
├── config.yaml              # Harbor job configuration
├── logs/
│   └── agent/
│       └── skills/          # Injected skills (skill-augmented runs only)
│           ├── django-orm/
│           │   └── SKILL.md
│           └── ...
└── tasks/
    └── {task_id}/
        ├── trajectory.json  # Full execution trace
        ├── logs.txt         # Agent logs
        └── patch.diff       # Generated patch
```

## Analyzing Results

### View Trajectory

```bash
cat outputs/harbor/swebench-django-skillflow/tasks/{task-id}/trajectory.json | jq
```

### Check Injected Skills

```bash
ls -la outputs/harbor/swebench-django-skillflow/logs/agent/skills/
```

### View Agent Logs

```bash
cat outputs/harbor/swebench-django-skillflow/tasks/{task-id}/logs.txt
```

### Compare Baseline vs Skill-Augmented

```bash
# Baseline results
cat outputs/harbor/swebench-django-baseline/tasks/{task-id}/patch.diff

# Skill-augmented results
cat outputs/harbor/swebench-django-skillflow/tasks/{task-id}/patch.diff
```

## Advanced Configuration

### Custom Skills Directory

Specify a different skills directory:

```bash
poetry run harbor run \
    --agent-import-path benchmark.agents.codex_with_skills:CodexWithSkills \
    --model openai/gpt-5-mini \
    --dataset swebench-verified@1.0 \
    --agent-kwarg "skills_source_dir=/path/to/custom/skills"
```

### Multiple Skill Sets

Create different skill directories for different experiments:

```bash
outputs/skills/
├── django/              # Django-specific skills
├── pylint/              # Pylint-specific skills
└── experiments/         # Experimental skill sets
    ├── minimal/         # Minimal skill set
    └── comprehensive/   # Comprehensive skill set
```

Run with specific set:
```bash
./benchmark/scripts/run-harbor-with-skills.sh experiments/minimal django
```

## Troubleshooting

### Skills Not Loading

Check the agent setup logs:
```bash
cat outputs/harbor/swebench-django-skillflow/logs/agent/setup.txt
```

Verify skills were uploaded:
```bash
ls outputs/harbor/swebench-django-skillflow/logs/agent/skills/
```

### Import Errors

Ensure the module is importable:
```bash
poetry run python -c "from benchmark.agents import CodexWithSkills; print('OK')"
```

### Skills Directory Not Found

Ensure skills exist at the expected location:
```bash
ls -la outputs/skills/django/
find outputs/skills/django -name "SKILL.md"
```

## File Structure

```
benchmark/
├── README.md                    # This file
├── agents/
│   ├── __init__.py
│   └── codex_with_skills.py    # Custom agent with skill injection
└── scripts/
    ├── run-harbor-baseline.sh   # Baseline evaluation script
    └── run-harbor-with-skills.sh # Skill-augmented evaluation script

outputs/
├── skills/                      # Skill repository
│   ├── django/
│   │   ├── django-orm/
│   │   │   └── SKILL.md
│   │   └── ...
│   └── pylint/
│       └── ...
└── harbor/                      # Evaluation outputs
    ├── swebench-django-baseline/
    └── swebench-django-skillflow/
```

## Integration with SkillFlow Core

The evaluation framework is **independent** from the core SkillFlow peer-to-peer system:

- **Skill Storage**: Skills are stored locally in `outputs/skills/`, not in the core `skillflow/repository/`
- **No Network**: Evaluation doesn't use peer-to-peer SkillFlow network or discovery
- **Docker Isolation**: Each task runs in an isolated Docker container
- **Pre-existing Skills**: Skills are loaded from the filesystem, not discovered from peers

This separation allows for:
- Controlled experiments with fixed skill sets
- Reproducible benchmarks
- Baseline comparisons without network variability

## Next Steps

1. **Run Baseline Evaluation**: Establish baseline performance metrics
2. **Run Skill-Augmented Evaluation**: Measure impact of pre-existing skills
3. **Analyze Results**: Compare trajectories, patches, and success rates
4. **Iterate on Skills**: Refine skill content based on agent behavior
5. **Integrate SkillFlow Core**: Connect to SkillFlow P2P network for dynamic skill discovery (future work)

## Related Documentation

- [Harbor Documentation](https://github.com/laude-institute/harbor)
- [Codex CLI Documentation](https://github.com/openai/codex-cli)
- [SkillFlow Architecture](../CLAUDE.md)
- [SWE-bench](https://github.com/princeton-nlp/SWE-bench)

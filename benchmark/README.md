# SkillFlow Evaluation Framework

This evaluation framework uses Harbor to benchmark agent performance on SkillsBench tasks, comparing baseline agents against skill-augmented agents to measure the impact of skill retrieval.

## Quick Start

```bash
# Run with default config (baseline, no skills)
uv run python -m benchmark.scripts.cli run --config benchmark/config/default.json

# Run with golden skills injected
uv run python -m benchmark.scripts.cli run --config benchmark/config/skillsbench/2-inject-golden.json

# Run with SkillFlow-retrieved skills injected
uv run python -m benchmark.scripts.cli run --config benchmark/config/skillsbench/3-inject-skillflow.json

# Run MCP golden-skills experiment
./mcp_servers/scripts/start-skillflow-server.sh  # in another terminal
uv run python -m benchmark.scripts.cli run --config benchmark/config/skillsbench/5-mcp-skillflow.json
```

## Overview

The evaluation framework consists of:

1. **Baseline Agent** - Standard Harbor Codex agent without skill augmentation
2. **SkillFlow Injection Agent** (`SkillFlowInjectionAgent`) - Injects SkillFlow-retrieved SKILL.md files into containers
3. **SkillFlow MCP Agent** (`SkillFlowMcpAgent`) - Integrates with SkillFlow via MCP server
4. **SkillFlow MCP Cached Agent** (`SkillFlowMcpCachedAgent`) - MCP integration with cached skill results

## Evaluation Modes

Derived from config shape:

| Mode | Config Key | Agent | Description |
|------|-----------|-------|-------------|
| Baseline | (no `skills` field) | Codex | No skill augmentation |
| Skills | `skills.skills_dir` | `SkillAgent` | Injects SKILL.md files from a directory |
| SkillFlow | `skills.skillflow_peer_url` | `CodexWithSkillFlow` | Live SkillFlow HTTP peer integration |
| MCP | `skills.mcp` | `McpTestAgent` | MCP server integration |

## Configuration

Configs live in `benchmark/config/`. The default config (`default.json`) runs a baseline evaluation:

```json
{
  "job_name": "baseline-mini-10",
  "jobs_dir": "outputs/evaluation",
  "model": "openai/gpt-5-mini-2025-08-07",
  "reasoning_effort": "high",
  "tasks_path": "integration/skillsbench/tasks-no-skills",
  "num_runs": 1,
  "environment": {
    "use_daytona": true,
    "n_concurrent": 10
  },
  "tasks": {
    "include_tasks": ["sales-pivot-analysis", "..."],
    "exclude_tasks": []
  },
  "retry": {
    "resume": false,
    "retry_errors": false
  }
}
```

### Experiment Variants (SkillsBench ablation)

| Config | Description |
|--------|-------------|
| `default.json` | Baseline — no skills |
| `skillsbench/2-inject-golden.json` | GT skills injected into containers |
| `skillsbench/3-inject-skillflow.json` | SkillFlow-retrieved skills injected |
| `skillsbench/4-mcp-golden.json` | GT skills served via MCP |
| `skillsbench/5-mcp-skillflow.json` | Live SkillFlow retriever via MCP |

## How Skills Are Injected

The `SkillFlowInjectionAgent` extends the base Harbor agent to inject skills at setup time via `TarGzSkillInjector`:

1. **Agent Initialization**: `SkillFlowInjectionAgent.setup()` is called
2. **Standard Setup**: Runs normal Codex installation
3. **Skill Discovery**: Finds all SKILL.md files in the configured skills directory
4. **tar.gz Injection**: Packages and uploads skills into the Docker container
5. **Task Execution**: Agent references skills during problem-solving

## File Structure

```
benchmark/
├── README.md
├── config/
│   ├── default.json                    # Baseline config (no skills)
│   └── skillsbench/                    # SkillsBench experiment variants
│       ├── 2-inject-golden.json
│       ├── 3-inject-skillflow.json
│       ├── 4-mcp-golden.json
│       └── 5-mcp-skillflow.json
├── core/
│   ├── config.py                       # EvalConfig, SkillsConfig, EvalMode models
│   ├── runner.py                       # Harbor evaluation runner (single + multi-run)
│   ├── commands.py                     # Harbor CLI command builders
│   ├── display.py                      # Console output formatting
│   ├── utils.py                        # Job naming, task loading, Docker helpers
│   └── paths.py                        # Path management
├── scripts/
│   ├── cli.py                          # Main CLI: run subcommand
│   └── mcp-golden.sh                   # Run MCP golden-skills experiment
├── third_party/                       # External integrations (Vercel baseline)
└── agents/
    ├── base.py                         # BaseCodexAgent (shared reasoning_effort)
    ├── skillflow_injection_agent.py    # SkillFlow injection mode
    ├── skillflow_mcp_agent.py          # SkillFlow MCP mode
    ├── skillflow_mcp_cached_agent.py   # SkillFlow MCP cached mode
    ├── skills/                         # Skill management (manager.py, injector.py)
    └── instructions/                   # Jinja2 agent instruction templates
```

## Troubleshooting

### Import Errors

Ensure the module is importable:
```bash
uv run python -c "from benchmark.agents import SkillAgent; print('OK')"
```

## Related Documentation

- [SkillFlow Architecture](../CLAUDE.md)
- [Harbor Documentation](https://github.com/laude-institute/harbor)

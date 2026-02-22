# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SkillFlow** is an agent skill retrieval system that enables AI agents to discover and execute skills from online sources. The project has three main components:

1. **SkillFlow Core** (`skill_flow/`): A multi-stage semantic skill retrieval engine (FAISS vector search → cross-encoder reranking → optional second reranker → LLM selection) over ~36K agent skills
2. **Evaluation Framework** (`benchmark/`): A Harbor-based benchmarking system to evaluate agents with and without skill augmentation across benchmarks (Terminal Bench, SkillsBench)
3. **Analysis** (`analysis/`): LLM-based trajectory analysis for failure mode classification and skill amenability assessment

## Commands

```bash
# Install dependencies
uv sync

# Build FAISS index (one-time, requires ../skill-crawler/data/skills/)
uv run python -m skill_flow.cli build-index

# Search the index
uv run python -m skill_flow.cli search --query "write unit tests for FastAPI"

# Search with cross-encoder reranking (Stage 2)
uv run python -m skill_flow.cli search --query "write unit tests for FastAPI" --rerank

# Run config-driven evaluation (retriever + reranker + reranker2 + selector stages)
uv run python -m skill_flow.cli eval

# Run benchmark evaluation CLI
uv run python -m benchmark.scripts.cli run --config benchmark/config/default.json

# Run tests with coverage
uv run pytest tests/ -v

# Run failure analysis
uv run python -m analysis.failure.main --job-dir PATH --output-dir PATH [--model MODEL] [--limit N]
```

## Architecture

```
skill-flow/
├── skill_flow/                 # SkillFlow core library
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (build-index, search, eval)
│   ├── models/                 # Domain models
│   │   ├── __init__.py         # SkillRecord (Pydantic, frozen)
│   │   └── core.py             # SkillFlow facade (retriever + reranker + reranker2 + selector composition)
│   ├── config/                 # Configuration
│   │   ├── __init__.py         # Pydantic config models (Config, SystemConfig, IndexConfig, ModelsConfig, RetrieverConfig, RerankerConfig, Reranker2Config, SelectorConfig, QueryGenConfig, *EvalSettings)
│   │   ├── default.json        # Default config (system/index/models hierarchy)
│   │   ├── eval-reranker2.json # Config preset for reranker2 evaluation
│   │   └── eval-selector.json  # Config preset for selector-only evaluation
│   ├── corpus/                 # Corpus loading
│   │   └── loader.py           # load_corpus(), load_content()
│   ├── index/                  # FAISS index building
│   │   ├── encoder.py          # Encoder (BGE bi-encoder wrapper)
│   │   └── builder.py          # build_index() → embeddings.npy, faiss.index, skill_ids.json, skill_descriptions.json, skill_contents.json
│   ├── retriever/              # FAISS index search (Stage 1: retrieval)
│   │   └── retriever.py        # IndexSearcher, SearchResult (with content field)
│   ├── reranker/               # Cross-encoder reranking (Stage 2 + Stage 3)
│   │   ├── reranker.py         # Reranker (BGE cross-encoder, used for both reranker and reranker2)
│   │   └── query_gen.py        # QueryGenerator (LLM-based query generation with JSON caching)
│   ├── selector/               # LLM-based skill selection (Stage 4)
│   │   └── selector.py         # Selector (LLM-based skill selection with JSON caching)
│   └── eval/                   # Retriever + reranker + reranker2 + selector evaluation (against SkillsBench GT)
│       ├── models.py            # EvalRunConfig, RerankerEvalConfig, Reranker2EvalConfig, SelectorEvalConfig, TaskResult, EvalSummary, EvalReport
│       ├── metrics.py           # recall@k, precision@k, hit@k, reciprocal_rank
│       ├── ground_truth.py      # Load GT skills from SkillsBench tasks
│       ├── reporting.py         # Report building, writing, and incremental snapshot helpers
│       ├── cli_eval.py          # CLI eval helpers (extracted from cli.py for all 4 stages)
│       └── runner.py            # Orchestration: augment index + evaluate + report
│
├── benchmark/                  # Harbor evaluation framework
│   ├── config/                 # JSON config presets (default, pilot-*, skillbench-*)
│   ├── core/                   # Core modules
│   │   ├── config.py           # Configuration models (EvalConfig, SkillsConfig, etc.)
│   │   ├── runner.py           # Harbor evaluation runner (single + multi-run)
│   │   ├── commands.py         # Harbor CLI command builders
│   │   ├── display.py          # Console output formatting
│   │   ├── utils.py            # Job naming, task loading, Docker helpers
│   │   └── paths.py            # Path management
│   ├── scripts/
│   │   └── cli.py              # Main CLI: run, peer subcommands
│   └── agents/                 # Custom Harbor agents
│       ├── base.py             # BaseCodexAgent (shared reasoning_effort support)
│       ├── skill_agent.py      # Skills mode: injects SKILL.md files into containers
│       ├── codex_with_skillflow.py  # SkillFlow mode: HTTP peer integration
│       ├── skills/             # Skill management (manager.py, injector.py)
│       ├── skillsets/          # Curated skill lists (*.txt)
│       └── instructions/       # Jinja2 agent instruction templates
│
├── analysis/                   # Trajectory analysis
│   ├── failure/                # Failure mode classification (core.py, main.py)
│   └── eda/                    # Exploratory data analysis (skill_length.py)
│
├── integration/                # External benchmark integrations
│   └── skillsbench/            # SkillsBench benchmark (separate repo/venv)
│
├── tests/
│   ├── test_skill_flow/        # Core library tests
│   ├── test_benchmark/         # Evaluation framework tests
│   └── test_analysis/          # Analysis tools tests
│
├── e2e/                        # End-to-end testing
│   └── mcp/                    # MCP server integrations (dummy, skill_loader, skillsbench)
│
├── scripts/                    # Utility scripts
│   ├── setup-git-hooks.sh      # Install pre-commit hooks
│   ├── setup-claude-code.sh    # Configure Claude Code
│   └── find_skill_patterns.py  # Scan trajectories for SKILL.md references
│
├── jobs/                       # Job execution outputs (timestamped directories)
└── outputs/
    └── indices/                # Persisted FAISS index artifacts (gitignored)
```

## Key Patterns

- **Configuration**: Pydantic models with nested hierarchy (`system`/`index`/`models`) — `skill_flow/config/default.json` for core, `benchmark/config/` for eval
- **Eval Modes**: Derived from config shape — baseline (no skills field), skills (skills.skills_dir), skillflow (skills.skillflow_peer_url)
- **Multi-stage Retrieval**: Stage 1 FAISS bi-encoder (`BAAI/bge-base-en-v1.5`) → Stage 2 cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) → optional Stage 3 reranker2 (same cross-encoder class, configurable independently) → optional Stage 4 LLM selector (binary relevant/not-relevant filtering via OpenAI); `--rerank` enables Stage 2, Stage 3 chains automatically when `models.reranker2.enabled`, Stage 4 chains when `models.selector.enabled`
- **Query Generation**: Optional LLM-based step (`QueryGenerator`) that converts verbose task instructions into concise search queries before cross-encoder scoring; configured via `models.reranker.query_gen` / `models.reranker2.query_gen` with JSON file caching to avoid redundant LLM calls
- **FAISS Index**: Normalized embeddings + `IndexFlatIP` (inner product = cosine similarity)
- **Full Content Threading**: `skill_contents.json` persisted at build time; `SearchResult.content` carries full SKILL.md through the pipeline for cross-encoder scoring
- **Structured Skills**: SKILL.md format with YAML frontmatter for metadata
- **Lazy Content Loading**: `SkillRecord` stores metadata only; full SKILL.md loaded on demand via `load_content()`
- **Skill Injection**: tar.gz-based injection of SKILL.md files into Docker containers via `TarGzSkillInjector`
- **Job Naming**: Auto-generated as `{benchmark}-{mode}-{skills}-{model}-{effort}-{timestamp}`
- **Retriever Eval**: Injects all GT skills into FAISS index at eval time with task-scoped keys (`skillsbench/{task_id}/{name}`) so each task's exact skill content is evaluated; writes incremental report snapshots after each task
- **Chained Eval**: Reranker2 eval (`run_reranker2_evaluation`) reads cached Stage 2 report and re-ranks its results, sharing the same `_run_rerank_stage` logic as the Stage 2 reranker eval; Selector eval (`run_selector_evaluation`) reads cached Stage 3 report and applies LLM selection via `_run_selector_stage`
- **Eval Metrics**: recall@k, precision@k, hit@k, MRR (mean reciprocal rank)

## Code Standards

Configured in `pyproject.toml`:
- **Python**: 3.12+ required
- **Ruff**: E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, ERA, PL, RUF rules
- **MyPy**: Strict mode with type checking
- **Bandit**: Security scanning excluding tests
- **Pytest**: 80% coverage threshold (covers `skill_flow/` and `benchmark/`)

## General Rules

1. **File size limit**: Do not allow code files to exceed 300 lines. Refactor by splitting into smaller modules.
2. **No lazy bypasses**: Do not use `# noqa`, `# type: ignore` to bypass errors. Fix the underlying issue.
3. **Rely on pre-commit hooks**: Pre-commit hooks run on commit (ruff, mypy, bandit) and push (pytest). Only run checks manually when debugging.
4. **No cheating on test coverage**: Do not lower `--cov-fail-under` threshold or add files to `[tool.coverage.run] omit` to bypass failing coverage. Write proper tests instead.
5. **Use fixtures in tests**: When config classes have required fields, use fixtures or helper functions (e.g., `make_config()`) to construct test objects.
6. **Pydantic for all models**: Use Pydantic `BaseModel` consistently — not dataclasses.

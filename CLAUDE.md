# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SkillFlow** is an agent skill retrieval system that enables AI agents to discover and execute skills from online sources. The project has four main components:

1. **Skill Crawler** (`skill_crawler/`): Crawls and downloads ~36K agent skills from online marketplaces (SkillsMP) into `data/skills/`
2. **SkillFlow Core** (`skill_flow/`): A multi-stage semantic skill retrieval engine (FAISS vector search → cross-encoder reranking → optional second reranker → LLM selection) over ~36K agent skills
3. **Evaluation Framework** (`benchmark/`): A Harbor-based benchmarking system to evaluate agents with and without skill augmentation using SkillsBench and Terminal-Bench
4. **Analysis** (`analysis/`): Retrieval comparisons, statistical analysis, and paper table/figure generation
5. **Paper** (`paper/`): LaTeX source for the SkillFlow research paper, with Overleaf synchronization workflow

## Commands

```bash
# Install dependencies
uv sync

# Build FAISS index (one-time, requires data/skills/)
uv run python -m skill_flow.cli build-index

# Search the index
uv run python -m skill_flow.cli search --query "write unit tests for FastAPI"

# Search with cross-encoder reranking (Stage 2)
uv run python -m skill_flow.cli search --query "write unit tests for FastAPI" --rerank

# Run auto-chained evaluation (all 4 stages, default: skill_flow/config/default_eval.json)
uv run python -m skill_flow.cli eval

# Run eval with a specific config (e.g., retriever-only via default.json)
uv run python -m skill_flow.cli eval --config skill_flow/config/default.json

# Run benchmark evaluation CLI
uv run python -m benchmark.scripts.cli run --config benchmark/config/default.json

# Run tests with coverage
uv run pytest tests/ -v

# Run retriever comparison experiment
uv run python -m skill_flow.cli experiment --config skill_flow/config/experiments/retriever-comparison.json --max-tasks 2

# Run reranker comparison experiment
uv run python -m skill_flow.cli experiment --config skill_flow/config/experiments/reranker-comparison.json --max-tasks 2

# Compare two retriever eval reports (per-task win/loss analysis)
uv run python -m analysis.comparison.compare_retrievers REPORT_A REPORT_B --k 10 --top-n 10

# Regenerate all paper tables and figures from analysis data
bash analysis/results/generate-paper-assets.sh            # both
bash analysis/results/generate-paper-assets.sh --tables   # tables only
bash analysis/results/generate-paper-assets.sh --figures  # figures only

# Push paper/ to Overleaf (one-way sync)
bash paper/scripts/push-overleaf.sh

# Push paper/ to Overleaf with history reset (when histories diverge)
bash paper/scripts/push-overleaf.sh --reset

# Crawl all skill sources
uv run python -m skill_crawler crawl

# Crawl specific source
uv run python -m skill_crawler crawl --source skillsmp

# Dry run (list without downloading)
uv run python -m skill_crawler crawl --dry-run

# Search SkillsMP skills
uv run python -m skill_crawler search "PDF editing"

# Check crawler status
uv run python -m skill_crawler status

# Validate downloaded skills
uv run python -m skill_crawler validate ./data/skills
```

## Architecture

```
skill-flow/
├── skill_crawler/              # Skill crawler (data acquisition)
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI (crawl, search, status, validate, dedupe)
│   ├── config.py               # Pydantic settings (env vars + config.json)
│   ├── config/
│   │   └── config.json         # Default crawler configuration
│   ├── models/                 # Skill model, sync state
│   ├── crawlers/               # Source-specific crawlers (SkillsMP API/scraper)
│   ├── downloaders/            # GitHub archive/SKILL.md downloader
│   ├── storage/                # Save/index/validate/deduplicate skills
│   └── utils/                  # HTTP client, progress output
│
├── skill_flow/                 # SkillFlow core library
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (build-index, search, eval, experiment)
│   ├── models/                 # Domain models
│   │   ├── __init__.py         # SkillRecord (Pydantic, frozen)
│   │   └── core.py             # SkillFlow facade (retriever + reranker + deep_reranker + selector composition)
│   ├── config/                 # Configuration
│   │   ├── __init__.py         # Pydantic config models (Config, SystemConfig, IndexConfig, ModelsConfig, RetrieverConfig, RerankerConfig, DeepRerankerConfig, SelectorConfig, QueryGenConfig, RetrieverVariant, RetrieverExperimentConfig, RerankerVariant, RerankerExperimentConfig, *EvalSettings)
│   │   ├── default.json        # Default config (system/index/models hierarchy)
│   │   ├── default_eval.json           # Eval default config (all 4 stages enabled, auto-chained)
│   │   └── experiments/        # Config presets for evaluation experiments
│   │       ├── eval-deep-reranker.json # Config preset for deep reranker evaluation
│   │       ├── eval-selector.json  # Config preset for selector-only evaluation
│   │       ├── retriever-comparison.json      # Multi-retriever experiment (dense + BM25)
│   │       ├── retriever-querygen.json    # Retriever query gen grid search experiment
│   │       └── reranker-comparison.json   # Multi-reranker experiment (model/content variants)
│   ├── corpus/                 # Corpus loading
│   │   └── loader.py           # load_corpus(), load_content()
│   ├── index/                  # FAISS index building
│   │   ├── encoder.py          # Encoder (BGE bi-encoder wrapper), pick_device() (auto GPU selection)
│   │   └── builder.py          # build_index() → embeddings.npy, faiss.index, skill_ids.json, skill_descriptions.json, skill_contents.json
│   ├── retriever/              # Stage 1 retrieval (dense + BM25)
│   │   ├── protocol.py         # Searcher Protocol (shared interface)
│   │   ├── retriever.py        # IndexSearcher (FAISS dense), SearchResult
│   │   ├── multi_search.py     # Multi-query search orchestration (search_multi)
│   │   └── bm25.py             # BM25Searcher (rank-bm25 sparse retrieval)
│   ├── query_gen/              # LLM-based query generation (shared by retrieval + reranking)
│   │   └── query_gen.py        # QueryGenerator (LLM-based query generation with JSON caching)
│   ├── reranker/               # Cross-encoder reranking (Stage 2 + Stage 3)
│   │   └── reranker.py         # Reranker (BGE cross-encoder, used for both reranker and deep_reranker)
│   ├── selector/               # LLM-based skill selection (Stage 4)
│   │   └── selector.py         # Selector (LLM-based skill selection with JSON caching)
│   └── eval/                   # Retriever + reranker + deep_reranker + selector evaluation (against SkillsBench GT)
│       ├── models.py            # EvalRunConfig, RerankerEvalConfig, DeepRerankerEvalConfig, SelectorEvalConfig, TaskResult, EvalSummary, EvalReport
│       ├── cli_eval.py          # CLI eval helpers (extracted from cli.py for all 4 stages)
│       ├── runner.py            # Orchestration: augment searcher + evaluate + report
│       ├── experiments/         # Experiment runners for multi-variant comparisons
│       │   ├── __init__.py      # Re-exports public API (run_experiment, run_reranker_experiment, etc.)
│       │   ├── grid.py          # Shared grid-search helpers (query gen, aggregation, reaggregation)
│       │   ├── retriever.py     # Retriever comparison experiment runner
│       │   ├── reranker.py      # Reranker comparison experiment runner (with grid-search aggregation)
│       │   └── selector.py      # Selector comparison experiment runner
│       └── utils/               # Stateless utility modules
│           ├── __init__.py      # Re-exports public names from all submodules
│           ├── metrics.py       # recall@k, precision@k, hit@k, reciprocal_rank
│           ├── ground_truth.py  # Load GT skills from SkillsBench tasks
│           ├── reporting.py     # Report building, writing, and incremental snapshot helpers
│           └── helpers.py       # Shared utilities (slug)
│
├── benchmark/                  # Harbor evaluation framework
│   ├── config/                 # Benchmark configs (ablation hierarchy)
│   │   ├── default.json        # SkillsBench baseline — no skills
│   │   └── skillsbench/        # SkillsBench experiment variants
│   │       ├── 2-inject-golden.json      # GT skills injected
│   │       ├── 3-inject-skillflow.json   # SkillFlow-retrieved skills injected
│   │       ├── 4-mcp-golden.json         # GT skills via MCP
│   │       └── 5-mcp-skillflow.json      # Live SkillFlow via MCP
│   ├── core/                   # Core modules
│   │   ├── config.py           # Configuration models (EvalConfig, SkillsConfig, etc.)
│   │   ├── runner.py           # Harbor evaluation runner (single + multi-run)
│   │   ├── commands.py         # Harbor CLI command builders
│   │   ├── display.py          # Console output formatting
│   │   ├── utils.py            # Job naming, task loading, Docker helpers
│   │   └── paths.py            # Path management
│   ├── scripts/
│   │   ├── cli.py              # Main CLI: run, peer subcommands
│   │   └── mcp-golden.sh       # Run MCP golden-skills experiment (per-task server)
│   ├── third_party/            # External integrations (Vercel baseline)
│   └── agents/                 # Custom Harbor agents
│       ├── base.py             # BaseCodexAgent (shared reasoning_effort support)
│       ├── skillflow_injection_agent.py   # SkillFlow injection mode
│       ├── skillflow_mcp_agent.py         # SkillFlow MCP mode
│       ├── skillflow_mcp_cached_agent.py  # SkillFlow MCP cached mode
│       ├── skills/             # Skill management (manager.py, injector.py)
│       └── instructions/       # Jinja2 agent instruction templates
│
├── analysis/                   # Analysis and comparison tools
│   ├── comparison/            # Retrieval and benchmark comparisons
│   │   ├── compare_conditions.py  # Compare evaluation conditions
│   │   ├── compare_retrievers.py  # Per-task win/loss comparison between two retriever reports
│   │   ├── compare_runs.py        # Compare task-level results between two Harbor eval runs
│   │   └── utils/                 # Display and data loading helpers
│   ├── results/               # Paper table and figure generators (numbered by paper table order)
│   │   ├── t1_generate_results.py        # T1: tab:results
│   │   ├── t2_generate_adoption.py       # T2: tab:adoption
│   │   ├── t3_generate_retrieval_stages.py # T3: tab:retrieval_stages
│   │   ├── t4_generate_stage_ablation.py # T4: tab:stage_ablation
│   │   ├── t5_generate_latency.py        # T5: tab:latency
│   │   ├── t6_generate_corpus_stats.py   # T6: tab:corpus_stats (from crawler metadata)
│   │   ├── t7_generate_query_examples.py # T7: tab:query_examples (from query cache)
│   │   ├── t8_9_generate_retriever_comparison.py  # T8-9: retriever + query config
│   │   ├── t10_11_generate_reranker_comparison.py  # T10-11: reranker + deep reranker
│   │   ├── t12_generate_excluded_tasks.py # T12: tab:excluded_tasks
│   │   ├── t13_14_15_generate_case_studies.py  # T13-15: case study tables
│   │   ├── f2_plot_quality_proxies.py   # F2: quality proxy comparison → paper/figures/
│   │   ├── f3_plot_query_impact.py   # F3: composite multi-query impact (2×2 grid) → paper/figures/
│   │   ├── f4_plot_skill_dist.py        # F4: skill content distribution → paper/figures/
│   │   ├── generate-paper-assets.sh    # Regenerate all paper tables and figures
│   │   └── utils/                      # Shared helpers for generators
│   ├── stats/                 # Statistical utilities
│   │   ├── bootstrap.py       # Bootstrap confidence interval computation
│   │   ├── proportions.py     # Proportion comparison tests
│   │   ├── benchmark_stats.py # Benchmark-level statistical helpers
│   │   ├── retrieval_stats.py # Retrieval-level statistical helpers
│   │   └── types.py           # Shared statistical types
│   └── utils/                 # Shared utilities
│       └── find_skill_patterns.py  # Scan trajectories for SKILL.md references
│
├── mcp_servers/                # MCP server implementations
│   ├── dummy_server.py         # Minimal test server
│   ├── skillsbench_server.py   # SkillsBench per-task MCP server (golden skills)
│   ├── skillflow_server.py     # Live SkillFlow retriever server (all 4 stages)
│   ├── utils/
│   │   └── skill_loader.py     # Skill loading/resolving helpers
│   └── scripts/
│       ├── start-skillflow-server.sh  # Launch SkillFlow retriever server
│       └── start-ngrok.sh             # Start ngrok tunnel for MCP server
│
├── integration/                # External benchmark integrations
│   ├── skillsbench/            # SkillsBench benchmark (separate repo/venv)
│   └── terminal-bench/         # Terminal-Bench benchmark (separate repo/venv)
│
├── tests/
│   ├── test_skill_crawler/     # Crawler tests
│   ├── test_skill_flow/        # Core library tests
│   ├── test_benchmark/         # Evaluation framework tests
│   ├── test_analysis/          # Analysis tools tests
│   └── test_mcp_servers/       # MCP server tests
│
├── paper/                      # LaTeX paper source
│   ├── main.tex                # Paper source (uses \input{tables/...} for all tables)
│   ├── main.bib                # Bibliography
│   ├── main.pdf                # Compiled PDF
│   ├── tables/                 # Table .tex files (generated + static, included via \input)
│   ├── generated_case_studies.tex  # Case study section (narrative + \input{tables/case_study_N})
│   └── figures/                # Paper figures
│
├── docs/                       # Documentation
│   ├── plan.md                 # Mode detection table and skill injection strategies
│   └── ai-review/              # AI review documents
│
├── scripts/                    # Utility scripts
│   ├── setup-git-hooks.sh      # Install pre-commit hooks
│   ├── setup-claude-code.sh    # Configure Claude Code
│   ├── scripts/
│   │   └── push-overleaf.sh    # One-way push paper/ to Overleaf (subtree or --reset)
│
├── jobs/                       # Job execution outputs (timestamped directories)
└── outputs/
    └── indices/                # Persisted FAISS index artifacts (gitignored)
```

## Key Patterns

- **Configuration**: Pydantic models with nested hierarchy (`system`/`index`/`models`) — `skill_flow/config/default.json` for core, `skill_flow/config/default_eval.json` for eval (all stages enabled), `benchmark/config/default.json` + `benchmark/config/skillsbench/` for benchmark eval
- **Eval Modes**: Derived from config shape — baseline (no skills field), skills (skills.skills_dir), skillflow (skills.skillflow_peer_url)
- **Multi-stage Retrieval**: Stage 1 retrieval via `Searcher` protocol (dense FAISS or BM25 sparse, with optional LLM query generation via `models.retriever.query_gen`) → Stage 2 cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) → optional Stage 3 deep_reranker (same cross-encoder class, configurable independently) → optional Stage 4 LLM selector (binary relevant/not-relevant filtering via OpenAI); `--rerank` enables Stage 2, Stage 3 chains automatically when `models.deep_reranker.enabled`, Stage 4 chains when `models.selector.enabled`
- **Searcher Protocol**: `Searcher` in `skill_flow/retriever/protocol.py` defines the shared interface (`search`, `augment`, `add_descriptions`, `add_contents`) that both `IndexSearcher` (dense FAISS) and `BM25Searcher` (rank-bm25 sparse) implement
- **Retriever Experiments**: `RetrieverExperimentConfig` defines multi-retriever comparison experiments; `skill_flow.cli experiment` runs all variants and prints a comparison table
- **Reranker Experiments**: `RerankerExperimentConfig` defines multi-reranker comparison experiments; same `skill_flow.cli experiment` subcommand auto-detects type from config (`"rerankers"` key vs `"retrievers"` key). Supports grid-search aggregation: when `query_gen.aggregation` is a list (e.g. `["max", "mean", "rrf"]`), the cross-encoder scores once and results are re-aggregated for each method, producing separate reports without redundant inference. List aggregation is only allowed in experiment mode; non-experiment eval raises `TypeError` if aggregation is a list.
- **Query Generation**: Optional LLM-based step (`QueryGenerator`) that converts verbose task instructions into concise search queries; available for Stage 1 retrieval (`models.retriever.query_gen`, multi-query via `search_multi`) and Stages 2-3 reranking (`models.reranker.query_gen` / `models.deep_reranker.query_gen`). JSON file caching avoids redundant LLM calls. Supports multi-query generation (`num_queries > 1`) where the LLM produces N facet-focused queries per task; scores are aggregated across queries using configurable strategy (`aggregation`: `"max"`, `"mean"`, `"rrf"`, or a list for grid search in experiments)
- **GPU Device Selection**: `pick_device()` in `skill_flow/index/encoder.py` auto-selects the CUDA device with the most free memory; called by default in `Encoder.__init__` when no explicit device is given
- **FAISS Index**: Normalized embeddings + `IndexFlatIP` (inner product = cosine similarity)
- **Full Content Threading**: `skill_contents.json` persisted at build time; `SearchResult.content` carries full SKILL.md through the pipeline for cross-encoder scoring
- **Structured Skills**: SKILL.md format with YAML frontmatter for metadata
- **Lazy Content Loading**: `SkillRecord` stores metadata only; full SKILL.md loaded on demand via `load_content()`
- **Skill Injection**: tar.gz-based injection of SKILL.md files into Docker containers via `TarGzSkillInjector`
- **Job Naming**: Auto-generated as `{benchmark}-{mode}-{skills}-{model}-{effort}-{timestamp}`
- **Retriever Eval**: Injects all GT skills into FAISS index at eval time with task-scoped keys (`skillsbench/{task_id}/{name}`) so each task's exact skill content is evaluated; writes incremental report snapshots after each task
- **Auto-chained Eval**: `run_eval()` threads each stage's output path as the next stage's input — no explicit `stage*_report_path` needed when running the full pipeline; each `run_*_eval` function accepts an optional `prev_output_path` that overrides the configured input path, falling back to config for standalone usage
- **Eval Metrics**: recall@k, precision@k, hit@k, MRR (mean reciprocal rank)
- **Overleaf Sync**: `paper/scripts/push-overleaf.sh` does a one-way subtree push of `paper/` to Overleaf; `--reset` clones Overleaf, replaces contents, and force-pushes when histories diverge. Requires `OVERLEAF_API_KEY` and `OVERLEAF_REPO_URL` in `.env`

## Code Standards

Configured in `pyproject.toml`:
- **Python**: 3.12+ required
- **Ruff**: E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, ERA, PL, RUF rules
- **MyPy**: Strict mode with type checking
- **Bandit**: Security scanning excluding tests
- **Pytest**: 80% coverage threshold (covers `skill_flow/`, `skill_crawler/`, and `benchmark/`)

## General Rules

1. **File size limit**: Do not allow code files to exceed 300 lines. Refactor by splitting into smaller modules.
2. **No lazy bypasses**: Do not use `# noqa`, `# type: ignore` to bypass errors. Fix the underlying issue.
3. **Rely on pre-commit hooks**: Pre-commit hooks run on commit (ruff, mypy, bandit) and push (pytest). Only run checks manually when debugging.
4. **No cheating on test coverage**: Do not lower `--cov-fail-under` threshold or add files to `[tool.coverage.run] omit` to bypass failing coverage. Write proper tests instead.
5. **Use fixtures in tests**: When config classes have required fields, use fixtures or helper functions (e.g., `make_config()`) to construct test objects.
6. **Pydantic for all models**: Use Pydantic `BaseModel` consistently — not dataclasses.

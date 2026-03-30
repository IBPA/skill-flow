# Analysis

Post-hoc analysis tools for SkillFlow evaluation results, retrieval comparisons, and paper asset generation.

## Directory Structure

```
analysis/
├── comparison/                    # Cross-run and cross-condition comparisons
│   ├── compare_conditions.py      # Multi-condition aggregate comparison (pass rate, reward, pairwise deltas)
│   ├── compare_retrievers.py      # Per-task win/loss between two retriever eval reports
│   ├── compare_runs.py            # Per-task win/loss between two Harbor evaluation runs
│   └── utils/
│       ├── loader.py              # Result loading, discovery, and pricing helpers
│       ├── display_conditions.py  # Plain-text display for multi-condition comparison
│       └── display_stats.py       # CI-enhanced display with significance stars
├── results/                       # Paper table and figure generators
│   ├── t1_generate_results.py     # T1: main results table
│   ├── t2_generate_adoption.py    # T2: skill adoption rates
│   ├── t3_generate_retrieval_stages.py  # T3: retrieval stage metrics
│   ├── t4_generate_stage_ablation.py    # T4: stage ablation
│   ├── t5_generate_latency.py     # T5: latency breakdown
│   ├── t6_generate_corpus_stats.py      # T6: corpus statistics
│   ├── t7_generate_query_examples.py    # T7: query generation examples
│   ├── t8_9_generate_retriever_comparison.py   # T8-9: retriever + query config
│   ├── t10_11_generate_reranker_comparison.py  # T10-11: reranker + deep reranker
│   ├── t12_generate_excluded_tasks.py   # T12: excluded tasks
│   ├── t13_14_15_generate_case_studies.py      # T13-15: case study tables
│   ├── f2_plot_quality_proxies.py       # F2: quality proxy comparison
│   ├── f3_plot_query_impact.py          # F3: multi-query impact (2x2 grid)
│   ├── f4_plot_skill_dist.py            # F4: skill content distribution
│   ├── generate-paper-assets.sh         # Regenerate all tables and/or figures
│   └── utils/                     # Shared helpers (LaTeX formatting, fusion, quality metrics)
├── stats/                         # Statistical utilities
│   ├── types.py                   # Pydantic models (ConfidenceInterval, PairedTestResult, etc.)
│   ├── bootstrap.py               # Bootstrap CIs and paired hypothesis tests
│   ├── proportions.py             # Wilson interval, McNemar's test, Cohen's h
│   ├── benchmark_stats.py         # Bridge: benchmark ConditionResults -> stats
│   └── retrieval_stats.py         # Bridge: retrieval EvalReport -> stats
└── utils/
    └── find_skill_patterns.py     # Scan trajectories for SKILL.md references
```

## Comparison

**compare_conditions.py** -- Aggregate comparison across multiple experimental conditions (e.g. baseline vs skillflow vs golden). Auto-discovers conditions from job directories or accepts explicit labels.

```bash
uv run python -m analysis.comparison.compare_conditions outputs/evaluation

uv run python -m analysis.comparison.compare_conditions outputs/evaluation \
    --conditions baseline skillflow-inject skillsbench-inject \
    --labels baseline skillflow skillsbench
```

**compare_retrievers.py** -- Per-task win/loss comparison between two retriever eval reports. Outputs summary metrics, win/loss/tie counts, top deltas, and plots (RR scatter, per-task delta bar chart).

```bash
uv run python -m analysis.comparison.compare_retrievers \
    outputs/experiments/retriever-comparison/bge-m3.json \
    outputs/experiments/retriever-comparison/bm25-bge-base.json \
    --k 10 --top-n 10 --output-dir outputs/compare
```

**compare_runs.py** -- Per-task comparison between two Harbor evaluation runs. Shows aggregate pass rates, per-task win/loss/tie, and lists improved/regressed/errored tasks.

```bash
uv run python -m analysis.comparison.compare_runs \
    outputs/evaluation/run-a \
    outputs/evaluation/run-b \
    --label-a baseline --label-b skillflow
```

## Paper Assets

Regenerate all tables and figures written to `paper/tables/` and `paper/figures/`:

```bash
bash analysis/results/generate-paper-assets.sh            # both
bash analysis/results/generate-paper-assets.sh --tables    # tables only
bash analysis/results/generate-paper-assets.sh --figures   # figures only
```

## Utilities

**find_skill_patterns.py** -- Scan agent trajectories for SKILL.md file references to understand how agents load and use skills.

```bash
uv run python analysis/utils/find_skill_patterns.py TRAJECTORY_DIR
```

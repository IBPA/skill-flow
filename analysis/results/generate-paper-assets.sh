#!/usr/bin/env bash
# Regenerate paper tables and/or figures from analysis data.
#
# Usage:
#   bash analysis/results/generate-paper-assets.sh            # both
#   bash analysis/results/generate-paper-assets.sh --tables    # tables only
#   bash analysis/results/generate-paper-assets.sh --figures   # figures only
set -euo pipefail

DO_TABLES=false
DO_FIGURES=false

if [[ $# -eq 0 ]]; then
    DO_TABLES=true
    DO_FIGURES=true
else
    for arg in "$@"; do
        case "$arg" in
            --tables)  DO_TABLES=true ;;
            --figures) DO_FIGURES=true ;;
            *)
                echo "Usage: $0 [--tables] [--figures]"
                exit 1
                ;;
        esac
    done
fi

# ------------------------------------------------------------------
# Tables → paper/tables/
# ------------------------------------------------------------------
if $DO_TABLES; then
    echo "Generating paper tables..."

    uv run python -m analysis.results.t1_generate_results
    uv run python -m analysis.results.t2_generate_adoption
    uv run python -m analysis.results.t3_generate_retrieval_stages
    uv run python -m analysis.results.t4_generate_stage_ablation
    uv run python -m analysis.results.t5_generate_latency \
        --run-dir outputs/pipeline/skillsbench/latency-run
    uv run python -m analysis.results.t6_generate_corpus_stats
    uv run python -m analysis.results.t7_generate_query_examples
    uv run python -m analysis.results.t8_9_generate_retriever_comparison
    uv run python -m analysis.results.t10_11_generate_reranker_comparison
    uv run python -m analysis.results.t12_generate_excluded_tasks
    uv run python -m analysis.results.t13_14_15_generate_case_studies

    echo "Done. Tables written to paper/tables/"
fi

# ------------------------------------------------------------------
# Figures → paper/figures/
# ------------------------------------------------------------------
if $DO_FIGURES; then
    echo "Generating paper figures..."

    # F2: quality proxy comparison (oracle vs. community skills)
    uv run python -m analysis.results.f2_plot_quality_proxies

    # F3 (composite): multi-query impact (recall curves + delta summary)
    # Default reports are built into the script; override with --report if needed.
    uv run python -m analysis.results.f3_plot_query_impact

    uv run python -m analysis.results.f4_plot_skill_dist

    echo "Done. Figures written to paper/figures/"
fi

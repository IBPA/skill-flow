#!/bin/bash
# Stage-1 retrieval eval for Octen-Embedding-8B (author-intended prompts) at
# 1 and 5 generated queries, on the SkillsBench retrieval set. Reuses the
# prebuilt outputs/indices/octen-8b/ index and the cached generated queries,
# so it makes no OpenAI calls and rebuilds nothing.

# Reduce CUDA fragmentation so the 8B model + GT-injection encode fits on one
# 24 GB card (paired with batch_size=8 in the config).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python scripts/run-retriever-experiment.py \
  --config skill_flow/config/experiments/retriever-octen.json

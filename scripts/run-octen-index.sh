#!/bin/bash

uv run python scripts/build-index-multigpu.py \
  --model Octen/Octen-Embedding-8B \
  --revision 5adcfa292e712091dfc30f0e97f0b2282e6cc66c \
  --output-dir outputs/indices/octen-8b/ \
  --corpus-path data/skills/ \
  --doc-prompt "- " --max-seq-length 512 --batch-size 32 \
  --devices cuda:0,cuda:1,cuda:3

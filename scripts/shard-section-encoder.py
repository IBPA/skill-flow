"""Encode one shard of a section's text under a chosen encoder.

This is the per-shard worker used to parallelise the structure-aware ``code``
(or any other) section build across multiple GPUs.  Each invocation:

1. Loads the full corpus, takes a contiguous slice based on ``--shard-index``
   and ``--n-shards`` (so order is deterministic and shards do not overlap).
2. Splits each SKILL.md into ``(yaml, prose, code)`` and selects ``--section``.
3. Encodes the section text with the given encoder and writes per-shard
   intermediates under ``--shard-dir``:

   * ``embeddings.npy`` — ``(N_shard, dim)`` L2-normalised float32 matrix
   * ``skill_ids.json``  — ordered list of skill keys
   * ``skill_section_texts.json`` — key -> section text (post sentinel
     substitution)

Use :mod:`scripts.merge-section-shards` afterwards to concatenate shards into
the final section index.

Usage::

    CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.shard-section-encoder \\
        --section code \\
        --shard-index 0 --n-shards 2 \\
        --model BAAI/bge-code-v1 \\
        --revision bd67852057c5d7ddcc7b8234d9d6c410117ed851 \\
        --batch-size 8 --max-seq-length 2048 \\
        --shard-dir outputs/indices/section-mixed/_shards/code_0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from skill_flow.config import RetrieverConfig
from skill_flow.corpus.loader import load_content, load_corpus
from skill_flow.corpus.splitter import safe_section_text, split_skill_sections
from skill_flow.index.encoder import Encoder

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--section",
        choices=("yaml", "prose", "code"),
        required=True,
    )
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="")
    ap.add_argument(
        "--query-prompt",
        default="Represent this sentence for searching relevant passages: ",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=0)
    ap.add_argument("--corpus-path", type=Path, default=Path("data/skills/"))
    ap.add_argument("--shard-dir", type=Path, required=True)
    args = ap.parse_args()

    args.shard_dir.mkdir(parents=True, exist_ok=True)

    skills = load_corpus(args.corpus_path)
    total = len(skills)
    chunk = (total + args.n_shards - 1) // args.n_shards
    start = args.shard_index * chunk
    end = min(start + chunk, total)
    my = skills[start:end]
    logger.info(
        "shard %d/%d  skills %d/%d  slice [%d:%d]",
        args.shard_index,
        args.n_shards,
        len(my),
        total,
        start,
        end,
    )

    texts: list[str] = []
    keys: list[str] = []
    section_texts: dict[str, str] = {}
    for s in my:
        try:
            content = load_content(args.corpus_path, s)
        except FileNotFoundError:
            logger.warning("SKILL.md not found for %s, skipping", s.key)
            continue
        yaml, prose, code = split_skill_sections(content)
        pick = {"yaml": yaml, "prose": prose, "code": code}[args.section]
        safe = safe_section_text(pick)
        texts.append(safe)
        keys.append(s.key)
        section_texts[s.key] = safe

    rc = RetrieverConfig(
        model_name=args.model,
        revision=args.revision,
        query_prompt=args.query_prompt,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    encoder = Encoder(rc)
    logger.info(
        "shard %d: encoding %d %s sections (batch=%d, max_seq=%d) ...",
        args.shard_index,
        len(texts),
        args.section,
        args.batch_size,
        args.max_seq_length,
    )
    embeddings = encoder.encode_documents(texts, batch_size=args.batch_size)

    np.save(args.shard_dir / "embeddings.npy", embeddings)
    (args.shard_dir / "skill_ids.json").write_text(json.dumps(keys), encoding="utf-8")
    (args.shard_dir / "skill_section_texts.json").write_text(
        json.dumps(section_texts), encoding="utf-8"
    )
    logger.info(
        "shard %d DONE OK -- %s -> %s",
        args.shard_index,
        embeddings.shape,
        args.shard_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Concatenate per-shard section index artifacts into a single section index.

Counterpart to :mod:`scripts.shard-section-encoder`. Reads each shard directory
under ``--shards-parent`` in lexical order (``code_0``, ``code_1``, …) and
writes the consolidated section sub-index at ``--output-dir`` in the same
layout :func:`skill_flow.index.section_builder._write_subdir` produces:

* ``faiss.index`` — ``IndexFlatIP`` over the concatenated embeddings
* ``embeddings.npy`` — ``(N, dim)`` L2-normalised float32 matrix
* ``skill_ids.json``  — ordered list of skill keys
* ``skill_section_texts.json``

Usage::

    uv run python -m scripts.merge-section-shards \\
        --shards-parent outputs/indices/section-mixed/_shards \\
        --shard-prefix code_ \\
        --output-dir outputs/indices/section-mixed/code/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards-parent", type=Path, required=True)
    ap.add_argument(
        "--shard-prefix",
        required=True,
        help="lexical prefix of each shard subdir (e.g. 'code_')",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(
        d
        for d in args.shards_parent.iterdir()
        if d.is_dir() and d.name.startswith(args.shard_prefix)
    )
    if not shard_dirs:
        msg = (
            f"no shards found under {args.shards_parent} "
            f"with prefix {args.shard_prefix!r}"
        )
        raise FileNotFoundError(msg)

    logger.info("merging %d shards: %s", len(shard_dirs), [d.name for d in shard_dirs])
    emb_parts: list[np.ndarray] = []
    keys: list[str] = []
    section_texts: dict[str, str] = {}
    for d in shard_dirs:
        emb_parts.append(np.load(d / "embeddings.npy"))
        keys.extend(json.loads((d / "skill_ids.json").read_text(encoding="utf-8")))
        texts_path = d / "skill_section_texts.json"
        if texts_path.exists():
            section_texts.update(json.loads(texts_path.read_text(encoding="utf-8")))

    embeddings = np.concatenate(emb_parts, axis=0)
    if embeddings.shape[0] != len(keys):
        msg = (
            f"shard alignment broken: {embeddings.shape[0]} vectors vs {len(keys)} keys"
        )
        raise RuntimeError(msg)

    dim = int(embeddings.shape[1])
    logger.info("building FAISS IndexFlatIP from %d vectors (dim=%d)", len(keys), dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    np.save(args.output_dir / "embeddings.npy", embeddings)
    faiss.write_index(index, str(args.output_dir / "faiss.index"))
    (args.output_dir / "skill_ids.json").write_text(json.dumps(keys), encoding="utf-8")
    (args.output_dir / "skill_section_texts.json").write_text(
        json.dumps(section_texts), encoding="utf-8"
    )
    logger.info("merged shards -> %s (ntotal=%d)", args.output_dir, index.ntotal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

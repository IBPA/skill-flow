"""Build a dense FAISS index across multiple GPUs (one model replica per GPU).

The in-repo ``skill_flow.index.builder.build_index`` encodes on a single
device. For a large encoder (e.g. Octen-Embedding-8B) we want to spread the
~36K description encodes across several GPUs, so this driver uses
sentence-transformers' multi-process pool and then writes the same artifacts
that ``build_index`` does, minus ``skill_contents.json`` — the retriever loads
content lazily (``IndexSearcher`` guards on its existence) and content is
encoder-independent, so a stage-1 retrieval eval does not need it.

Usage:
    uv run python scripts/build-index-multigpu.py \
        --model Octen/Octen-Embedding-8B \
        --revision 5adcfa292e712091dfc30f0e97f0b2282e6cc66c \
        --output-dir outputs/indices/octen-8b/ \
        --corpus-path data/skills/ \
        --doc-prompt "- " \
        --max-seq-length 512 \
        --batch-size 32 \
        --devices cuda:0,cuda:1,cuda:3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from skill_flow.corpus.loader import load_corpus

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build-index-multigpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model id for the encoder")
    p.add_argument("--revision", default="", help="HF commit to pin (reproducibility)")
    p.add_argument("--output-dir", required=True, help="Index artifact directory")
    p.add_argument("--corpus-path", default="data/skills/", help="Corpus directory")
    p.add_argument("--doc-prompt", default="", help="Prefix prepended to each document")
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=0,
        help="Cap model max_seq_length (0 = leave default)",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Per-process batch size")
    p.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:3",
        help="Comma-separated CUDA devices for the encode pool",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading corpus from %s ...", args.corpus_path)
    skills = load_corpus(Path(args.corpus_path))
    keys = [s.key for s in skills]
    descriptions = [s.description for s in skills]
    texts = (
        [args.doc_prompt + d for d in descriptions] if args.doc_prompt else descriptions
    )
    logger.info("Encoding %d descriptions on %s ...", len(texts), devices)

    # Load the main replica on CPU so it does not pin VRAM on one GPU; the pool
    # spawns one full replica per target device.
    model = SentenceTransformer(
        args.model, device="cpu", revision=args.revision or None
    )
    if args.max_seq_length > 0:
        model.max_seq_length = args.max_seq_length

    t0 = time.time()
    pool = model.start_multi_process_pool(target_devices=devices)
    try:
        embeddings = model.encode_multi_process(
            texts,
            pool,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    finally:
        model.stop_multi_process_pool(pool)
    emb = np.asarray(embeddings, dtype=np.float32)
    logger.info(
        "Encoded %d x %d in %.1fs (%.1f docs/s)",
        emb.shape[0],
        emb.shape[1],
        time.time() - t0,
        emb.shape[0] / max(time.time() - t0, 1e-9),
    )

    logger.info("Building FAISS IndexFlatIP (dim=%d) ...", emb.shape[1])
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    np.save(out_dir / "embeddings.npy", emb)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    (out_dir / "skill_ids.json").write_text(json.dumps(keys), encoding="utf-8")
    desc_map = dict(zip(keys, descriptions, strict=True))
    (out_dir / "skill_descriptions.json").write_text(
        json.dumps(desc_map), encoding="utf-8"
    )
    logger.info(
        "Index built: %d vectors, dim=%d -> %s", index.ntotal, emb.shape[1], out_dir
    )


if __name__ == "__main__":
    main()

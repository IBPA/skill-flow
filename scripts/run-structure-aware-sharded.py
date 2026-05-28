"""Build the structure-aware experiment's indices across multiple GPUs.

Unlike ``scripts/run-reranker-sharded.py`` (which shards 87 tasks across
GPUs), this driver shards **encoding jobs** across GPUs — there are ~6
distinct (encoder, target_dir) pairs implied by
``skill_flow/config/experiments/structure-aware.json``, and each is a
single-GPU subprocess pinned via ``CUDA_VISIBLE_DEVICES``.

After all index dirs are populated, run the eval:

    uv run python -m skill_flow.cli experiment \
        --config skill_flow/config/experiments/structure-aware.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WHOLE_JOB_SCRIPT = """
from pathlib import Path
from skill_flow.config import RetrieverConfig
from skill_flow.corpus.loader import load_corpus
from skill_flow.index.builder import build_index
from skill_flow.index.encoder import Encoder

rc = RetrieverConfig(
    model_name={model_name!r},
    query_prompt={query_prompt!r},
    revision={revision!r},
    batch_size={batch_size},
)
encoder = Encoder(rc)
build_index(
    load_corpus(Path({corpus_path!r})),
    encoder,
    Path({output_dir!r}),
    batch_size={batch_size},
    corpus_path=Path({corpus_path!r}),
    max_content_tokens={max_content_tokens},
)
"""

_SECTION_JOB_SCRIPT = """
from pathlib import Path
from skill_flow.config import RetrieverConfig
from skill_flow.corpus.loader import load_corpus
from skill_flow.index.encoder import Encoder
from skill_flow.index.section_builder import build_section_indices

rc = RetrieverConfig(
    model_name={model_name!r},
    query_prompt={query_prompt!r},
    revision={revision!r},
    batch_size={batch_size},
)
encoder = Encoder(rc)
build_section_indices(
    load_corpus(Path({corpus_path!r})),
    Path({parent_dir!r}),
    {{ {section!r}: encoder }},
    Path({corpus_path!r}),
    batch_size={batch_size},
    sections=({section!r},),
)
"""


def _collect_jobs(cfg: dict[str, Any], corpus_path: str) -> list[dict[str, Any]]:
    """Return a deduplicated list of (encoder, target_dir) build jobs."""
    seen: dict[str, dict[str, Any]] = {}
    for v in cfg["retrievers"]:
        if not v.get("enabled", True):
            continue
        rtype = v.get("retriever_type", "dense")
        if rtype == "dense":
            target = v["index_dir"]
            if target in seen:
                continue
            seen[target] = {
                "kind": "whole",
                "model_name": v["model_name"],
                "query_prompt": v.get(
                    "query_prompt",
                    "Represent this sentence for searching relevant passages: ",
                ),
                "revision": v.get("revision", ""),
                "batch_size": v.get("batch_size", 256),
                "max_content_tokens": v.get("max_content_tokens", 0),
                "output_dir": target,
                "corpus_path": corpus_path,
            }
        elif rtype == "section":
            for section, sec in v["sections"].items():
                target = sec["index_dir"]
                if target in seen:
                    continue
                parent = str(Path(target).parent)
                seen[target] = {
                    "kind": "section",
                    "section": section,
                    "model_name": sec["model_name"],
                    "query_prompt": sec.get(
                        "query_prompt",
                        "Represent this sentence for searching relevant passages: ",
                    ),
                    "revision": sec.get("revision", ""),
                    "batch_size": sec.get("batch_size", 256),
                    "parent_dir": parent,
                    "corpus_path": corpus_path,
                }
    return list(seen.values())


def _already_built(job: dict[str, Any]) -> bool:
    if job["kind"] == "whole":
        return bool((Path(job["output_dir"]) / "faiss.index").exists())
    section_dir = Path(job["parent_dir"]) / job["section"]
    return bool((section_dir / "faiss.index").exists())


def _render_script(job: dict[str, Any]) -> str:
    if job["kind"] == "whole":
        return _WHOLE_JOB_SCRIPT.format(**job)
    return _SECTION_JOB_SCRIPT.format(**job)


def _launch_job(
    job: dict[str, Any], gpu_id: int, log_dir: Path
) -> subprocess.Popen[bytes]:
    log_dir.mkdir(parents=True, exist_ok=True)
    target = job.get("output_dir") or f"{job['parent_dir']}/{job['section']}"
    safe = target.replace("/", "_").strip("_")
    log_path = log_dir / f"{safe}.log"
    code = _render_script(job)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
    }
    with log_path.open("w") as logf:
        return subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )


def _run_pool(jobs: list[dict[str, Any]], gpus: list[int], log_dir: Path) -> int:
    """Run *jobs* with a worker pool of GPUs. Returns 0 on full success."""
    pending = list(jobs)
    in_flight: dict[int, tuple[subprocess.Popen[bytes], dict[str, Any]]] = {}
    failures = 0
    t0 = time.time()

    def _drain_one_finished() -> None:
        nonlocal failures
        while True:
            for gpu_id, (proc, job) in list(in_flight.items()):
                rc = proc.poll()
                if rc is not None:
                    target = (
                        job.get("output_dir") or f"{job['parent_dir']}/{job['section']}"
                    )
                    status = "OK" if rc == 0 else f"FAIL rc={rc}"
                    logger.info(
                        "[gpu %d] %s  %s  (%.1fs since start)",
                        gpu_id,
                        status,
                        target,
                        time.time() - t0,
                    )
                    if rc != 0:
                        failures += 1
                    del in_flight[gpu_id]
                    return
            time.sleep(2.0)

    for gpu in gpus:
        if not pending:
            break
        job = pending.pop(0)
        proc = _launch_job(job, gpu, log_dir)
        in_flight[gpu] = (proc, job)

    while in_flight:
        _drain_one_finished()
        while pending and len(in_flight) < len(gpus):
            free_gpus = [g for g in gpus if g not in in_flight]
            if not free_gpus:
                break
            gpu = free_gpus[0]
            job = pending.pop(0)
            proc = _launch_job(job, gpu, log_dir)
            in_flight[gpu] = (proc, job)

    return failures


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument(
        "--log-dir",
        default=Path("outputs/experiments/structure-aware/_build_logs"),
        type=Path,
    )
    ap.add_argument("--force", action="store_true", help="Rebuild existing indices.")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    corpus_path = cfg.get("corpus_path", "data/skills/")
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]

    jobs = _collect_jobs(cfg, corpus_path)
    if not args.force:
        skipped = [j for j in jobs if _already_built(j)]
        jobs = [j for j in jobs if not _already_built(j)]
        for j in skipped:
            target = j.get("output_dir") or f"{j['parent_dir']}/{j['section']}"
            logger.info("skip (exists): %s", target)

    if not jobs:
        logger.info("nothing to build.")
        return 0

    logger.info("Running %d jobs across %d GPUs (%s)", len(jobs), len(gpus), gpus)
    for j in jobs:
        target = j.get("output_dir") or f"{j['parent_dir']}/{j['section']}"
        logger.info("  queued: %s  [%s]", target, j["model_name"])
    fails = _run_pool(jobs, gpus, args.log_dir)
    if fails:
        logger.error("%d job(s) failed; see %s for logs", fails, args.log_dir)
        return 1
    logger.info("all jobs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run a reranker experiment sharded across multiple GPUs.

Each shard is a single-GPU subprocess (``CUDA_VISIBLE_DEVICES=i``) over a
disjoint subset of tasks. After all shards finish, per-shard reports are
merged into one report compatible with the non-sharded output. Per-task
latency reported is the wall-clock of a single shard divided by its task
count, i.e. the single-GPU per-task time a deployment would see; multi-GPU
usage only reduces wall-clock and does not contaminate the latency number.

Usage:
    uv run python scripts/run-reranker-sharded.py \\
        --config skill_flow/config/experiments/reranker-qwen3.json \\
        --gpus 1,2,3 \\
        --out-dir outputs/experiments/reranker-comparison/sharded-qwen3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_SHARD_SCRIPT = """
import json, time
from skill_flow.config import RerankerExperimentConfig
from skill_flow.eval.experiments.reranker import run_reranker_experiment
cfg = RerankerExperimentConfig(**json.loads(open({cfg_path!r}).read()))
t0 = time.time()
run_reranker_experiment(cfg)
print(f"SHARD_TIME {{time.time() - t0:.2f}} TASKS {ntasks}")
"""


def _task_ids(input_report: Path) -> list[str]:
    tr = json.loads(input_report.read_text())["task_results"]
    items = tr if isinstance(tr, list) else list(tr.values())
    return [t["task_id"] for t in items]


def _launch_shard(
    base_cfg: dict,
    gpu_id: int,
    task_ids: list[str],
    shard_dir: Path,
) -> tuple[subprocess.Popen[bytes], Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_cfg)
    cfg["include_tasks"] = task_ids
    cfg["output_dir"] = str(shard_dir)
    cfg_path = shard_dir / "_input_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    code = _SHARD_SCRIPT.format(cfg_path=str(cfg_path), ntasks=len(task_ids))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "TRANSFORMERS_VERBOSITY": "error",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
    }
    log_path = shard_dir / "shard.log"
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return proc, log_path


def _shard_report_file(shard_dir: Path) -> Path:
    """The reranker report file in a shard dir (ignores _input_config.json)."""
    candidates = [p for p in shard_dir.glob("*.json") if not p.name.startswith("_")]
    if len(candidates) != 1:
        msg = f"expected 1 report in {shard_dir}, found {len(candidates)}: {candidates}"
        raise RuntimeError(msg)
    return candidates[0]


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _merge(shard_reports: list[dict]) -> dict:
    tasks: list[dict] = []
    for r in shard_reports:
        tr = r["task_results"]
        tasks.extend(tr if isinstance(tr, list) else list(tr.values()))
    ks = list(tasks[0]["recall_at"].keys())
    metrics = {
        "mean_recall_at": {k: _mean([t["recall_at"][k] for t in tasks]) for k in ks},
        "mean_precision_at": {
            k: _mean([t["precision_at"][k] for t in tasks]) for k in ks
        },
        "mean_hit_at": {k: _mean([t["hit_at"][k] for t in tasks]) for k in ks},
        "mrr": _mean([t["reciprocal_rank"] for t in tasks]),
    }
    base_summary = shard_reports[0]["summary"]
    summary = {
        **{
            k: base_summary[k]
            for k in ("num_tasks_total", "num_skills_injected", "num_tasks_no_skills")
            if k in base_summary
        },
        "num_tasks_evaluated": len(tasks),
        **metrics,
    }
    return {
        "summary": summary,
        "task_results": tasks,
        "config": shard_reports[0].get("config"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--gpus",
        required=True,
        help="comma-separated CUDA device IDs (e.g. '1,2,3')",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    base_cfg = json.loads(args.config.read_text())
    tasks = _task_ids(Path(base_cfg["input_report_path"]))
    shards = [tasks[i :: len(gpus)] for i in range(len(gpus))]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sharding {len(tasks)} tasks across GPUs {gpus}: {[len(s) for s in shards]}")
    t0 = time.time()
    procs: list[tuple[subprocess.Popen[bytes], Path, Path]] = []
    for gpu, shard_tasks in zip(gpus, shards, strict=True):
        shard_dir = args.out_dir / f"gpu{gpu}"
        proc, log = _launch_shard(base_cfg, gpu, shard_tasks, shard_dir)
        procs.append((proc, log, shard_dir))

    failed: list[int] = []
    for proc, log, _ in procs:
        rc = proc.wait()
        if rc != 0:
            failed.append(rc)
            print(f"shard FAILED (rc={rc}); log: {log}", file=sys.stderr)
    if failed:
        return 1
    wall = time.time() - t0

    shard_times: list[float] = []
    reports: list[dict] = []
    for _, log, shard_dir in procs:
        for line in log.read_text().splitlines():
            if line.startswith("SHARD_TIME"):
                shard_times.append(float(line.split()[1]))
        reports.append(json.loads(_shard_report_file(shard_dir).read_text()))

    merged = _merge(reports)
    label = _shard_report_file(procs[0][2]).stem
    merged_path = args.out_dir / f"{label}.json"
    merged_path.write_text(json.dumps(merged, indent=2))

    per_task = [st / max(len(shards[i]), 1) for i, st in enumerate(shard_times)]
    print(f"merged report -> {merged_path}")
    print(
        f"WALL {wall:.1f}s (multi-GPU, {len(gpus)}x) | "
        f"per-shard wall {[round(t, 1) for t in shard_times]} | "
        f"per-task (single-GPU) {[round(t, 2) for t in per_task]}s "
        f"(mean {statistics.mean(per_task):.2f}s)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

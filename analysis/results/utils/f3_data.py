"""Constants, types, and data-loading helpers for the multi-query impact figure (F3)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

STAGE_ORDER = ["Retriever", "Shallow Reranker", "Deep Reranker"]
RECALL_KS = [1, 5, 10, 50, 100, 500, 1000]
OUTPUT_K: dict[str, int] = {
    "Retriever": 1000,
    "Shallow Reranker": 100,
    "Deep Reranker": 10,
}

M_STYLE: dict[int, dict[str, str]] = {
    1: {"color": "#0072B2", "marker": "o", "ls": "-"},
    3: {"color": "#E69F00", "marker": "s", "ls": "--"},
    5: {"color": "#009E73", "marker": "D", "ls": "-."},
}
M_LABEL: dict[int, str] = {m: f"$M{{=}}{m}$" for m in M_STYLE}

_STAGE_KEY: dict[str, str] = {
    "retriever": "Retriever",
    "reranker": "Shallow Reranker",
    "shallow": "Shallow Reranker",
    "deep": "Deep Reranker",
}

# -------------------------------------------------------------------
# Types
# -------------------------------------------------------------------

ReportData = dict[str, object]
StageData = dict[str, dict[int, ReportData]]

# -------------------------------------------------------------------
# Data helpers
# -------------------------------------------------------------------


def _load_report(path: Path) -> ReportData:
    data: ReportData = json.loads(path.read_text("utf-8"))
    return data


def summ(report: ReportData) -> dict[str, object]:
    """Extract the summary dict from a full report."""
    val = report["summary"]
    if not isinstance(val, dict):
        return {}
    return val


def tasks(report: ReportData) -> list[dict[str, object]]:
    """Extract task_results list from a full report."""
    val = report.get("task_results", [])
    if not isinstance(val, list):
        return []
    return val


def recall_curve(summary: dict[str, object]) -> dict[int, float]:
    """Extract mean_recall_at mapping from a summary."""
    mapping = summary.get("mean_recall_at", {})
    if not isinstance(mapping, dict):
        return {}
    return {int(k): float(v) for k, v in mapping.items()}


def recall_at(summary: dict[str, object], k: int) -> float:
    """Get recall@k from a summary."""
    return recall_curve(summary).get(k, 0.0)


def mrr(summary: dict[str, object]) -> float:
    """Get MRR from a summary."""
    val = summary.get("mrr", 0.0)
    return float(val) if isinstance(val, (int, float)) else 0.0


def parse_reports(raw: list[list[str]] | None) -> StageData:
    """Parse ``--report STAGE M PATH`` triples into nested dict."""
    if not raw:
        return {}
    out: StageData = {}
    for stage_raw, m_raw, path_raw in raw:
        stage = _STAGE_KEY.get(stage_raw.lower(), stage_raw)
        m = int(m_raw)
        out.setdefault(stage, {})[m] = _load_report(Path(path_raw))
        logger.info("Loaded %s M=%d <- %s", stage, m, path_raw)
    return out


# -------------------------------------------------------------------
# Bootstrap helpers
# -------------------------------------------------------------------

_N_BOOT = 10_000
_SEED = 42


def task_recall_at_k(report: ReportData, k: int) -> np.ndarray:
    """Extract per-task recall@k as an array."""
    vals: list[float] = []
    for t in tasks(report):
        recall = t["recall_at"]
        val = recall.get(str(k), 0.0) if isinstance(recall, dict) else 0.0
        vals.append(float(val) if isinstance(val, (int, float)) else 0.0)
    return np.array(vals)


def bootstrap_delta_ci(
    base_vals: np.ndarray,
    test_vals: np.ndarray,
) -> tuple[float, float, float, float]:
    """Bootstrap 95% CI for relative delta (%) of means.

    Returns (point_estimate, ci_lo, ci_hi, p_value).
    p_value is two-tailed: fraction of bootstrap replicates on
    the opposite side of zero from the point estimate, doubled.
    """
    rng = np.random.default_rng(_SEED)
    n = len(base_vals)
    deltas = np.empty(_N_BOOT)
    for b in range(_N_BOOT):
        idx = rng.integers(0, n, size=n)
        mb = base_vals[idx].mean()
        mt = test_vals[idx].mean()
        deltas[b] = (mt - mb) / mb * 100 if mb > 0 else 0.0
    point = (test_vals.mean() - base_vals.mean()) / base_vals.mean() * 100
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    p_tail = float(np.mean(deltas >= 0)) if point < 0 else float(np.mean(deltas <= 0))
    p_val = min(2.0 * p_tail, 1.0)
    return point, lo, hi, p_val


def holm_bonferroni(raw_pvals: dict[str, float]) -> dict[str, float]:
    """Apply Holm-Bonferroni correction to a dict of raw p-values."""
    m = len(raw_pvals)
    sorted_keys = sorted(raw_pvals, key=raw_pvals.__getitem__)
    adjusted: dict[str, float] = {}
    max_so_far = 0.0
    for rank_0, key in enumerate(sorted_keys):
        adj = min(1.0, raw_pvals[key] * (m - rank_0))
        max_so_far = max(max_so_far, adj)
        adjusted[key] = max_so_far
    return adjusted

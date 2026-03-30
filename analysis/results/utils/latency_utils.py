"""Load per-experiment latency data from JSON reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def load_latency_map(*dirs: Path) -> dict[str, float]:
    """Load ``mean_elapsed_ms`` from experiment JSONs, keyed by stem.

    Each directory is scanned for ``*.json`` files.  For every file whose
    ``summary.mean_elapsed_ms`` field is present, an entry
    ``{filename_stem: mean_elapsed_ms}`` is added.  Later directories
    override earlier ones when stems collide.
    """
    latency: dict[str, float] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                ms = data.get("summary", {}).get("mean_elapsed_ms")
                if ms is not None:
                    latency[p.stem] = float(ms)
            except (ValueError, KeyError, OSError):
                continue
    return latency

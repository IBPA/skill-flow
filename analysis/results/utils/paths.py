"""Output locations for generated paper assets.

Assets are written into the sibling ``skill-flow-manuscript`` repo checkout;
set ``SKILLFLOW_MANUSCRIPT_DIR`` to point elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANUSCRIPT_DIR = Path(
    os.environ.get(
        "SKILLFLOW_MANUSCRIPT_DIR", REPO_ROOT.parent / "skill-flow-manuscript"
    )
)
TABLES_DIR = MANUSCRIPT_DIR / "tables"
FIGURES_DIR = MANUSCRIPT_DIR / "figures"

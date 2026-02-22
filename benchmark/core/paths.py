"""Path utilities for evaluation scripts."""

from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root
    """
    current = Path(__file__).resolve()
    # Go up from evaluation/core/paths.py to project root
    return current.parent.parent.parent

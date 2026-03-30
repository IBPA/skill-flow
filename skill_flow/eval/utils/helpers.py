"""Shared utilities for evaluation modules."""

import re


def slug(name: str) -> str:
    """Convert a model name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

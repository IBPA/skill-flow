"""Column-wise formatting utilities for LaTeX table generation.

Applies bold/italic/underline decorations to best-in-column values,
with support for row and column inclusion/exclusion.
"""

from __future__ import annotations

import re


def _extract_number(cell: str) -> float | None:
    """Extract the leading numeric value from a LaTeX cell.

    Handles formats like ``0.634``, ``0.634{\\scriptsize~[...]}``,
    ``\\textit{19.5}``, ``16.4$^{*}$``, ``\\$0.025{...}``,
    ``24.9``, ``1404.9``, ``---``.
    """
    # Strip leading LaTeX commands (\textit{, \textbf{, \underline{, \$)
    s = re.sub(r"\\(?:textit|textbf|underline)\{", "", cell)
    s = s.replace("\\$", "")
    # Find first number (possibly negative, with decimals)
    m = re.search(r"-?\d+\.?\d*", s)
    if m is None:
        return None
    return float(m.group())


def mark_best(
    cells: list[str],
    *,
    exclude: set[int] | None = None,
    direction: str = "max",
    fmt: str = "bold",
) -> list[str]:
    """Return cells with the best value decorated.

    For each cell, the leading numeric portion is compared. The cell with
    the best value (max or min) gets its leading number wrapped in
    ``\\textbf{}`` (bold) or ``\\underline{}`` (underline).

    Args:
        cells: Cell strings for one column across rows.
        exclude: Row indices to exclude from both comparison and decoration.
        direction: ``"max"`` to bold the largest, ``"min"`` for smallest.
        fmt: ``"bold"`` for ``\\textbf{}``, ``"underline"`` for ``\\underline{}``.

    Returns:
        New list of cells with decoration applied.
    """
    if exclude is None:
        exclude = set()
    cmd = "\\textbf" if fmt == "bold" else "\\underline"

    # Parse values from eligible cells
    values: dict[int, float] = {}
    for i, cell in enumerate(cells):
        if i in exclude:
            continue
        v = _extract_number(cell)
        if v is not None:
            values[i] = v

    if not values:
        return list(cells)

    best_val = max(values.values()) if direction == "max" else min(values.values())
    best_idx = next(i for i, v in values.items() if v == best_val)

    result = list(cells)
    # Wrap the leading number in the best cell
    cell = result[best_idx]
    m = re.search(r"-?\d+\.?\d*", cell)
    if m:
        start, end = m.start(), m.end()
        result[best_idx] = cell[:start] + f"{cmd}{{{cell[start:end]}}}" + cell[end:]
    return result


def mark_best_whole(
    cells: list[str],
    *,
    exclude: set[int] | None = None,
    direction: str = "min",
) -> list[str]:
    """Bold the best cell(s), wrapping the entire value (handles ties)."""
    if exclude is None:
        exclude = set()
    values: dict[int, float] = {}
    for i, cell in enumerate(cells):
        if i in exclude:
            continue
        v = _extract_number(cell)
        if v is not None:
            values[i] = v
    if not values:
        return list(cells)
    best_val = max(values.values()) if direction == "max" else min(values.values())
    result = list(cells)
    for i, v in values.items():
        if v == best_val:
            result[i] = f"\\textbf{{{cells[i]}}}"
    return result


def mark_row_italic(cells: list[str]) -> list[str]:
    """Wrap every cell value in ``\\textit{}``."""
    result: list[str] = []
    for cell in cells:
        stripped = cell.strip()
        if stripped and stripped != "---":
            result.append(f"\\textit{{{stripped}}}")
        else:
            result.append(cell)
    return result

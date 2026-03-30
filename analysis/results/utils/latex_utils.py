"""Shared LaTeX formatting helpers for paper table generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from skill_flow.eval.models import EvalReport

if TYPE_CHECKING:
    from pathlib import Path

    from analysis.stats.types import ConfidenceInterval


def fmt_ci_pct(ci: ConfidenceInterval, decimals: int = 1) -> str:
    """Format CI as percentage: ``16.4{\\scriptsize~[13.1, 19.8]}``."""
    f = f".{decimals}f"
    m = f"{ci.mean * 100:{f}}"
    lo = f"{ci.ci_lo * 100:{f}}"
    hi = f"{ci.ci_hi * 100:{f}}"
    return f"{m}{{\\scriptsize~[{lo}, {hi}]}}"


def fmt_ci_dec(ci: ConfidenceInterval, decimals: int = 3) -> str:
    """Format CI as decimal: ``0.487{\\scriptsize~[0.42, 0.55]}``."""
    f = f".{decimals}f"
    m = f"{ci.mean:{f}}"
    lo = f"{ci.ci_lo:{f}}"
    hi = f"{ci.ci_hi:{f}}"
    return f"{m}{{\\scriptsize~[{lo}, {hi}]}}"


def _drop_leading_zero(s: str) -> str:
    """Drop leading zero from a decimal string: ``0.12`` -> ``.12``."""
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _round_ci_bound(value: float, decimals: int = 2) -> str:
    """Round a CI bound, nudging to compensate for bootstrap discretisation.

    Bootstrap CIs on discrete metrics (binary hit, precision multiples of
    0.1) land exactly on grid points like k/87.  A half-unit nudge at the
    next decimal place (5e-4 for 2-decimal rounding) prevents values that
    sit just below the rounding threshold from appearing one step too low.
    """
    nudge = 0.5 * 10 ** -(decimals + 1)
    return f"{value + nudge:.{decimals}f}"


def fmt_ci_abbrev(ci: ConfidenceInterval, decimals: int = 3) -> str:
    """Format CI with abbreviated bounds (2-decimal, with leading zero).

    Example: ``0.174{\\scriptsize~[0.12, 0.24]}``
    """
    m = f"{ci.mean:.{decimals}f}"
    lo = _round_ci_bound(ci.ci_lo)
    hi = _round_ci_bound(ci.ci_hi)
    return f"{m}{{\\scriptsize~[{lo}, {hi}]}}"


def fmt_ci_cost(ci: ConfidenceInterval) -> str:
    """Format CI as dollar cost: ``\\$0.025{\\scriptsize~[0.023, 0.027]}``."""
    m = f"{ci.mean:.3f}"
    lo = f"{ci.ci_lo:.3f}"
    hi = f"{ci.ci_hi:.3f}"
    return f"\\${m}{{\\scriptsize~[{lo}, {hi}]}}"


def load_report(path: Path) -> EvalReport | None:
    """Load an EvalReport from a JSON file, returning None if missing."""
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalReport.model_validate(data)


def load_reports_from_dir(directory: Path) -> dict[str, EvalReport]:
    """Load all EvalReport JSON files from a directory, keyed by stem."""
    reports: dict[str, EvalReport] = {}
    if not directory.is_dir():
        return reports
    for p in sorted(directory.glob("*.json")):
        try:
            report = EvalReport.model_validate_json(
                p.read_text(encoding="utf-8"),
            )
        except (ValueError, KeyError, OSError):
            continue
        reports[p.stem] = report
    return reports


def write_or_print(lines: list[str], output: Path | None) -> None:
    """Write lines to a file or print to stdout."""
    text = "\n".join(lines) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Written to {output}")
    else:
        print(text)


def table_env(
    col_spec: str,
    header: str,
    *,
    resize: bool = True,
    extra_preamble: str = "",
) -> tuple[list[str], list[str]]:
    """Return (opening lines, closing lines) for tabular content only.

    The ``\\begin{table}`` wrapper, caption, and label live in
    ``main.tex``; generated files contain only the tabular body.
    """
    top: list[str] = []
    if resize:
        top.append(r"\resizebox{\columnwidth}{!}{%")
    if extra_preamble:
        top.append(extra_preamble)
    top.extend(
        [
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"  \toprule",
            f"  {header} \\\\",
            r"  \midrule",
        ]
    )
    bot = (
        [r"  \bottomrule", r"\end{tabular}%"]
        if resize
        else [
            r"  \bottomrule",
            r"\end{tabular}",
        ]
    )
    if resize:
        bot.append(r"}")
    return top, bot


def ci_cells(
    cis: dict[str, ConfidenceInterval],
    ks: list[int],
    prefix: str,
) -> list[str]:
    """Format CI cells for a list of k values, returning ``---`` for missing."""
    return [
        fmt_ci_dec(cis[f"{prefix}@{k}"]) if f"{prefix}@{k}" in cis else "---"
        for k in ks
    ]


def available_ks(
    reports: dict[str, EvalReport],
    ks: list[int],
) -> list[int]:
    """Filter ks to those present in at least one report."""
    present: set[int] = set()
    for r in reports.values():
        present.update(r.summary.mean_recall_at)
    return [k for k in ks if k in present]

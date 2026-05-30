"""Corpus functionality audit: structural distribution + sampled LLM judge.

Two layers answer "what fraction of community skills are functional":

* **Structural (full corpus, objective)** — for every skill, whether it is
  *code-bearing* (bundles an executable script or contains a fenced code
  block) plus supporting proxies (bundled-file count, code fraction). These
  reproduce and extend the Figure 2 proxies over the whole corpus.
* **Judged (stratified sample)** — an LLM rates each sampled skill in context
  (SKILL.md + its bundle listing) on three axes: *code-sound* (bundled/fenced
  code is complete and runnable, not a stub), *no-missing-files* (every
  bundled-file reference resolves; user-project paths do not count), and
  *purpose-aligned*. The headline functional fraction is
  ``code-bearing AND code-sound AND no-missing-files``; runtime correctness is
  out of scope (it would require executing every skill).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pydantic import BaseModel

from analysis.results.utils.quality_utils import code_block_count, code_fraction

if TYPE_CHECKING:
    from pathlib import Path

_SCRIPT_EXTS = frozenset(
    {"py", "sh", "js", "ts", "mjs", "rb", "go", "java", "rs", "cpp", "c"}
)


class SkillStructural(BaseModel, frozen=True):
    """Objective structural proxies for one skill (full-corpus layer)."""

    name: str
    has_scripts: bool
    code_blocks: int
    code_fraction: float
    bundled_files: int
    code_bearing: bool


class JudgeVerdict(BaseModel, frozen=True):
    """LLM judgement for one sampled skill."""

    name: str
    code_sound: bool
    missing_files: bool
    purpose_aligned: bool
    tier: str  # functional | partial | reference_only
    reason: str


class CorpusDistribution(BaseModel, frozen=True):
    """Full-corpus structural rates (the overall distribution)."""

    n: int
    code_bearing: float
    has_scripts: float
    has_code_block: float
    has_bundled_file: float
    mean_code_fraction: float


class AuditSummary(BaseModel, frozen=True):
    """Judged-sample functionality funnel + Wilson CI."""

    n_judged: int
    n_code_bearing: int
    n_code_sound: int
    n_functional: int
    n_purpose_aligned: int
    functional_fraction: float
    functional_ci: tuple[float, float]
    tier_functional: int
    tier_partial: int
    tier_reference_only: int


def scan_skill(skill_dir: Path) -> SkillStructural | None:
    """Compute structural proxies for one skill directory (None if no SKILL.md)."""
    md = skill_dir / "SKILL.md"
    if not md.exists():
        return None
    try:
        text = md.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    bundled = 0
    has_scripts = False
    for f in skill_dir.rglob("*"):
        if f.is_file() and f.name != "SKILL.md":
            bundled += 1
            if f.suffix.lstrip(".").lower() in _SCRIPT_EXTS:
                has_scripts = True
    cbc = code_block_count(text)
    return SkillStructural(
        name=skill_dir.name,
        has_scripts=has_scripts,
        code_blocks=cbc,
        code_fraction=round(code_fraction(text), 4),
        bundled_files=bundled,
        code_bearing=has_scripts or cbc >= 1,
    )


def scan_corpus(corpus_dir: Path) -> list[SkillStructural]:
    """Structural proxies for every skill directory in the corpus."""
    out: list[SkillStructural] = []
    for d in sorted(corpus_dir.iterdir()):
        if not d.is_dir():
            continue
        rec = scan_skill(d)
        if rec is not None:
            out.append(rec)
    return out


def structural_distribution(rows: list[SkillStructural]) -> CorpusDistribution:
    """Full-corpus rates (the overall distribution the audit reports)."""
    n = len(rows)
    if n == 0:
        return CorpusDistribution(
            n=0,
            code_bearing=0.0,
            has_scripts=0.0,
            has_code_block=0.0,
            has_bundled_file=0.0,
            mean_code_fraction=0.0,
        )
    return CorpusDistribution(
        n=n,
        code_bearing=sum(r.code_bearing for r in rows) / n,
        has_scripts=sum(r.has_scripts for r in rows) / n,
        has_code_block=sum(r.code_blocks >= 1 for r in rows) / n,
        has_bundled_file=sum(r.bundled_files > 0 for r in rows) / n,
        mean_code_fraction=sum(r.code_fraction for r in rows) / n,
    )


def stratified_sample(
    rows: list[SkillStructural], k: int, *, seed: int = 42
) -> list[SkillStructural]:
    """Proportional sample stratified by ``has_scripts`` (corpus-representative).

    Proportional allocation keeps the sample's script/no-script mix equal to
    the corpus, so the judged functional fraction estimates the corpus value
    without reweighting.
    """
    rng = random.Random(seed)
    scripts = [r for r in rows if r.has_scripts]
    rest = [r for r in rows if not r.has_scripts]
    n = len(rows)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    k_scripts = round(k * len(scripts) / n)
    k_scripts = min(k_scripts, len(scripts))
    k_rest = min(k - k_scripts, len(rest))
    return rng.sample(scripts, k_scripts) + rng.sample(rest, k_rest)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion (stable for small/extreme p)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def audit_summary(
    structural: dict[str, SkillStructural],
    verdicts: list[JudgeVerdict],
) -> AuditSummary:
    """Funnel over the judged sample: code-bearing -> sound -> no-missing.

    ``structural`` maps skill name -> its structural record (for the objective
    code-bearing flag). The headline functional fraction is
    code-bearing AND code-sound AND not missing_files.
    """
    n = len(verdicts)
    if n == 0:
        return AuditSummary(
            n_judged=0,
            n_code_bearing=0,
            n_code_sound=0,
            n_functional=0,
            n_purpose_aligned=0,
            functional_fraction=0.0,
            functional_ci=(0.0, 0.0),
            tier_functional=0,
            tier_partial=0,
            tier_reference_only=0,
        )
    bearing = [v for v in verdicts if structural[v.name].code_bearing]
    sound = [v for v in bearing if v.code_sound]
    functional = [v for v in sound if not v.missing_files]
    purpose = [v for v in functional if v.purpose_aligned]
    lo, hi = wilson_interval(len(functional), n)
    return AuditSummary(
        n_judged=n,
        n_code_bearing=len(bearing),
        n_code_sound=len(sound),
        n_functional=len(functional),
        n_purpose_aligned=len(purpose),
        functional_fraction=len(functional) / n,
        functional_ci=(round(lo, 4), round(hi, 4)),
        tier_functional=sum(1 for v in verdicts if v.tier == "functional"),
        tier_partial=sum(1 for v in verdicts if v.tier == "partial"),
        tier_reference_only=sum(1 for v in verdicts if v.tier == "reference_only"),
    )

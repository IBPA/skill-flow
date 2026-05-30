"""Tests for the corpus functionality audit aggregation (functional_utils)."""

from __future__ import annotations

from analysis.results.utils.functional_utils import (
    JudgeVerdict,
    SkillStructural,
    audit_summary,
    stratified_sample,
    structural_distribution,
    wilson_interval,
)


def _struct(name: str, *, scripts: bool, blocks: int) -> SkillStructural:
    return SkillStructural(
        name=name,
        has_scripts=scripts,
        code_blocks=blocks,
        code_fraction=0.3 if blocks else 0.0,
        bundled_files=1 if scripts else 0,
        code_bearing=scripts or blocks >= 1,
    )


def _verdict(name: str, *, sound: bool, missing: bool, tier: str) -> JudgeVerdict:
    return JudgeVerdict(
        name=name,
        code_sound=sound,
        missing_files=missing,
        purpose_aligned=True,
        tier=tier,
        reason="",
    )


def test_structural_distribution_rates() -> None:
    rows = [
        _struct("a", scripts=True, blocks=3),
        _struct("b", scripts=False, blocks=2),
        _struct("c", scripts=False, blocks=0),  # not code-bearing
        _struct("d", scripts=False, blocks=0),
    ]
    dist = structural_distribution(rows)
    assert dist.n == 4
    assert dist.code_bearing == 0.5  # a, b
    assert dist.has_scripts == 0.25  # a
    assert dist.has_code_block == 0.5  # a, b


def test_audit_summary_funnel() -> None:
    structural = {
        "a": _struct("a", scripts=True, blocks=3),  # code-bearing
        "b": _struct("b", scripts=True, blocks=1),  # code-bearing, but unsound
        "c": _struct("c", scripts=True, blocks=1),  # code-bearing, sound but missing
        "p": _struct("p", scripts=False, blocks=0),  # NOT code-bearing
    }
    verdicts = [
        _verdict("a", sound=True, missing=False, tier="functional"),
        _verdict("b", sound=False, missing=False, tier="partial"),
        _verdict("c", sound=True, missing=True, tier="partial"),
        _verdict("p", sound=True, missing=False, tier="reference_only"),
    ]
    s = audit_summary(structural, verdicts)
    assert s.n_judged == 4
    assert s.n_code_bearing == 3  # a, b, c (p excluded by structural flag)
    assert s.n_code_sound == 2  # a, c
    assert s.n_functional == 1  # only a (c has missing files)
    assert s.functional_fraction == 0.25
    assert s.tier_reference_only == 1


def test_stratified_sample_is_proportional_and_seeded() -> None:
    rows = [_struct(f"s{i}", scripts=True, blocks=1) for i in range(20)]
    rows += [_struct(f"n{i}", scripts=False, blocks=1) for i in range(80)]
    sample = stratified_sample(rows, 10, seed=1)
    assert len(sample) == 10
    # 20% scripts in corpus -> ~2 in a sample of 10
    assert sum(r.has_scripts for r in sample) == 2
    # deterministic under fixed seed
    assert [r.name for r in sample] == [
        r.name for r in stratified_sample(rows, 10, seed=1)
    ]


def test_wilson_interval_bounds() -> None:
    lo, hi = wilson_interval(97, 300)
    assert 0.0 <= lo < 0.323 < hi <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)

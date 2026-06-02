"""Tests for skill-usage aggregation (analysis.results.utils.usage_utils)."""

from __future__ import annotations

from analysis.results.utils.usage_utils import (
    SkillUsage,
    TaskOutcome,
    helphurt,
    usage_summary,
)


def _usage(name: str, *, oracle: bool, loaded: bool) -> SkillUsage:
    return SkillUsage(
        run="r1",
        task="t",
        folder=name,
        name=name,
        is_oracle=oracle,
        loaded=loaded,
    )


def test_usage_summary_load_rates_and_share() -> None:
    """Oracle vs. non-oracle load rates and non-oracle share are correct."""
    records = [
        _usage("o1", oracle=True, loaded=True),
        _usage("o2", oracle=True, loaded=False),  # oracle load rate = 1/2
        _usage("n1", oracle=False, loaded=True),
        _usage("n2", oracle=False, loaded=False),
        _usage("n3", oracle=False, loaded=False),  # non-oracle load = 1/3
    ]
    s = usage_summary(records)
    assert s["n_injected"] == 5
    assert s["n_oracle"] == 2
    assert s["n_nonoracle"] == 3
    assert s["nonoracle_share"] == 3 / 5
    assert s["oracle_load_rate"] == 1 / 2
    assert s["nonoracle_load_rate"] == 1 / 3


def _outcome(
    task: str, sf: tuple[bool, ...], bl: tuple[bool, ...], n_inj: int, n_or: int
) -> TaskOutcome:
    return TaskOutcome(
        task=task, skillflow=sf, baseline=bl, n_injected=n_inj, n_oracle=n_or
    )


def test_helphurt_partitions_by_injection_group() -> None:
    """Tasks split into oracle-present / non-oracle-only / no-skills."""
    outcomes = [
        # oracle-present: SF helps
        _outcome("op_help", (True, True), (False, False), n_inj=2, n_or=1),
        # oracle-present: SF hurts
        _outcome("op_hurt", (False, False), (True, True), n_inj=2, n_or=1),
        # non-oracle-only: tie (both fail)
        _outcome("no_only", (False, False), (False, False), n_inj=2, n_or=0),
        # no skills injected
        _outcome("none", (False, False), (False, False), n_inj=0, n_or=0),
    ]
    hh = helphurt(outcomes)
    groups = {p.group: p for p in hh.partitions}

    assert groups["all"].n == 4
    assert groups["all"].helped == 1
    assert groups["all"].hurt == 1
    assert groups["oracle_present"].n == 2
    assert groups["oracle_present"].helped == 1
    assert groups["oracle_present"].hurt == 1
    assert groups["nonoracle_only"].n == 1
    assert groups["nonoracle_only"].helped == 0
    assert groups["nonoracle_only"].hurt == 0
    assert hh.nonoracle_only_tasks == ["no_only"]
    assert hh.hurt_tasks == ["op_hurt"]


def test_helphurt_passrate_is_mean_over_runs() -> None:
    """SF pass-rate aggregates mean-over-runs across tasks in the group."""
    outcomes = [
        _outcome("a", (True, False), (False, False), n_inj=1, n_or=1),
        _outcome("b", (True, True), (False, False), n_inj=1, n_or=1),
    ]
    hh = helphurt(outcomes)
    op = next(p for p in hh.partitions if p.group == "oracle_present")
    # mean pass-rates: a=0.5, b=1.0 -> group mean 0.75
    assert op.sf_passrate == 0.75
    assert op.bl_passrate == 0.0

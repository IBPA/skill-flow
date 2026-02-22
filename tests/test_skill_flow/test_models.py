"""Tests for src.models."""

import pytest
from pydantic import ValidationError
from skill_flow.models import SkillRecord


def test_skill_record_creation():
    record = SkillRecord(
        key="skillsmp/test-skill",
        name="test-skill",
        description="A test skill",
        source="skillsmp",
        local_path="skillsmp/test-skill",
        metadata={"stars": 42},
    )
    assert record.key == "skillsmp/test-skill"
    assert record.name == "test-skill"
    assert record.description == "A test skill"
    assert record.source == "skillsmp"
    assert record.local_path == "skillsmp/test-skill"
    assert record.metadata == {"stars": 42}


def test_skill_record_default_metadata():
    record = SkillRecord(
        key="skillsmp/x",
        name="x",
        description="desc",
        source="skillsmp",
        local_path="skillsmp/x",
    )
    assert record.metadata == {}


def test_skill_record_frozen():
    record = SkillRecord(
        key="skillsmp/x",
        name="x",
        description="desc",
        source="skillsmp",
        local_path="skillsmp/x",
    )
    attr = "key"
    with pytest.raises(ValidationError):
        setattr(record, attr, "other")

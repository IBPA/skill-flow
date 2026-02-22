"""Tests for src.corpus.loader."""

import json

import pytest
from skill_flow.corpus.loader import load_content, load_corpus


@pytest.fixture()
def corpus_dir(tmp_path):
    """Create a minimal corpus directory structure."""
    meta = tmp_path / "_metadata"
    meta.mkdir()

    skills = {
        "skillsmp/alpha": {
            "name": "alpha",
            "description": "Alpha skill description",
            "source": "skillsmp",
            "local_path": "data/skills/skillsmp/alpha",
            "stars": 10,
        },
        "skillsmp/beta": {
            "name": "beta",
            "description": "Beta skill description",
            "source": "skillsmp",
            "local_path": "data/skills/skillsmp/beta",
            "stars": 20,
        },
        "skillsmp/empty": {
            "name": "empty",
            "description": "",
            "source": "skillsmp",
            "local_path": "data/skills/skillsmp/empty",
        },
        "skillsmp/none-desc": {
            "name": "none-desc",
            "description": None,
            "source": "skillsmp",
            "local_path": "data/skills/skillsmp/none-desc",
        },
    }
    index_data = {"version": "1.0", "skills": skills}
    (meta / "index.json").write_text(json.dumps(index_data))

    # Create SKILL.md files
    for key in ("skillsmp/alpha", "skillsmp/beta"):
        skill_dir = tmp_path / key
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {key}\nContent here.")

    return tmp_path


def test_load_corpus(corpus_dir):
    records = load_corpus(corpus_dir)
    assert len(records) == 2

    keys = {r.key for r in records}
    assert keys == {"skillsmp/alpha", "skillsmp/beta"}


def test_load_corpus_field_values(corpus_dir):
    records = load_corpus(corpus_dir)
    alpha = next(r for r in records if r.key == "skillsmp/alpha")
    assert alpha.name == "alpha"
    assert alpha.description == "Alpha skill description"
    assert alpha.source == "skillsmp"
    assert alpha.local_path == "skillsmp/alpha"
    assert alpha.metadata["stars"] == 10


def test_load_corpus_filters_empty_descriptions(corpus_dir):
    records = load_corpus(corpus_dir)
    keys = {r.key for r in records}
    assert "skillsmp/empty" not in keys
    assert "skillsmp/none-desc" not in keys


def test_load_content(corpus_dir):
    records = load_corpus(corpus_dir)
    alpha = next(r for r in records if r.key == "skillsmp/alpha")
    content = load_content(corpus_dir, alpha)
    assert "# skillsmp/alpha" in content
    assert "Content here." in content

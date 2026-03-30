"""Tests for src.index.builder."""

import json
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest
from skill_flow.index.builder import build_index
from skill_flow.models import SkillRecord

RNG = np.random.default_rng(42)
DIM = 768


def _make_skills(n: int) -> list[SkillRecord]:
    return [
        SkillRecord(
            key=f"skillsmp/skill-{i}",
            name=f"skill-{i}",
            description=f"Description for skill {i}",
            source="skillsmp",
            local_path=f"skillsmp/skill-{i}",
        )
        for i in range(n)
    ]


def _make_mock_encoder(n: int, max_seq_length: int = 512) -> MagicMock:
    """Return a mock encoder that produces normalized random embeddings."""
    encoder = MagicMock()
    vecs = RNG.random((n, DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    encoder.encode_documents.return_value = vecs / norms
    encoder.max_seq_length = max_seq_length
    # truncate_text: simulate token ≈ 4 chars
    encoder.truncate_text.side_effect = lambda text, n: text[: n * 4]
    return encoder


@pytest.fixture()
def index_artifacts(tmp_path):
    """Build an index and return (output_dir, skills, encoder)."""
    skills = _make_skills(5)
    encoder = _make_mock_encoder(5)
    build_index(skills, encoder, tmp_path)
    return tmp_path, skills, encoder


def test_artifacts_exist(index_artifacts):
    output_dir, _, _ = index_artifacts
    assert (output_dir / "embeddings.npy").exists()
    assert (output_dir / "faiss.index").exists()
    assert (output_dir / "skill_ids.json").exists()
    assert (output_dir / "skill_descriptions.json").exists()


def test_embeddings_shape(index_artifacts):
    output_dir, _, _ = index_artifacts
    embeddings = np.load(output_dir / "embeddings.npy")
    assert embeddings.shape == (5, DIM)
    assert embeddings.dtype == np.float32


def test_faiss_index_size(index_artifacts):
    output_dir, _, _ = index_artifacts
    index = faiss.read_index(str(output_dir / "faiss.index"))
    assert index.ntotal == 5


def test_skill_ids(index_artifacts):
    output_dir, skills, _ = index_artifacts
    ids = json.loads((output_dir / "skill_ids.json").read_text())
    assert ids == [s.key for s in skills]


def test_skill_descriptions(index_artifacts):
    output_dir, skills, _ = index_artifacts
    descs = json.loads((output_dir / "skill_descriptions.json").read_text())
    assert descs == {s.key: s.description for s in skills}


def test_encoder_called_with_descriptions(index_artifacts):
    _, skills, encoder = index_artifacts
    descriptions = [s.description for s in skills]
    encoder.encode_documents.assert_called_once_with(descriptions, batch_size=256)


def test_creates_output_dir(tmp_path):
    nested = tmp_path / "a" / "b"
    skills = _make_skills(2)
    encoder = _make_mock_encoder(2)
    build_index(skills, encoder, nested)
    assert nested.exists()
    assert (nested / "faiss.index").exists()


def test_skill_contents_saved_with_corpus_path(tmp_path):
    """When corpus_path is provided, skill_contents.json is persisted."""
    corpus = tmp_path / "corpus"
    output = tmp_path / "index"

    skills = _make_skills(2)
    for s in skills:
        skill_dir = corpus / s.local_path
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# Content for {s.name}")

    encoder = _make_mock_encoder(2)
    build_index(skills, encoder, output, corpus_path=corpus)

    contents_path = output / "skill_contents.json"
    assert contents_path.exists()
    contents = json.loads(contents_path.read_text())
    assert contents[skills[0].key] == f"# Content for {skills[0].name}"
    assert contents[skills[1].key] == f"# Content for {skills[1].name}"


def test_no_skill_contents_without_corpus_path(index_artifacts):
    """Without corpus_path, skill_contents.json is not created."""
    output_dir, _, _ = index_artifacts
    assert not (output_dir / "skill_contents.json").exists()


def test_max_content_tokens_positive_encodes_with_content(tmp_path):
    """When max_content_tokens > 0, encoder receives description + content."""
    corpus = tmp_path / "corpus"
    output = tmp_path / "index"

    skills = _make_skills(2)
    for s in skills:
        skill_dir = corpus / s.local_path
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# Full content for {s.name}")

    encoder = _make_mock_encoder(2)
    build_index(
        skills,
        encoder,
        output,
        corpus_path=corpus,
        max_content_tokens=500,
    )

    assert encoder.truncate_text.call_count == 2
    for call_args_item in encoder.truncate_text.call_args_list:
        text, token_limit = call_args_item.args
        assert "\n# Full content for" in text
        assert token_limit == 500


def test_max_content_tokens_negative_one_uses_model_limit(tmp_path):
    """max_content_tokens=-1 fills to the model's max_seq_length."""
    corpus = tmp_path / "corpus"
    output = tmp_path / "index"

    skills = _make_skills(1)
    skill_dir = corpus / skills[0].local_path
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("x" * 100)

    encoder = _make_mock_encoder(1, max_seq_length=256)
    build_index(
        skills,
        encoder,
        output,
        corpus_path=corpus,
        max_content_tokens=-1,
    )

    _, token_limit = encoder.truncate_text.call_args.args
    assert token_limit == 256


def test_max_content_tokens_zero_uses_descriptions_only(tmp_path):
    """max_content_tokens=0 (default) encodes descriptions only."""
    corpus = tmp_path / "corpus"
    output = tmp_path / "index"

    skills = _make_skills(2)
    for s in skills:
        skill_dir = corpus / s.local_path
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("should not appear")

    encoder = _make_mock_encoder(2)
    build_index(
        skills,
        encoder,
        output,
        corpus_path=corpus,
        max_content_tokens=0,
    )

    call_args = encoder.encode_documents.call_args[0][0]
    assert call_args == [s.description for s in skills]
    encoder.truncate_text.assert_not_called()

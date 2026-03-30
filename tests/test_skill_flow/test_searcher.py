"""Tests for src.index.searcher."""

import json
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest
from skill_flow.retriever.retriever import IndexSearcher, SearchResult

DIM = 768


def _build_test_index(tmp_path, n: int = 5, *, with_contents: bool = False):
    """Build a small FAISS index with known embeddings."""
    rng = np.random.default_rng(99)
    vecs = rng.random((n, DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / "faiss.index"))

    keys = [f"skillsmp/skill-{i}" for i in range(n)]
    (tmp_path / "skill_ids.json").write_text(json.dumps(keys))

    descriptions = {k: f"Description for {k}" for k in keys}
    (tmp_path / "skill_descriptions.json").write_text(json.dumps(descriptions))

    if with_contents:
        contents = {k: f"Full SKILL.md for {k}" for k in keys}
        (tmp_path / "skill_contents.json").write_text(json.dumps(contents))

    return vecs, keys


def _make_mock_encoder(query_vec: np.ndarray) -> MagicMock:
    encoder = MagicMock()
    encoder.encode_query.return_value = query_vec.reshape(1, DIM)
    return encoder


@pytest.fixture()
def index_dir(tmp_path):
    vecs, keys = _build_test_index(tmp_path)
    return tmp_path, vecs, keys


def test_loads_index(index_dir):
    dir_path, vecs, keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)
    assert searcher._index.ntotal == len(keys)


def test_search_returns_results(index_dir):
    dir_path, vecs, _keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)

    results = searcher.search("test query", top_k=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_top_result_is_self(index_dir):
    """Querying with a document's own vector should rank it first."""
    dir_path, vecs, keys = index_dir
    encoder = _make_mock_encoder(vecs[2])
    searcher = IndexSearcher(dir_path, encoder)

    results = searcher.search("test", top_k=5)
    assert results[0].key == keys[2]
    np.testing.assert_allclose(results[0].score, 1.0, atol=1e-5)


def test_search_scores_descending(index_dir):
    dir_path, vecs, _keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)

    results = searcher.search("test", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_clamped(index_dir):
    """top_k larger than index size should not error."""
    dir_path, vecs, keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)

    results = searcher.search("test", top_k=100)
    assert len(results) == len(keys)


def test_search_result_fields(index_dir):
    dir_path, vecs, keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)

    results = searcher.search("test", top_k=1)
    r = results[0]
    assert isinstance(r.key, str)
    assert isinstance(r.score, float)
    assert r.key in keys


def test_search_populates_descriptions(index_dir):
    dir_path, vecs, _keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)

    results = searcher.search("test", top_k=3)
    for r in results:
        assert r.description == f"Description for {r.key}"


def test_add_descriptions(index_dir):
    dir_path, vecs, _keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)

    searcher.add_descriptions({"new/skill": "new description"})
    assert searcher._descriptions["new/skill"] == "new description"


def test_search_populates_contents(tmp_path):
    vecs, _keys = _build_test_index(tmp_path, with_contents=True)
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(tmp_path, encoder)

    results = searcher.search("test", top_k=3)
    for r in results:
        assert r.content == f"Full SKILL.md for {r.key}"


def test_add_contents(index_dir):
    dir_path, vecs, _keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)

    searcher.add_contents({"new/skill": "# SKILL.md content"})
    assert searcher._contents["new/skill"] == "# SKILL.md content"


def test_missing_contents_file(index_dir):
    """Searcher works without skill_contents.json (backward compat)."""
    dir_path, vecs, _keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(dir_path, encoder)

    results = searcher.search("test", top_k=2)
    assert results[0].content == ""


def test_augment_adds_vectors(index_dir):
    dir_path, vecs, _keys = index_dir
    encoder = _make_mock_encoder(vecs[0])
    encoder.encode_documents = MagicMock(
        return_value=np.random.default_rng(1).random((1, DIM)).astype(np.float32)
    )
    searcher = IndexSearcher(dir_path, encoder)

    original_count = searcher._index.ntotal
    searcher.augment(["new/skill"], ["new description"])

    assert searcher._index.ntotal == original_count + 1
    assert "new/skill" in searcher._skill_keys
    encoder.encode_documents.assert_called_once_with(["new description"])


def test_missing_descriptions_file(tmp_path):
    """Searcher works without skill_descriptions.json (backward compat)."""
    rng = np.random.default_rng(99)
    vecs = rng.random((2, DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    index = faiss.IndexFlatIP(DIM)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / "faiss.index"))
    (tmp_path / "skill_ids.json").write_text(json.dumps(["a", "b"]))

    encoder = _make_mock_encoder(vecs[0])
    searcher = IndexSearcher(tmp_path, encoder)

    results = searcher.search("test", top_k=2)
    assert results[0].description == ""

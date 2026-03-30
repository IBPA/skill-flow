"""Tests for skill_flow.retriever.protocol — both searchers satisfy Searcher."""

import json
from unittest.mock import MagicMock

import faiss
import numpy as np
from skill_flow.config import RetrieverConfig
from skill_flow.retriever.bm25 import BM25Searcher, _tokenize
from skill_flow.retriever.protocol import Searcher
from skill_flow.retriever.retriever import IndexSearcher


def _make_bm25() -> BM25Searcher:
    keys = ["skill/a"]
    descs = {"skill/a": "python testing"}
    tokens = [_tokenize(descs["skill/a"])]
    return BM25Searcher(keys, tokens, descs, {})


def _make_dense(tmp_path) -> IndexSearcher:
    dim = 32
    vecs = np.random.default_rng(42).random((2, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(tmp_path / "faiss.index"))
    (tmp_path / "skill_ids.json").write_text(json.dumps(["a", "b"]))
    (tmp_path / "skill_descriptions.json").write_text(
        json.dumps({"a": "desc a", "b": "desc b"})
    )

    encoder = MagicMock()
    encoder.encode_query.return_value = vecs[0].reshape(1, dim)
    return IndexSearcher(tmp_path, encoder, RetrieverConfig())


class TestProtocolConformance:
    def test_bm25_satisfies_protocol(self):
        searcher = _make_bm25()
        assert isinstance(searcher, Searcher)

    def test_dense_satisfies_protocol(self, tmp_path):
        searcher = _make_dense(tmp_path)
        assert isinstance(searcher, Searcher)

    def test_bm25_has_required_methods(self):
        searcher = _make_bm25()
        assert hasattr(searcher, "search")
        assert hasattr(searcher, "augment")
        assert hasattr(searcher, "add_descriptions")
        assert hasattr(searcher, "add_contents")

    def test_dense_has_required_methods(self, tmp_path):
        searcher = _make_dense(tmp_path)
        assert hasattr(searcher, "search")
        assert hasattr(searcher, "augment")
        assert hasattr(searcher, "add_descriptions")
        assert hasattr(searcher, "add_contents")

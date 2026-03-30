"""Tests for skill_flow.retriever.bm25."""

import json

from skill_flow.config import RetrieverConfig
from skill_flow.retriever.bm25 import BM25Searcher, _tokenize
from skill_flow.retriever.retriever import SearchResult


def _make_bm25(
    keys: list[str] | None = None,
    descriptions: dict[str, str] | None = None,
    contents: dict[str, str] | None = None,
) -> BM25Searcher:
    keys = keys or ["skill/a", "skill/b", "skill/c"]
    descriptions = descriptions or {
        "skill/a": "python unit testing with pytest",
        "skill/b": "javascript react component",
        "skill/c": "python flask web application",
    }
    contents = contents or {}
    tokens = [_tokenize(descriptions.get(k, "")) for k in keys]
    config = RetrieverConfig(retriever_type="bm25", top_k=10)
    return BM25Searcher(keys, tokens, descriptions, contents, config)


class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize("") == []


class TestBM25Search:
    def test_returns_results(self):
        searcher = _make_bm25()
        results = searcher.search("python testing")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_k(self):
        searcher = _make_bm25()
        results = searcher.search("python", top_k=2)
        assert len(results) <= 2

    def test_scores_descending(self):
        searcher = _make_bm25()
        results = searcher.search("python")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_populates_description(self):
        searcher = _make_bm25()
        results = searcher.search("python")
        assert results[0].description != ""

    def test_populates_content(self):
        searcher = _make_bm25(
            contents={"skill/a": "# Full content"},
        )
        results = searcher.search("python testing")
        content_result = next((r for r in results if r.key == "skill/a"), None)
        assert content_result is not None
        assert content_result.content == "# Full content"

    def test_no_negative_scores(self):
        searcher = _make_bm25()
        results = searcher.search("completely unrelated xyz")
        for r in results:
            assert r.score > 0


class TestBM25Augment:
    def test_augment_adds_skills(self):
        searcher = _make_bm25()
        searcher.augment(["skill/new"], ["docker containerization"])
        results = searcher.search("docker")
        keys = [r.key for r in results]
        assert "skill/new" in keys

    def test_augment_empty(self):
        searcher = _make_bm25()
        searcher.augment([], [])
        results = searcher.search("python")
        assert len(results) > 0


class TestBM25AddDescriptionsContents:
    def test_add_descriptions(self):
        searcher = _make_bm25()
        searcher.add_descriptions({"skill/new": "new desc"})
        assert searcher._descriptions["skill/new"] == "new desc"

    def test_add_contents(self):
        searcher = _make_bm25()
        searcher.add_contents({"skill/new": "# Content"})
        assert searcher._contents["skill/new"] == "# Content"


class TestBM25FromIndexDir:
    def test_loads_from_dir(self, tmp_path):
        keys = ["skill/a", "skill/b", "skill/c"]
        descs = {
            "skill/a": "python testing with pytest framework",
            "skill/b": "react component javascript frontend",
            "skill/c": "docker containerization deployment",
        }
        contents = {"skill/a": "# Full a", "skill/b": "# Full b", "skill/c": "# Full c"}

        (tmp_path / "skill_ids.json").write_text(json.dumps(keys))
        (tmp_path / "skill_descriptions.json").write_text(json.dumps(descs))
        (tmp_path / "skill_contents.json").write_text(json.dumps(contents))

        config = RetrieverConfig(retriever_type="bm25")
        searcher = BM25Searcher.from_index_dir(tmp_path, config)

        results = searcher.search("python testing")
        assert len(results) > 0
        assert results[0].content != ""

    def test_missing_optional_files(self, tmp_path):
        (tmp_path / "skill_ids.json").write_text(json.dumps([]))

        config = RetrieverConfig(retriever_type="bm25")
        searcher = BM25Searcher.from_index_dir(tmp_path, config)

        results = searcher.search("anything")
        assert isinstance(results, list)
        assert len(results) == 0

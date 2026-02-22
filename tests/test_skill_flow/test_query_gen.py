"""Tests for skill_flow.reranker.query_gen."""

import json
from unittest.mock import MagicMock, patch

from skill_flow.config import QueryGenConfig
from skill_flow.reranker.query_gen import QueryGenerator


def _make_config(tmp_path: object) -> QueryGenConfig:
    return QueryGenConfig(
        enabled=True,
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.0,
        cache_path=str(tmp_path) + "/cache.json",
    )


def _mock_response(content="concise query"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestQueryGenerator:
    @patch("skill_flow.reranker.query_gen.OpenAI")
    def test_generate_calls_openai(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            "parse STL files"
        )

        gen = QueryGenerator(config)
        result = gen.generate("task-1", "Long detailed instruction about 3D printing")

        assert result == "parse STL files"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": config.system_prompt},
                {
                    "role": "user",
                    "content": "Long detailed instruction about 3D printing",
                },
            ],
            max_tokens=200,
            temperature=0.0,
        )

    @patch("skill_flow.reranker.query_gen.OpenAI")
    def test_cache_hit_skips_llm(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"task-1": "cached query"}))

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        gen = QueryGenerator(config)
        result = gen.generate("task-1", "Some instruction")

        assert result == "cached query"
        mock_client.chat.completions.create.assert_not_called()

    @patch("skill_flow.reranker.query_gen.OpenAI")
    def test_cache_persists_to_disk(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("new query")

        gen = QueryGenerator(config)
        gen.generate("task-1", "instruction")

        cache_path = tmp_path / "cache.json"
        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert data["task-1"] == "new query"

    @patch("skill_flow.reranker.query_gen.OpenAI")
    def test_loads_existing_cache(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"task-old": "old query"}))

        mock_openai_cls.return_value = MagicMock()

        gen = QueryGenerator(config)
        assert gen._cache == {"task-old": "old query"}

    @patch("skill_flow.reranker.query_gen.OpenAI")
    def test_strips_whitespace(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            "  padded query  \n"
        )

        gen = QueryGenerator(config)
        result = gen.generate("task-1", "instruction")

        assert result == "padded query"

    @patch("skill_flow.reranker.query_gen.OpenAI")
    def test_handles_none_content(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(None)

        gen = QueryGenerator(config)
        result = gen.generate("task-1", "original instruction")

        assert result == "original instruction"

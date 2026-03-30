"""Tests for skill_flow.query_gen."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from skill_flow.config import QueryGenConfig
from skill_flow.query_gen.query_gen import QueryGenerator

_MULTI_PROMPT_PATH = str(
    Path("skill_flow/query_gen/instructions/multi_v2.txt").resolve()
)


def _make_config(tmp_path: object) -> QueryGenConfig:
    return QueryGenConfig(
        enabled=True,
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.0,
        cache_path=str(tmp_path) + "/cache.json",
        multi_query_prompt=_MULTI_PROMPT_PATH,
    )


def _mock_response(content="concise query"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class TestQueryGenerator:
    @patch("skill_flow.query_gen.query_gen.OpenAI")
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

    @patch("skill_flow.query_gen.query_gen.OpenAI")
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

    @patch("skill_flow.query_gen.query_gen.OpenAI")
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

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_loads_existing_cache(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"task-old": "old query"}))

        mock_openai_cls.return_value = MagicMock()

        gen = QueryGenerator(config)
        assert gen._cache == {"task-old": "old query"}

    @patch("skill_flow.query_gen.query_gen.OpenAI")
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

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_strips_surrounding_quotes(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '"quoted query"'
        )

        gen = QueryGenerator(config)
        result = gen.generate("task-1", "instruction")

        assert result == "quoted query"

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_handles_none_content(self, mock_openai_cls, tmp_path):
        config = _make_config(tmp_path)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(None)

        gen = QueryGenerator(config)
        result = gen.generate("task-1", "original instruction")

        assert result == "original instruction"


def _make_multi_config(tmp_path: object, num_queries: int = 3) -> QueryGenConfig:
    return QueryGenConfig(
        enabled=True,
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.0,
        cache_path=str(tmp_path) + "/cache.json",
        num_queries=num_queries,
        aggregation="max",
        multi_query_prompt=_MULTI_PROMPT_PATH,
    )


class TestGenerateMulti:
    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_single_query_delegates_to_generate(self, mock_openai_cls, tmp_path):
        config = _make_multi_config(tmp_path, num_queries=1)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response("single q")

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["single q"]

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_parses_json_array(self, mock_openai_cls, tmp_path):
        config = _make_multi_config(tmp_path, num_queries=3)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        json_resp = '["FastAPI testing", "SQLAlchemy mocking", "Docker setup"]'
        mock_client.chat.completions.create.return_value = _mock_response(json_resp)

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "Write FastAPI tests with SQLAlchemy")

        assert result == ["FastAPI testing", "SQLAlchemy mocking", "Docker setup"]

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_truncates_to_num_queries(self, mock_openai_cls, tmp_path):
        config = _make_multi_config(tmp_path, num_queries=2)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        json_resp = '["q1", "q2", "q3"]'
        mock_client.chat.completions.create.return_value = _mock_response(json_resp)

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["q1", "q2"]

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_cache_backward_compat_str_to_list(self, mock_openai_cls, tmp_path):
        config = _make_multi_config(tmp_path, num_queries=3)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"task-1": "old single query"}))

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["old single query"]
        mock_client.chat.completions.create.assert_not_called()

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_cache_hit_with_list(self, mock_openai_cls, tmp_path):
        config = _make_multi_config(tmp_path, num_queries=3)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"task-1": ["q1", "q2"]}))

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["q1", "q2"]
        mock_client.chat.completions.create.assert_not_called()

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_caches_as_list(self, mock_openai_cls, tmp_path):
        config = _make_multi_config(tmp_path, num_queries=2)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_response(
            '["q1", "q2"]'
        )

        gen = QueryGenerator(config)
        gen.generate_multi("task-1", "instruction")

        cache_path = tmp_path / "cache.json"
        data = json.loads(cache_path.read_text())
        assert data["task-1"] == ["q1", "q2"]

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_generate_returns_first_from_cached_list(self, mock_openai_cls, tmp_path):
        config = _make_multi_config(tmp_path, num_queries=3)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"task-1": ["first", "second"]}))

        mock_openai_cls.return_value = MagicMock()

        gen = QueryGenerator(config)
        result = gen.generate("task-1", "instruction")

        assert result == "first"

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_invalid_json_falls_back_to_n_single(
        self, mock_openai_cls, tmp_path
    ):
        config = _make_multi_config(tmp_path, num_queries=3)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        bad = _mock_response("not valid json")
        singles = [_mock_response(f"query {i}") for i in range(3)]
        mock_client.chat.completions.create.side_effect = [bad, bad, *singles]

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        # 2 multi retries + 3 single-query calls = 5 total
        assert result == ["query 0", "query 1", "query 2"]
        assert mock_client.chat.completions.create.call_count == 5

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_fallback_feeds_context(self, mock_openai_cls, tmp_path):
        """Each single-query fallback call includes previously generated queries."""
        config = _make_multi_config(tmp_path, num_queries=2)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        bad = _mock_response("not valid json")
        s0 = _mock_response("first query")
        s1 = _mock_response("second query")
        mock_client.chat.completions.create.side_effect = [bad, bad, s0, s1]

        gen = QueryGenerator(config)
        gen.generate_multi("task-1", "instruction")

        # Second single call should include "first query" in user message
        calls = mock_client.chat.completions.create.call_args_list
        second_single_msg = calls[3].kwargs["messages"][1]["content"]
        assert "Already generated" in second_single_msg
        assert "first query" in second_single_msg

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_fallback_dedupes(self, mock_openai_cls, tmp_path):
        """Duplicate queries from single-query fallback are removed."""
        config = _make_multi_config(tmp_path, num_queries=3)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        bad = _mock_response("not valid json")
        same = _mock_response("same query")
        mock_client.chat.completions.create.side_effect = [bad, bad, same, same, same]

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["same query"]

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_long_string_falls_back_to_n_single(
        self, mock_openai_cls, tmp_path
    ):
        """LLM returning overly long strings (likely task output) triggers retry."""
        config = _make_multi_config(tmp_path, num_queries=2)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        long_str = "x" * 600
        bad = _mock_response(json.dumps([long_str]))
        singles = [_mock_response("short a"), _mock_response("short b")]
        mock_client.chat.completions.create.side_effect = [bad, bad, *singles]

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["short a", "short b"]

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_retry_succeeds_on_second_attempt(
        self, mock_openai_cls, tmp_path
    ):
        config = _make_multi_config(tmp_path, num_queries=2)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        bad = _mock_response('{"not": "an array"}')
        good = _mock_response('["query a", "query b"]')
        mock_client.chat.completions.create.side_effect = [bad, good]

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["query a", "query b"]
        assert mock_client.chat.completions.create.call_count == 2

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_fallback_strips_surrounding_quotes(self, mock_openai_cls, tmp_path):
        """Single-query fallback strips surrounding quotes from LLM output."""
        config = _make_multi_config(tmp_path, num_queries=2)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        bad = _mock_response("not valid json")
        s0 = _mock_response('"quoted first"')
        s1 = _mock_response('"quoted second"')
        mock_client.chat.completions.create.side_effect = [bad, bad, s0, s1]

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["quoted first", "quoted second"]

    @patch("skill_flow.query_gen.query_gen.OpenAI")
    def test_multi_query_strips_code_fence(self, mock_openai_cls, tmp_path):
        config = _make_multi_config(tmp_path, num_queries=3)
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        fenced = '```json\n["query one", "query two", "query three"]\n```'
        mock_client.chat.completions.create.return_value = _mock_response(fenced)

        gen = QueryGenerator(config)
        result = gen.generate_multi("task-1", "instruction")

        assert result == ["query one", "query two", "query three"]

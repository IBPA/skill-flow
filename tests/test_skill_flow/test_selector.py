"""Tests for skill_flow.selector.selector."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from skill_flow.config import SelectorConfig
from skill_flow.retriever.retriever import SearchResult
from skill_flow.selector.selector import Selector


def _make_config(tmp_path: Path, **overrides: object) -> SelectorConfig:
    # Create minimal j2 template files for tests
    instructions_dir = tmp_path / "instructions"
    instructions_dir.mkdir(exist_ok=True)
    system_j2 = instructions_dir / "system.j2"
    user_j2 = instructions_dir / "user.j2"
    if not system_j2.exists():
        system_j2.write_text("You are a skill selector.")
    if not user_j2.exists():
        user_j2.write_text(
            "{{ query }}\n"
            "{% for c in candidates %}"
            "{{ loop.index }}: {{ c.content }}"
            "{% endfor %}"
        )

    defaults: dict[str, object] = {
        "enabled": True,
        "cache_path": str(tmp_path / "cache.json"),
        "top_k": 5,
        "system_instruction": str(system_j2),
        "user_instruction": str(user_j2),
    }
    defaults.update(overrides)
    return SelectorConfig(**defaults)


def _make_candidates(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            key=f"skill-{i}",
            score=float(n - i),
            description=f"desc {i}",
            content=f"# Skill {i}\nContent for skill {i}",
        )
        for i in range(n)
    ]


class TestSelector:
    @patch("skill_flow.selector.selector.OpenAI")
    def test_select_calls_llm_and_filters(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        # 1-based: candidate 1 = skill-0, candidate 3 = skill-2
        mock_response.choices[0].message.content = "[1, 3]"
        mock_client.chat.completions.create.return_value = mock_response

        config = _make_config(tmp_path)
        selector = Selector(config)
        candidates = _make_candidates(3)

        results = selector.select("test query", candidates, task_id="t1")

        assert len(results) == 2
        assert results[0].key == "skill-0"
        assert results[1].key == "skill-2"
        mock_client.chat.completions.create.assert_called_once()

    @patch("skill_flow.selector.selector.OpenAI")
    def test_preserves_original_scores(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        # 1-based: candidate 2 = skill-1
        mock_response.choices[0].message.content = "[2]"
        mock_client.chat.completions.create.return_value = mock_response

        config = _make_config(tmp_path)
        selector = Selector(config)
        candidates = _make_candidates(3)

        results = selector.select("test query", candidates, task_id="t2")

        assert len(results) == 1
        assert results[0].key == "skill-1"
        assert results[0].score == 2.0  # original score preserved

    @patch("skill_flow.selector.selector.OpenAI")
    def test_empty_candidates(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        config = _make_config(tmp_path)
        selector = Selector(config)

        results = selector.select("test query", [])

        assert results == []
        mock_openai_cls.return_value.chat.completions.create.assert_not_called()

    @patch("skill_flow.selector.selector.OpenAI")
    def test_cache_hit(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        cache_path = tmp_path / "cache.json"
        cache_path.write_text(json.dumps({"t1": ["skill-0"]}))

        config = _make_config(tmp_path, cache_path=str(cache_path))
        selector = Selector(config)
        candidates = _make_candidates(3)

        results = selector.select("test query", candidates, task_id="t1")

        assert len(results) == 1
        assert results[0].key == "skill-0"
        mock_openai_cls.return_value.chat.completions.create.assert_not_called()

    @patch("skill_flow.selector.selector.OpenAI")
    def test_cache_write_through(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        # 1-based: candidate 1 = skill-0
        mock_response.choices[0].message.content = "[1]"
        mock_client.chat.completions.create.return_value = mock_response

        cache_path = tmp_path / "cache.json"
        config = _make_config(tmp_path, cache_path=str(cache_path))
        selector = Selector(config)
        candidates = _make_candidates(2)

        selector.select("test query", candidates, task_id="t1")

        assert cache_path.exists()
        cached: dict[str, list[str]] = json.loads(cache_path.read_text())
        assert cached["t1"] == ["skill-0"]

    @patch("skill_flow.selector.selector.OpenAI")
    def test_markdown_fence_parsing(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        # 1-based: candidates 1 and 2
        mock_response.choices[0].message.content = (
            "```json\n[1, 2]\n```"
        )
        mock_client.chat.completions.create.return_value = mock_response

        config = _make_config(tmp_path)
        selector = Selector(config)
        candidates = _make_candidates(3)

        results = selector.select("test query", candidates, task_id="t3")

        assert len(results) == 2
        assert {r.key for r in results} == {"skill-0", "skill-1"}

    @patch("skill_flow.selector.selector.OpenAI")
    def test_unparseable_response_returns_all(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I cannot parse this"
        mock_client.chat.completions.create.return_value = mock_response

        config = _make_config(tmp_path)
        selector = Selector(config)
        candidates = _make_candidates(3)

        results = selector.select("test query", candidates, task_id="t4")

        assert len(results) == 3

    @patch("skill_flow.selector.selector.OpenAI")
    def test_invalid_indices_filtered(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        # 1-based: candidate 1 = skill-0, 99 is out of range
        mock_response.choices[0].message.content = "[1, 99]"
        mock_client.chat.completions.create.return_value = mock_response

        config = _make_config(tmp_path)
        selector = Selector(config)
        candidates = _make_candidates(3)

        results = selector.select("test query", candidates, task_id="t5")

        assert len(results) == 1
        assert results[0].key == "skill-0"

    @patch("skill_flow.selector.selector.OpenAI")
    def test_top_k_config_trims_candidates(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        # 1-based: candidates 1 and 2 (top_k=2 so only first 2)
        mock_response.choices[0].message.content = "[1, 2]"
        mock_client.chat.completions.create.return_value = mock_response

        config = _make_config(tmp_path, top_k=2)
        selector = Selector(config)
        candidates = _make_candidates(5)

        results = selector.select("test query", candidates, task_id="t6")

        assert all(r.key in {"skill-0", "skill-1"} for r in results)

    @patch("skill_flow.selector.selector.OpenAI")
    def test_uses_query_as_cache_key_when_no_task_id(
        self, mock_openai_cls: MagicMock, tmp_path: Path,
    ):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "[1]"
        mock_client.chat.completions.create.return_value = mock_response

        cache_path = tmp_path / "cache.json"
        config = _make_config(tmp_path, cache_path=str(cache_path))
        selector = Selector(config)
        candidates = _make_candidates(2)

        selector.select("my query", candidates)

        cached: dict[str, list[str]] = json.loads(cache_path.read_text())
        assert "my query" in cached


class TestParseResponse:
    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("[1, 2]", ["a", "b"]),
            ("```json\n[1]\n```", ["a"]),
            ("```\n[1, 2]\n```", ["a", "b"]),
        ],
    )
    def test_valid_formats(self, content: str, expected: list[str]):
        mapping = {1: "a", 2: "b", 3: "c"}
        assert Selector._parse_response(content, mapping) == expected

    def test_filters_invalid_indices(self):
        mapping = {1: "a", 2: "b"}
        assert Selector._parse_response("[1, 99]", mapping) == ["a"]

    def test_raises_on_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            Selector._parse_response("not json", {1: "a"})

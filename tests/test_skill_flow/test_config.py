"""Tests for skill_flow.config."""

import json

from skill_flow.config import (
    Config,
    IndexConfig,
    ModelsConfig,
    QueryGenConfig,
    Reranker2Config,
    RerankerConfig,
    RetrieverConfig,
    SelectorConfig,
    SystemConfig,
    load_config,
)


def test_default_retriever_config():
    config = RetrieverConfig()
    assert config.model_name == "BAAI/bge-base-en-v1.5"
    assert "Represent this sentence" in config.query_prompt
    assert config.batch_size == 256
    assert config.top_k == 100
    assert config.eval is None


def test_default_reranker_config():
    config = RerankerConfig()
    assert config.enabled is False
    assert config.model_name == "BAAI/bge-reranker-v2-m3"
    assert config.top_k == 10
    assert config.batch_size == 64
    assert config.eval is None


def test_default_reranker2_config():
    config = Reranker2Config()
    assert config.enabled is False
    assert config.model_name == "BAAI/bge-reranker-v2-m3"
    assert config.top_k == 10
    assert config.batch_size == 32
    assert config.max_content_chars == 32000
    assert config.eval is None


def test_default_selector_config():
    config = SelectorConfig()
    assert config.enabled is False
    assert config.model == "gpt-4o-mini"
    assert "system_v0.1.j2" in config.system_instruction
    assert "user_v0.1.j2" in config.user_instruction
    assert config.max_tokens == 1024
    assert config.temperature == 0.0
    assert config.top_k == 5
    assert config.cache_path == "outputs/selector_cache.json"
    assert config.eval is None


def test_config_defaults():
    config = Config()
    assert isinstance(config.system, SystemConfig)
    assert isinstance(config.index, IndexConfig)
    assert isinstance(config.models, ModelsConfig)
    assert config.index.input_corpus_path == "../skill-crawler/data/skills/"
    assert config.index.output_index_path == "outputs/indices/"
    assert isinstance(config.models.retriever, RetrieverConfig)
    assert isinstance(config.models.reranker, RerankerConfig)
    assert isinstance(config.models.reranker2, Reranker2Config)
    assert isinstance(config.models.selector, SelectorConfig)


def test_default_query_gen_config():
    config = QueryGenConfig()
    assert config.enabled is False
    assert config.model == "gpt-4o-mini"
    assert config.max_tokens == 200
    assert config.temperature == 0.0
    assert config.cache_path == "outputs/query_gen_cache.json"


def test_load_config_from_default():
    config = load_config()
    assert config.models.retriever.model_name == "BAAI/bge-base-en-v1.5"
    assert config.models.retriever.top_k == 1000

    assert config.models.reranker.enabled is True
    assert config.models.reranker.top_k == 100

    assert config.models.reranker2.enabled is True
    assert config.models.reranker2.max_content_chars == 32000
    assert config.models.reranker2.top_k == 10

    assert config.models.selector.enabled is True
    assert config.models.selector.model == "gpt-4o-mini"
    assert config.models.selector.top_k == 5


def test_load_config_custom(tmp_path):
    custom = {
        "models": {
            "retriever": {
                "model_name": "custom/model",
                "query_prompt": "custom: ",
                "batch_size": 64,
                "top_k": 50,
            },
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(custom))

    config = load_config(path)
    assert config.models.retriever.model_name == "custom/model"
    assert config.models.retriever.query_prompt == "custom: "
    assert config.models.retriever.batch_size == 64
    assert config.models.retriever.top_k == 50


def test_load_config_partial_override(tmp_path):
    """Missing keys should use defaults."""
    custom = {"models": {"retriever": {"batch_size": 128}}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(custom))

    config = load_config(path)
    assert config.models.retriever.batch_size == 128
    assert config.models.retriever.model_name == "BAAI/bge-base-en-v1.5"
    assert config.models.retriever.top_k == 100

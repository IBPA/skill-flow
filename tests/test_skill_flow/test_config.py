"""Tests for skill_flow.config."""

import json

import pytest
from skill_flow.config import (
    Config,
    DeepRerankerConfig,
    IndexConfig,
    ModelsConfig,
    QueryGenConfig,
    RerankerConfig,
    RetrieverConfig,
    RetrieverExperimentConfig,
    RetrieverVariant,
    SelectorConfig,
    SystemConfig,
    load_config,
)


def test_default_retriever_config():
    config = RetrieverConfig()
    assert config.model_name == "BAAI/bge-base-en-v1.5"
    assert "Represent this sentence" in config.query_prompt
    assert config.doc_prompt == ""
    assert config.retriever_type == "dense"
    assert config.batch_size == 256
    assert config.top_k == 100
    assert config.query_gen is None
    assert config.variants is None


def test_retriever_config_with_query_gen():
    qg = QueryGenConfig(enabled=True, model="gpt-4o-mini")
    config = RetrieverConfig(query_gen=qg)
    assert config.query_gen is not None
    assert config.query_gen.enabled is True
    assert config.query_gen.model == "gpt-4o-mini"


def test_retriever_config_bm25():
    config = RetrieverConfig(retriever_type="bm25")
    assert config.retriever_type == "bm25"


def test_retriever_config_doc_prompt():
    config = RetrieverConfig(doc_prompt="passage: ")
    assert config.doc_prompt == "passage: "


def test_default_reranker_config():
    config = RerankerConfig()
    assert config.enabled is False
    assert config.model_name == "BAAI/bge-reranker-v2-m3"
    assert config.input_top_k == 0
    assert config.batch_size == 64
    assert config.variants is None


def test_default_deep_reranker_config():
    config = DeepRerankerConfig()
    assert config.enabled is False
    assert config.model_name == "BAAI/bge-reranker-v2-m3"
    assert config.input_top_k == 0
    assert config.batch_size == 32
    assert config.max_content_chars == 32000
    assert config.variants is None


def test_default_selector_config():
    config = SelectorConfig()
    assert config.enabled is False
    assert config.model == "gpt-4o-mini"
    assert "system_v1.0.j2" in config.system_instruction
    assert "user_v0.1.j2" in config.user_instruction
    assert config.max_tokens == 1024
    assert config.temperature == 0.0
    assert config.input_top_k == 5
    assert config.cache_path == "outputs/selector_cache.json"
    assert config.variants is None


def test_config_defaults():
    config = Config()
    assert isinstance(config.system, SystemConfig)
    assert isinstance(config.index, IndexConfig)
    assert isinstance(config.models, ModelsConfig)
    assert config.index.input_corpus_path == "data/skills/"
    assert config.index.output_index_path == "outputs/indices/"
    assert isinstance(config.models.retriever, RetrieverConfig)
    assert isinstance(config.models.reranker, RerankerConfig)
    assert isinstance(config.models.deep_reranker, DeepRerankerConfig)
    assert isinstance(config.models.selector, SelectorConfig)


def test_default_query_gen_config():
    config = QueryGenConfig()
    assert config.enabled is False
    assert config.model == "gpt-4o-mini"
    assert config.max_tokens == 200
    assert config.temperature == 0.0
    assert config.cache_path == "outputs/query_gen_cache.json"
    assert config.aggregation == "max"


def test_query_gen_aggregation_list():
    config = QueryGenConfig(aggregation=["max", "mean", "rrf"])
    assert config.aggregation == ["max", "mean", "rrf"]


def test_query_gen_aggregation_union():
    config = QueryGenConfig(aggregation="union")
    assert config.aggregation == "union"


def test_query_gen_aggregation_union_in_list():
    config = QueryGenConfig(aggregation=["rrf", "union"])
    assert config.aggregation == ["rrf", "union"]


def test_query_gen_top_k_per_query_default():
    config = QueryGenConfig()
    assert config.top_k_per_query == 0


def test_query_gen_top_k_per_query_list():
    config = QueryGenConfig(top_k_per_query=[100, 200])
    assert config.top_k_per_query == [100, 200]


def test_query_gen_aggregation_single_string():
    config = QueryGenConfig(aggregation="rrf")
    assert config.aggregation == "rrf"


def test_query_gen_aggregation_invalid():
    with pytest.raises(ValueError, match="aggregation must be one of"):
        QueryGenConfig(aggregation="invalid")


def test_query_gen_aggregation_invalid_list_item():
    with pytest.raises(ValueError, match="aggregation must be one of"):
        QueryGenConfig(aggregation=["max", "invalid"])


def test_query_gen_aggregation_empty_list():
    with pytest.raises(ValueError, match="aggregation must not be empty"):
        QueryGenConfig(aggregation=[])


def test_load_config_from_default():
    config = load_config()
    assert config.models.retriever.model_name == "BAAI/bge-base-en-v1.5"
    assert config.models.retriever.top_k == 1000

    assert config.models.reranker.enabled is True
    assert config.models.reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.models.reranker.max_content_chars == 512

    assert config.models.deep_reranker.enabled is True
    assert config.models.deep_reranker.max_content_chars == 4096

    assert config.models.selector.enabled is True


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


def test_retriever_variant_defaults():
    v = RetrieverVariant()
    assert v.retriever_type == "dense"
    assert v.model_name == "BAAI/bge-base-en-v1.5"
    assert v.doc_prompt == ""
    assert v.top_k == 100
    assert v.index_dir == "outputs/indices/"
    assert v.query_gen is None


def test_retriever_variant_with_query_gen():
    qg = QueryGenConfig(enabled=True, num_queries=[1, 3], aggregation=["max", "rrf"])
    v = RetrieverVariant(query_gen=qg)
    assert v.query_gen is not None
    assert v.query_gen.num_queries == [1, 3]
    assert v.query_gen.aggregation == ["max", "rrf"]


def test_retriever_variant_bm25():
    v = RetrieverVariant(retriever_type="bm25", model_name="bm25")
    assert v.retriever_type == "bm25"


def test_retriever_experiment_config_defaults():
    c = RetrieverExperimentConfig()
    assert c.name == ""
    assert c.tasks_dir == "integration/skillsbench/tasks"
    assert c.max_tasks == 0
    assert c.include_tasks == []
    assert c.cache_dir == ""
    assert c.retrievers == []


def test_retriever_experiment_config_with_variants():
    c = RetrieverExperimentConfig(
        name="test",
        retrievers=[
            RetrieverVariant(model_name="a"),
            RetrieverVariant(model_name="b", retriever_type="bm25"),
        ],
    )
    assert len(c.retrievers) == 2
    assert c.retrievers[1].retriever_type == "bm25"

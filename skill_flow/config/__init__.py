"""Configuration models for SkillFlow, loaded from default.json."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

_CONFIG_DIR = Path(__file__).parent
_DEFAULT_PATH = _CONFIG_DIR / "default.json"


class SystemConfig(BaseModel):
    """Placeholder for future system-level settings."""


class IndexConfig(BaseModel):
    input_corpus_path: str = "data/skills/"
    output_index_path: str = "outputs/indices/"


_DEFAULT_QUERY_GEN_PROMPT = (
    "You are a search query generator. Given a detailed task"
    " instruction for a coding agent, generate a concise search"
    " query (1-2 sentences, under 200 characters) that captures"
    " the core technical skill needed. Focus on the primary"
    " technology, tool, or technique required. Omit file paths,"
    " specific data values, and implementation details."
)


class QueryGenConfig(BaseModel):
    """LLM-based query generation for retrieval and reranking stages."""

    enabled: bool = False
    model: str = "gpt-4o-mini"
    system_prompt: str = _DEFAULT_QUERY_GEN_PROMPT
    multi_query_prompt: str = "skill_flow/query_gen/instructions/multi_v2.txt"
    max_tokens: int = 200
    temperature: float = 0.0
    cache_path: str = "outputs/query_gen_cache.json"
    num_queries: int | list[int] = 1
    aggregation: str | list[str] = "max"
    top_k_per_query: int | list[int] = 0

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str | list[str]) -> str | list[str]:
        """Ensure each aggregation value is one of 'max', 'mean', 'rrf', 'union'."""
        valid = {"max", "mean", "rrf", "union"}
        items = v if isinstance(v, list) else [v]
        if not items:
            msg = "aggregation must not be empty"
            raise ValueError(msg)
        for item in items:
            if item not in valid:
                msg = f"aggregation must be one of {valid}, got {item!r}"
                raise ValueError(msg)
        return v


class RetrieverConfig(BaseModel):
    model_name: str = "BAAI/bge-base-en-v1.5"
    query_prompt: str = "Represent this sentence for searching relevant passages: "
    doc_prompt: str = ""
    retriever_type: Literal["dense", "bm25"] = "dense"
    batch_size: int = 256
    top_k: int = 100
    query_gen: QueryGenConfig | None = None
    variants: list[dict[str, object]] | None = None


class RerankerConfig(BaseModel):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-v2-m3"
    input_top_k: int = 0
    batch_size: int = 64
    max_content_chars: int = 0
    query_gen: QueryGenConfig | None = None
    variants: list[dict[str, object]] | None = None


class DeepRerankerConfig(BaseModel):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-v2-m3"
    input_top_k: int = 0
    batch_size: int = 32
    max_content_chars: int = 32000
    query_gen: QueryGenConfig | None = None
    variants: list[dict[str, object]] | None = None


_SELECTOR_INSTRUCTIONS_DIR = "skill_flow/selector/instructions"


class SelectorConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o-mini"
    system_instruction: str = f"{_SELECTOR_INSTRUCTIONS_DIR}/system_v1.0.j2"
    user_instruction: str = f"{_SELECTOR_INSTRUCTIONS_DIR}/user_v0.1.j2"
    specificity_instruction: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0
    input_top_k: int = 5
    max_selected: int = 0
    max_workers: int = 1
    cache_path: str = "outputs/selector_cache.json"
    specificity_cache_path: str = "outputs/specificity_cache.json"
    variants: list[dict[str, object]] | None = None


class ModelsConfig(BaseModel):
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    deep_reranker: DeepRerankerConfig = DeepRerankerConfig()
    selector: SelectorConfig = SelectorConfig()


class PipelineConfig(BaseModel):
    """Configuration for the ``pipeline`` CLI subcommand."""

    tasks_dir: str = "integration/terminal-bench/tasks"
    output_dir: str = "outputs/pipeline/terminal-bench"
    max_tasks: int = 0
    include_tasks: list[str] = []
    max_query_chars: int = 0


class Config(BaseModel):
    system: SystemConfig = SystemConfig()
    index: IndexConfig = IndexConfig()
    models: ModelsConfig = ModelsConfig()
    pipeline: PipelineConfig = PipelineConfig()


# ---------------------------------------------------------------------------
# Experiment variant models (used by experiment runners)
# ---------------------------------------------------------------------------


class RetrieverVariant(BaseModel):
    """Configuration for a single retriever in an experiment."""

    enabled: bool = True
    retriever_type: Literal["dense", "bm25"] = "dense"
    model_name: str = "BAAI/bge-base-en-v1.5"
    query_prompt: str = "Represent this sentence for searching relevant passages: "
    doc_prompt: str = ""
    top_k: int = 100
    batch_size: int = 256
    device: str = ""
    index_dir: str = "outputs/indices/"
    max_content_tokens: int = 0
    query_gen: QueryGenConfig | None = None


class RetrieverExperimentConfig(BaseModel):
    """Configuration for a multi-retriever comparison experiment."""

    name: str = ""
    tasks_dir: str = "integration/skillsbench/tasks"
    max_tasks: int = 0
    include_tasks: list[str] = []
    output_dir: str = "outputs/experiments/"
    corpus_path: str = "data/skills/"
    cache_dir: str = ""
    retrievers: list[RetrieverVariant] = []


class RerankerVariant(BaseModel):
    """Configuration for a single reranker in an experiment.

    ``max_content_chars`` accepts a list for grid search in experiment mode.
    """

    enabled: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"
    input_top_k: int = 0
    batch_size: int = 64
    max_content_chars: int | list[int] = 0
    query_gen: QueryGenConfig | None = None


class RerankerExperimentConfig(BaseModel):
    """Configuration for a multi-reranker comparison experiment."""

    name: str = ""
    tasks_dir: str = "integration/skillsbench/tasks"
    max_tasks: int = 0
    include_tasks: list[str] = []
    output_dir: str = "outputs/experiments/"
    input_report_path: str = "outputs/eval-retriever-results.json"
    index_dir: str = "outputs/indices/"
    cache_dir: str = ""
    rerankers: list[RerankerVariant] = []


class SelectorVariant(BaseModel):
    """Configuration for a single selector in an experiment.

    ``input_top_k`` accepts a list for grid search in experiment mode.
    """

    enabled: bool = True
    model: str = "gpt-4o-mini"
    system_instruction: str = f"{_SELECTOR_INSTRUCTIONS_DIR}/system_v1.0.j2"
    user_instruction: str = f"{_SELECTOR_INSTRUCTIONS_DIR}/user_v0.1.j2"
    specificity_instruction: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0
    input_top_k: int | list[int] = 5
    max_selected: int = 0
    specificity_cache_path: str = "outputs/specificity_cache.json"


class SelectorExperimentConfig(BaseModel):
    """Configuration for a multi-selector comparison experiment."""

    name: str = ""
    tasks_dir: str = "integration/skillsbench/tasks"
    max_tasks: int = 0
    include_tasks: list[str] = []
    output_dir: str = "outputs/experiments/"
    input_report_path: str = "outputs/eval-deep-reranker-results.json"
    index_dir: str = "outputs/indices/"
    cache_dir: str = ""
    selectors: list[SelectorVariant] = []


def load_config(path: Path | None = None) -> Config:
    """Load config from a JSON file, falling back to defaults."""
    config_path = path or _DEFAULT_PATH
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return Config.model_validate(data)

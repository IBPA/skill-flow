"""Configuration models for SkillFlow, loaded from default.json."""

import json
from pathlib import Path

from pydantic import BaseModel

_CONFIG_DIR = Path(__file__).parent
_DEFAULT_PATH = _CONFIG_DIR / "default.json"


class SystemConfig(BaseModel):
    """Placeholder for future system-level settings."""


class IndexConfig(BaseModel):
    input_corpus_path: str = "../skill-crawler/data/skills/"
    output_index_path: str = "outputs/indices/"


class RetrieverEvalSettings(BaseModel):
    """JSON-level eval settings nested under models.retriever."""

    enabled: bool = True
    tasks_dir: str = "integration/skillsbench/tasks"
    max_query_chars: int = 0
    output_path: str = "outputs/eval-retriever-results.json"


class RetrieverConfig(BaseModel):
    model_name: str = "BAAI/bge-base-en-v1.5"
    query_prompt: str = "Represent this sentence for searching relevant passages: "
    batch_size: int = 256
    top_k: int = 100
    eval: RetrieverEvalSettings | None = None


_DEFAULT_QUERY_GEN_PROMPT = (
    "You are a search query generator. Given a detailed task"
    " instruction for a coding agent, generate a concise search"
    " query (1-2 sentences, under 200 characters) that captures"
    " the core technical skill needed. Focus on the primary"
    " technology, tool, or technique required. Omit file paths,"
    " specific data values, and implementation details."
)


class QueryGenConfig(BaseModel):
    """LLM-based query generation for cross-encoder reranking."""

    enabled: bool = False
    model: str = "gpt-4o-mini"
    system_prompt: str = _DEFAULT_QUERY_GEN_PROMPT
    max_tokens: int = 200
    temperature: float = 0.0
    cache_path: str = "outputs/query_gen_cache.json"


class RerankerEvalSettings(BaseModel):
    """JSON-level eval settings nested under models.reranker."""

    enabled: bool = True
    stage1_report_path: str = "outputs/eval-retriever-results.json"
    tasks_dir: str = "integration/skillsbench/tasks"
    output_path: str = "outputs/eval-reranker-results.json"


class RerankerConfig(BaseModel):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 10
    batch_size: int = 64
    max_content_chars: int = 0
    query_gen: QueryGenConfig | None = None
    eval: RerankerEvalSettings | None = None


class Reranker2EvalSettings(BaseModel):
    """JSON-level eval settings nested under models.reranker2."""

    enabled: bool = True
    stage2_report_path: str = "outputs/eval-reranker-results.json"
    tasks_dir: str = "integration/skillsbench/tasks"
    output_path: str = "outputs/eval-reranker2-results.json"


class Reranker2Config(BaseModel):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 10
    batch_size: int = 32
    max_content_chars: int = 32000
    query_gen: QueryGenConfig | None = None
    eval: Reranker2EvalSettings | None = None


_SELECTOR_INSTRUCTIONS_DIR = "skill_flow/selector/instructions"


class SelectorEvalSettings(BaseModel):
    """JSON-level eval settings nested under models.selector."""

    enabled: bool = True
    stage3_report_path: str = "outputs/eval-reranker2-results.json"
    tasks_dir: str = "integration/skillsbench/tasks"
    output_path: str = "outputs/eval-selector-results.json"


class SelectorConfig(BaseModel):
    enabled: bool = False
    model: str = "gpt-4o-mini"
    system_instruction: str = f"{_SELECTOR_INSTRUCTIONS_DIR}/system_v0.1.j2"
    user_instruction: str = f"{_SELECTOR_INSTRUCTIONS_DIR}/user_v0.1.j2"
    max_tokens: int = 1024
    temperature: float = 0.0
    top_k: int = 5
    cache_path: str = "outputs/selector_cache.json"
    eval: SelectorEvalSettings | None = None


class ModelsConfig(BaseModel):
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    reranker2: Reranker2Config = Reranker2Config()
    selector: SelectorConfig = SelectorConfig()


class Config(BaseModel):
    system: SystemConfig = SystemConfig()
    index: IndexConfig = IndexConfig()
    models: ModelsConfig = ModelsConfig()


def load_config(path: Path | None = None) -> Config:
    """Load config from a JSON file, falling back to defaults."""
    config_path = path or _DEFAULT_PATH
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return Config.model_validate(data)

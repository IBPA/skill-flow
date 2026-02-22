"""Pydantic models for retriever evaluation."""

from pathlib import Path

from pydantic import BaseModel

from skill_flow.config import (
    Reranker2Config,
    RerankerConfig,
    RetrieverConfig,
    SelectorConfig,
)

EVAL_KS: list[int] = [1, 2, 5, 10, 50, 100, 500, 1000]


def filter_ks(top_k: int) -> list[int]:
    """Return EVAL_KS values that are <= top_k."""
    return [k for k in EVAL_KS if k <= top_k]


class RerankerEvalConfig(BaseModel):
    """Runtime configuration for a Stage 2-only reranker evaluation."""

    stage1_report_path: Path
    tasks_dir: Path
    index_dir: Path = Path("outputs/indices/")
    reranker: RerankerConfig = RerankerConfig(enabled=True)
    max_tasks: int = 0
    output_path: Path | None = None


class Reranker2EvalConfig(BaseModel):
    """Runtime configuration for a Stage 3 reranker2 evaluation."""

    stage2_report_path: Path
    tasks_dir: Path
    index_dir: Path = Path("outputs/indices/")
    reranker: Reranker2Config = Reranker2Config(enabled=True)
    max_tasks: int = 0
    output_path: Path | None = None


class SelectorEvalConfig(BaseModel):
    """Runtime configuration for a Stage 4 selector evaluation."""

    stage3_report_path: Path
    tasks_dir: Path
    index_dir: Path = Path("outputs/indices/")
    selector: SelectorConfig = SelectorConfig(enabled=True)
    max_tasks: int = 0
    output_path: Path | None = None


class EvalRunConfig(BaseModel):
    """Configuration for a retriever evaluation run."""

    tasks_dir: Path
    index_dir: Path = Path("outputs/indices/")
    retriever: RetrieverConfig = RetrieverConfig()
    max_query_chars: int = 0
    max_tasks: int = 0
    output_path: Path | None = None
    reranker: RerankerConfig | None = None


class InjectedSkill(BaseModel, frozen=True):
    """A ground-truth skill to inject into the index for evaluation."""

    key: str
    name: str
    description: str
    content: str = ""


class TaskGroundTruth(BaseModel, frozen=True):
    """Ground truth for a single evaluation task."""

    task_id: str
    query: str
    ground_truth_keys: list[str]
    all_skill_names: list[str]
    injected_skills: list[str]


class RetrievedSkill(BaseModel, frozen=True):
    """A retrieved skill with its score and content for diagnostics."""

    key: str
    score: float
    description: str = ""
    content: str = ""


class TaskResult(BaseModel, frozen=True):
    """Evaluation result for a single task."""

    task_id: str
    query: str = ""
    rerank_query: str = ""
    num_ground_truth: int
    num_injected: int
    retrieved_skills: list[RetrievedSkill] = []
    recall_at: dict[int, float]
    precision_at: dict[int, float] = {}
    hit_at: dict[int, float]
    reciprocal_rank: float


class EvalSummary(BaseModel, frozen=True):
    """Aggregated evaluation metrics across all tasks."""

    num_tasks_total: int
    num_tasks_evaluated: int
    num_tasks_no_skills: int
    num_skills_injected: int
    mean_recall_at: dict[int, float]
    mean_precision_at: dict[int, float] = {}
    mean_hit_at: dict[int, float]
    mrr: float


class EvalReport(BaseModel):
    """Complete evaluation report with summary and per-task results."""

    summary: EvalSummary
    task_results: list[TaskResult]
    config: (
        EvalRunConfig
        | RerankerEvalConfig
        | Reranker2EvalConfig
        | SelectorEvalConfig
    )

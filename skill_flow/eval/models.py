"""Pydantic models for retriever evaluation."""

from pydantic import BaseModel

EVAL_KS: list[int] = [1, 2, 5, 10, 50, 100, 500, 1000]


def filter_ks(top_k: int) -> list[int]:
    """Return EVAL_KS values that are <= top_k, or all if top_k is 0."""
    if top_k <= 0:
        return list(EVAL_KS)
    return [k for k in EVAL_KS if k <= top_k]


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
    query_scores: list[float] = []


class PerQueryResult(BaseModel, frozen=True):
    """Per-query breakdown of retrieved skills for multi-query retrieval."""

    query: str
    retrieved_skills: list[RetrievedSkill]


class TaskResult(BaseModel, frozen=True):
    """Evaluation result for a single task."""

    task_id: str
    query: str = ""
    retrieval_query: str = ""
    rerank_query: str = ""
    num_ground_truth: int
    num_injected: int
    retrieved_skills: list[RetrievedSkill] = []
    retrieval_queries: list[PerQueryResult] = []
    recall_at: dict[int, float]
    precision_at: dict[int, float] = {}
    hit_at: dict[int, float]
    reciprocal_rank: float
    select_recall: float = 0.0
    select_precision: float = 0.0
    elapsed_ms: float = 0.0


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
    mean_select_recall: float = 0.0
    mean_select_precision: float = 0.0
    mean_elapsed_ms: float = 0.0


class EvalReport(BaseModel):
    """Complete evaluation report with summary and per-task results."""

    summary: EvalSummary
    task_results: list[TaskResult]
    config: dict[str, object] = {}

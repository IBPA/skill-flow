"""Tests for skill_flow.eval.utils."""

from skill_flow.eval.models import PerQueryResult, RetrievedSkill, TaskResult
from skill_flow.eval.utils.helpers import slug
from skill_flow.eval.utils.reporting import _strip_skill_content


class TestSlug:
    def test_basic(self) -> None:
        assert slug("BAAI/bge-base-en-v1.5") == "baai-bge-base-en-v1-5"

    def test_simple(self) -> None:
        assert slug("bm25") == "bm25"

    def test_spaces(self) -> None:
        assert slug("My Model Name") == "my-model-name"


class TestStripSkillContent:
    def test_preserves_query_scores(self) -> None:
        task_results = [
            TaskResult(
                task_id="t1",
                num_ground_truth=1,
                num_injected=1,
                retrieved_skills=[
                    RetrievedSkill(
                        key="s1",
                        score=0.9,
                        description="desc",
                        content="# Full SKILL.md",
                        query_scores=[0.9, 0.5],
                    ),
                ],
                recall_at={1: 1.0},
                hit_at={1: 1.0},
                reciprocal_rank=1.0,
            )
        ]
        stripped = _strip_skill_content(task_results)
        skill = stripped[0].retrieved_skills[0]
        assert skill.query_scores == [0.9, 0.5]
        assert skill.content == ""

    def test_strips_retrieval_queries_content(self) -> None:
        task_results = [
            TaskResult(
                task_id="t1",
                num_ground_truth=1,
                num_injected=1,
                retrieved_skills=[],
                retrieval_queries=[
                    PerQueryResult(
                        query="q1",
                        retrieved_skills=[
                            RetrievedSkill(
                                key="s1",
                                score=0.9,
                                description="desc",
                                content="# Full content",
                                query_scores=[0.9],
                            ),
                        ],
                    ),
                ],
                recall_at={1: 1.0},
                hit_at={1: 1.0},
                reciprocal_rank=1.0,
            )
        ]
        stripped = _strip_skill_content(task_results)
        pq = stripped[0].retrieval_queries[0]
        assert pq.query == "q1"
        assert pq.retrieved_skills[0].content == ""
        assert pq.retrieved_skills[0].query_scores == [0.9]

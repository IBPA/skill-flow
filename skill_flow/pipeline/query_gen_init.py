"""Query generator initialization and validation for pipeline stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skill_flow.query_gen.query_gen import QueryGenerator

if TYPE_CHECKING:
    from skill_flow.config import QueryGenConfig


def init_retriever_query_gen(
    qg_config: QueryGenConfig | None,
    cache_path: str = "",
) -> tuple[QueryGenerator | None, str, int]:
    """Validate config (reject lists), build QueryGenerator.

    Returns (generator, aggregation, top_k_per_query).
    """
    if not qg_config or not qg_config.enabled:
        return None, "max", 0
    for field_name, field_type in [
        ("aggregation", "a single string"),
        ("num_queries", "a single int"),
        ("top_k_per_query", "a single int"),
    ]:
        if isinstance(getattr(qg_config, field_name), list):
            msg = (
                f"{field_name} must be {field_type} in pipeline/eval mode; "
                "use experiment mode for grid search"
            )
            raise TypeError(msg)
    assert isinstance(qg_config.aggregation, str)
    assert isinstance(qg_config.top_k_per_query, int)
    agg = qg_config.aggregation
    tkpq = qg_config.top_k_per_query
    if agg == "union" and tkpq <= 0:
        msg = "top_k_per_query must be > 0 when aggregation='union'"
        raise ValueError(msg)
    config_to_use = qg_config
    if cache_path:
        config_to_use = qg_config.model_copy(
            update={"cache_path": cache_path},
        )
    return QueryGenerator(config_to_use), agg, tkpq


def init_rerank_query_gen(
    qg_config: QueryGenConfig | None,
    cache_path: str = "",
) -> tuple[QueryGenerator | None, str]:
    """Validate config (reject lists), build QueryGenerator for rerank."""
    if not qg_config or not qg_config.enabled:
        return None, "max"
    for field in ("aggregation", "num_queries"):
        if isinstance(getattr(qg_config, field), list):
            msg = f"{field} must be scalar in pipeline/eval; use experiment mode"
            raise TypeError(msg)
    config_to_use = qg_config
    if cache_path:
        config_to_use = qg_config.model_copy(
            update={"cache_path": cache_path},
        )
    assert isinstance(qg_config.aggregation, str)
    return QueryGenerator(config_to_use), qg_config.aggregation

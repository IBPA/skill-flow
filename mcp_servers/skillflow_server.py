"""SkillFlow retriever MCP server with skill download endpoint.

Supports two modes:
- **Live**: runs the full GPU pipeline (FAISS -> reranker -> selector)
- **Cached** (``--cached``): no GPU. The benchmark agent sends skill keys
  per task via ``POST /set-task``, and ``retrieve_skill`` returns those.

Usage (live):
    uv run python -m mcp_servers.skillflow_server \\
        --port 8765 --base-url https://my-domain.ngrok-free.dev

Usage (cached):
    uv run python -m mcp_servers.skillflow_server \\
        --port 8765 --base-url https://my-domain.ngrok-free.dev --cached
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP
from skill_flow.config import Config, load_config
from skill_flow.index.encoder import Encoder
from skill_flow.models.core import SkillFlow
from skill_flow.reranker.reranker import Reranker
from skill_flow.retriever.retriever import IndexSearcher, SearchResult
from skill_flow.selector.selector import Selector
from starlette.responses import Response

from mcp_servers.utils.server_helpers import (
    RETRIEVE_SKILL_DOC,
    create_tar_gz,
    format_results,
    log_query,
    resolve_skill_folder,
    skill_name,
)

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)


# -- Pipeline init (live mode only) ----------------------------------------


def _init_pipeline(config: Config, index_dir: Path) -> SkillFlow:
    """Build the full retrieval pipeline from config."""
    encoder = Encoder(config.models.retriever)
    searcher = IndexSearcher(index_dir, encoder, config.models.retriever)

    reranker = (
        Reranker(config.models.reranker) if config.models.reranker.enabled else None
    )
    deep_reranker = (
        Reranker(config.models.deep_reranker)
        if reranker and config.models.deep_reranker.enabled
        else None
    )
    selector = (
        Selector(config.models.selector)
        if deep_reranker and config.models.selector.enabled
        else None
    )
    return SkillFlow(searcher, reranker, deep_reranker, selector)


# -- MCP tool registration -------------------------------------------------


def _register_tools(
    mcp: FastMCP,
    pipeline: SkillFlow,
    corpus_dir: Path,
    base_url: str,
    top_k: int,
    log_file: Path,
) -> None:
    """Register the live retrieve_skill MCP tool."""

    @mcp.tool(description=RETRIEVE_SKILL_DOC)
    def retrieve_skill(query: str) -> str:
        start = time.perf_counter()
        results = pipeline.search(query)[:top_k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        log_query(query, results, elapsed_ms, log_file)
        logger.info(
            "Query: %s -> %d results in %.0f ms",
            query[:80],
            len(results),
            elapsed_ms,
        )
        return format_results(results, base_url, corpus_dir)


def _register_cached_tools(
    mcp: FastMCP,
    task_state: dict[str, list[str]],
    corpus_dir: Path,
    base_url: str,
    log_file: Path,
    tasks_dir: Path | None = None,
) -> None:
    """Register retrieve_skill backed by per-task skill keys from /set-task."""

    @mcp.tool(description=RETRIEVE_SKILL_DOC)
    def retrieve_skill(query: str) -> str:
        start = time.perf_counter()
        keys = task_state.get("_keys", [])
        tid = task_state.get("_task_id", [""])
        task_id = tid[0] if tid else ""
        if not keys:
            logger.warning("No cached keys for task_id=%s", task_id)
            return "No matching skills found for this query."

        results = [SearchResult(key=k, score=1.0) for k in keys]
        elapsed_ms = (time.perf_counter() - start) * 1000

        log_query(query, results, elapsed_ms, log_file)
        logger.info(
            "Cached: task=%s, query=%s -> %d results",
            task_id,
            query[:80],
            len(results),
        )
        return format_results(results, base_url, corpus_dir, tasks_dir)


# -- HTTP routes ------------------------------------------------------------


def _register_routes(
    mcp: FastMCP,
    corpus_dir: Path,
    task_state: dict[str, list[str]] | None = None,
    tasks_dir: Path | None = None,
) -> None:
    """Register HTTP endpoints (download + optional set-task)."""

    @mcp.custom_route("/download/{key:path}", methods=["GET"])
    async def download_skill(request: Request) -> Response:
        key = request.path_params["key"]
        folder = resolve_skill_folder(key, corpus_dir, tasks_dir)
        if not folder:
            return Response(
                content=f"Skill not found: {key}",
                status_code=404,
                media_type="text/plain",
            )
        data = create_tar_gz(folder)
        name = skill_name(key)
        return Response(
            content=data,
            media_type="application/gzip",
            headers={"Content-Disposition": f"attachment; filename={name}.tar.gz"},
        )

    if task_state is not None:

        @mcp.custom_route("/set-task", methods=["POST"])
        async def set_task(request: Request) -> Response:
            body = await request.json()
            tid = body.get("task_id", "")
            keys: list[str] = body.get("skill_keys", [])
            task_state["_task_id"] = [tid]
            task_state["_keys"] = keys
            logger.info("Task set: %s (%d keys)", tid, len(keys))
            return Response(
                content=json.dumps({"task_id": tid, "n_keys": len(keys)}),
                media_type="application/json",
            )


# -- Entry point ------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SkillFlow retriever MCP server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--config", default=None, help="SkillFlow config JSON path")
    p.add_argument("--index-dir", default=None, help="FAISS index directory")
    p.add_argument("--corpus-dir", default=None, help="Skill corpus directory")
    p.add_argument("--top-k", type=int, default=3, help="Max skills to return")
    p.add_argument(
        "--base-url",
        required=True,
        help="External base URL for download links",
    )
    p.add_argument("--log-file", default="skillflow_server.jsonl")
    p.add_argument(
        "--cached",
        action="store_true",
        help="Cached mode: no GPU, skill keys pushed via /set-task",
    )
    p.add_argument(
        "--tasks-dir",
        default=None,
        help="SkillsBench tasks directory (for skillsbench/ key resolution)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    corpus_dir = Path(args.corpus_dir or config.index.input_corpus_path)
    base_url = args.base_url.rstrip("/")
    log_file = Path(args.log_file)

    mcp = FastMCP("skillflow", host=args.host, port=args.port)

    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else None

    if args.cached:
        task_state: dict[str, list[str]] = {}
        _register_cached_tools(
            mcp,
            task_state,
            corpus_dir,
            base_url,
            log_file,
            tasks_dir,
        )
        _register_routes(mcp, corpus_dir, task_state, tasks_dir)
        logger.info("Cached server ready: port=%d, corpus=%s", args.port, corpus_dir)
    else:
        index_dir = Path(args.index_dir or config.index.output_index_path)
        logger.info("Initializing SkillFlow pipeline ...")
        pipeline = _init_pipeline(config, index_dir)
        _register_tools(mcp, pipeline, corpus_dir, base_url, args.top_k, log_file)
        _register_routes(mcp, corpus_dir)
        logger.info("Live server ready: port=%d, top_k=%d", args.port, args.top_k)

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

"""Live SkillFlow retriever MCP server with skill download endpoint.

Runs the full SkillFlow pipeline (FAISS -> reranker -> reranker2 -> selector)
and serves skill folders as tar.gz downloads. The agent calls retrieve_skill
with a query, gets download commands, and executes them to save skills locally.

Usage:
    uv run python -m mcp_servers.skillflow_retriever_server \
        --port 8765 \
        --base-url https://my-ngrok-domain.ngrok-free.dev
"""

import argparse
import io
import json
import logging
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from skill_flow.config import Config, load_config
from skill_flow.index.encoder import Encoder
from skill_flow.models.core import SkillFlow
from skill_flow.reranker.reranker import Reranker
from skill_flow.retriever.retriever import IndexSearcher, SearchResult
from skill_flow.selector.selector import Selector
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

CONTAINER_SKILLS_DIR = "/logs/agent/skills"


def _skill_name(key: str) -> str:
    """Extract the short skill name from a key (e.g. 'skillsmp/foo' -> 'foo')."""
    return key.rsplit("/", maxsplit=1)[-1]


def _create_tar_gz(folder_path: Path) -> bytes:
    """Create a tar.gz archive of a skill folder's contents."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for child in sorted(folder_path.iterdir()):
            tar.add(str(child), arcname=child.name)
    return buf.getvalue()


def _format_results(
    results: list[SearchResult],
    base_url: str,
    corpus_dir: Path,
) -> str:
    """Format pipeline results as download commands for the agent."""
    if not results:
        return "No matching skills found for this query."

    valid: list[tuple[SearchResult, str]] = []
    for r in results:
        name = _skill_name(r.key)
        folder = corpus_dir / r.key
        if folder.is_dir():
            valid.append((r, name))
        else:
            logger.warning("Skill folder not found: %s", folder)

    if not valid:
        return "No matching skills found for this query."

    lines = [f"Found {len(valid)} skills. Download them by running:", ""]
    paths: list[str] = []
    for r, name in valid:
        path = f"{CONTAINER_SKILLS_DIR}/{name}"
        url = f"{base_url}/download/{r.key}"
        lines.append(f'mkdir -p {path} && curl -sL "{url}" | tar xz -C {path}')
        paths.append(path)

    lines.append("")
    lines.append("Use skills at:")
    for path in paths:
        lines.append(f"- {path}")

    return "\n".join(lines)


def _log_query(
    query: str,
    results: list[SearchResult],
    latency_ms: float,
    log_file: Path,
) -> None:
    """Append a JSONL log entry for a retrieval query."""
    entry = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "query": query,
        "retrieved_skills": [
            {"key": r.key, "score": round(r.score, 4)} for r in results
        ],
        "n_results": len(results),
        "latency_ms": round(latency_ms, 1),
    }
    with log_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _init_pipeline(config: Config, index_dir: Path) -> SkillFlow:
    """Build the full retrieval pipeline from config."""
    encoder = Encoder(config.models.retriever)
    searcher = IndexSearcher(index_dir, encoder, config.models.retriever)

    reranker = (
        Reranker(config.models.reranker) if config.models.reranker.enabled else None
    )
    reranker2 = (
        Reranker(config.models.reranker2)
        if reranker and config.models.reranker2.enabled
        else None
    )
    selector = (
        Selector(config.models.selector)
        if reranker2 and config.models.selector.enabled
        else None
    )
    return SkillFlow(searcher, reranker, reranker2, selector)


def _register_tools(
    mcp: FastMCP,
    pipeline: SkillFlow,
    corpus_dir: Path,
    base_url: str,
    top_k: int,
    log_file: Path,
) -> None:
    """Register the retrieve_skill MCP tool."""

    @mcp.tool()
    def retrieve_skill(query: str) -> str:
        """Search for an expert skill before attempting any task.

        IMPORTANT: You MUST call this tool at the start of every task,
        before writing any code or running any commands. Expert skills
        contain critical guidance, shortcuts, and pitfall warnings that
        dramatically improve success rates.

        Even if a task seems straightforward, call this tool — there may
        be non-obvious tricks or common failure modes you'd otherwise
        miss.

        Args:
            query: Describe what you're trying to accomplish. Be specific
                about the domain, technology, and goal.
        """
        start = time.perf_counter()
        results = pipeline.search(query)
        results = results[:top_k]
        elapsed_ms = (time.perf_counter() - start) * 1000

        _log_query(query, results, elapsed_ms, log_file)
        logger.info(
            "Query: %s -> %d results in %.0f ms",
            query[:80],
            len(results),
            elapsed_ms,
        )

        return _format_results(results, base_url, corpus_dir)


def _register_routes(mcp: FastMCP, corpus_dir: Path) -> None:
    """Register the skill download HTTP endpoint."""

    @mcp.custom_route("/download/{key:path}", methods=["GET"])
    async def download_skill(request: Request) -> Response:
        key = request.path_params["key"]
        folder = corpus_dir / key

        if not folder.is_dir():
            return Response(
                content=f"Skill not found: {key}",
                status_code=404,
                media_type="text/plain",
            )

        data = _create_tar_gz(folder)
        name = _skill_name(key)
        return Response(
            content=data,
            media_type="application/gzip",
            headers={
                "Content-Disposition": f"attachment; filename={name}.tar.gz",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SkillFlow retriever MCP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", default=None, help="SkillFlow config JSON path")
    parser.add_argument(
        "--index-dir",
        default=None,
        help="FAISS index directory (default: from config)",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Skill corpus directory (default: from config)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Max skills to return")
    parser.add_argument(
        "--base-url",
        required=True,
        help="External base URL for download links (e.g. https://x.ngrok-free.dev)",
    )
    parser.add_argument(
        "--log-file",
        default="skillflow_retriever.jsonl",
        help="Path to JSONL query log",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    index_dir = Path(args.index_dir or config.index.output_index_path)
    corpus_dir = Path(args.corpus_dir or config.index.input_corpus_path)
    base_url = args.base_url.rstrip("/")
    log_file = Path(args.log_file)

    logger.info("Initializing SkillFlow pipeline (this may take a moment)...")
    pipeline = _init_pipeline(config, index_dir)

    logger.info(
        "Server ready: port=%d, corpus=%s, top_k=%d",
        args.port,
        corpus_dir,
        args.top_k,
    )

    mcp = FastMCP("skillflow", host=args.host, port=args.port)
    _register_tools(mcp, pipeline, corpus_dir, base_url, args.top_k, log_file)
    _register_routes(mcp, corpus_dir)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

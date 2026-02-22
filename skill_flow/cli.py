"""CLI entry point for SkillFlow index operations."""

import argparse
import logging
from pathlib import Path

from skill_flow.config import Config, load_config
from skill_flow.corpus.loader import load_corpus
from skill_flow.eval.cli_eval import run_eval
from skill_flow.index.builder import build_index
from skill_flow.index.encoder import Encoder
from skill_flow.models.core import SkillFlow
from skill_flow.reranker.reranker import Reranker
from skill_flow.retriever.retriever import IndexSearcher
from skill_flow.selector.selector import Selector


def _build_index(args: argparse.Namespace, config: Config) -> None:
    corpus_path = Path(args.corpus_path or config.index.input_corpus_path)
    output_dir = Path(args.output_dir or config.index.output_index_path)

    skills = load_corpus(corpus_path)
    encoder = Encoder(config.models.retriever)
    build_index(
        skills, encoder, output_dir, batch_size=args.batch_size, corpus_path=corpus_path
    )


def _search(args: argparse.Namespace, config: Config) -> None:
    index_dir = Path(args.index_dir or config.index.output_index_path)
    encoder = Encoder(config.models.retriever)
    searcher = IndexSearcher(index_dir, encoder, config.models.retriever)

    rerank_enabled = (
        args.rerank if args.rerank is not None else config.models.reranker.enabled
    )
    reranker = Reranker(config.models.reranker) if rerank_enabled else None
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
    retriever = SkillFlow(searcher, reranker, reranker2, selector)

    results = retriever.search(args.query, top_k=args.top_k)
    for i, r in enumerate(results, 1):
        print(f"{i:>3}. [{r.score:.4f}] {r.key}")


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = argparse.ArgumentParser(
        prog="skillflow",
        description="SkillFlow index CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared --config flag added to each subcommand
    config_arg = {
        "default": None,
        "help": "Path to config JSON (default: skill_flow/config/default.json)",
    }

    build_p = sub.add_parser("build-index", help="Build FAISS index from corpus")
    build_p.add_argument("--config", **config_arg)
    build_p.add_argument(
        "--corpus-path",
        default=None,
        help="Path to corpus directory (default: from config)",
    )
    build_p.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write index artifacts (default: from config)",
    )
    build_p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Encoding batch size",
    )

    search_p = sub.add_parser("search", help="Search the FAISS index")
    search_p.add_argument("--config", **config_arg)
    search_p.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing index artifacts (default: from config)",
    )
    search_p.add_argument("--query", required=True, help="Search query")
    search_p.add_argument("--top-k", type=int, default=10, help="Number of results")
    search_p.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable cross-encoder reranking (overrides config)",
    )

    eval_p = sub.add_parser("eval", help="Evaluate retrieval pipeline")
    eval_p.add_argument("--config", **config_arg)
    eval_p.add_argument(
        "--tasks-dir",
        default=None,
        help="SkillsBench tasks directory (default: from config)",
    )
    eval_p.add_argument(
        "--index-dir",
        default=None,
        help="Index directory (default: from config)",
    )
    eval_p.add_argument(
        "--max-query-chars",
        type=int,
        default=0,
        help="Truncate queries (0=no limit)",
    )
    eval_p.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Limit number of tasks to evaluate (0=all)",
    )
    eval_p.add_argument("--output", default=None, help="Output JSON report path")
    eval_p.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable cross-encoder reranking (overrides config)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    if args.command == "build-index":
        _build_index(args, config)
    elif args.command == "search":
        _search(args, config)
    elif args.command == "eval":
        run_eval(args, config)


if __name__ == "__main__":
    main()

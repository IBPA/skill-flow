# SkillFlow

A semantic skill retrieval engine that enables AI agents to discover relevant skills from a corpus of ~36K agent skills. Given a natural-language task query, SkillFlow searches a pre-built vector index and returns the best-matching skills.

## Quick Start

```bash
# Install dependencies
uv sync

# Build the FAISS index (one-time, ~90s on GPU)
uv run python -m skill_flow.cli build-index

# Search for skills
uv run python -m skill_flow.cli search --query "help me write unit tests for a FastAPI application"

# Search with cross-encoder reranking (Stage 2)
uv run python -m skill_flow.cli search --query "help me write unit tests for a FastAPI application" --rerank

# Run config-driven evaluation (retriever + reranker stages)
uv run python -m skill_flow.cli eval
```

## Architecture

SkillFlow uses a multi-stage retrieval pipeline:

| Stage | Method | Candidates | Status |
|-------|--------|------------|--------|
| 1 — Retrieval | Bi-encoder (`BAAI/bge-base-en-v1.5`) + FAISS | 36K → 1000 | Implemented |
| 2 — Reranking | Cross-encoder (`BAAI/bge-reranker-v2-m3`) + full SKILL.md content | 1000 → 100 | Implemented |
| 2a — Query Gen | LLM-based query generation (optional, before reranking) | — | Implemented |
| 3 — Selection | LLM | 100 → 1 | Planned |

### Stage 1: Vector Search

- Encodes skill descriptions using a BGE bi-encoder (768-dim, L2-normalized)
- Builds a FAISS `IndexFlatIP` index (inner product on normalized vectors = cosine similarity)
- Persists five artifacts to `outputs/indices/`: `embeddings.npy`, `faiss.index`, `skill_ids.json`, `skill_descriptions.json`, `skill_contents.json`
- At query time: encodes with BGE query prefix, searches FAISS, returns top-k results with full SKILL.md content

### Stage 2: Cross-Encoder Reranking

- Rescores Stage 1 candidates using a BGE cross-encoder (`BAAI/bge-reranker-v2-m3`)
- Uses full SKILL.md content (not just descriptions) for more accurate relevance scoring
- Processes in small batches (default batch_size=4) to fit VRAM constraints with long content
- Configurable via `models.reranker` in config or `--rerank` CLI flag

### Stage 2a: LLM Query Generation (Optional)

- Converts verbose task instructions (~2,700 chars avg) into concise search queries (< 200 chars) before cross-encoder scoring
- Addresses input mismatch: cross-encoders are trained on short queries, not multi-paragraph instructions
- Uses OpenAI API (`gpt-4o-mini` by default); requires `OPENAI_API_KEY` in `.env`
- Results cached to `outputs/query_gen_cache.json` (write-through per task, survives interrupted runs)
- Enable via `models.reranker.query_gen.enabled = true` in config
- System prompt is configurable for easy experimentation

### Corpus

Skills are sourced from `../skill-crawler/data/skills/` (~36K SKILL.md files). Each skill has structured metadata (name, description, author, stars) in `_metadata/index.json` and full content in individual SKILL.md files.

## Project Structure

```
skill-flow/
├── skill_flow/             # Core retrieval library
│   ├── cli.py              # CLI (build-index, search, eval)
│   ├── models/             # SkillRecord, SkillFlow facade
│   ├── config/             # Pydantic config + default.json
│   ├── corpus/             # Corpus loader (metadata + full SKILL.md content)
│   ├── index/              # FAISS encoder, builder, searcher (Stage 1)
│   ├── rerank/             # Cross-encoder reranker + LLM query gen (Stage 2)
│   └── eval/               # Retriever + reranker evaluation against SkillsBench GT
├── benchmark/              # Harbor-based agent evaluation framework
├── analysis/               # Trajectory failure analysis
├── outputs/indices/        # Persisted FAISS index (gitignored)
└── tests/                  # Test suite (80% coverage threshold)
```

## Configuration

All configurable values live in `skill_flow/config/default.json`, organized in a nested `system`/`index`/`models` hierarchy:

```json
{
  "system": {},
  "index": {
    "input_corpus_path": "../skill-crawler/data/skills/",
    "output_index_path": "outputs/indices/"
  },
  "models": {
    "retriever": {
      "model_name": "BAAI/bge-base-en-v1.5",
      "query_prompt": "Represent this sentence for searching relevant passages: ",
      "batch_size": 256,
      "top_k": 1000,
      "eval": { "enabled": false, "..." : "..." }
    },
    "reranker": {
      "model_name": "BAAI/bge-reranker-v2-m3",
      "top_k": 100,
      "batch_size": 4,
      "query_gen": { "enabled": false, "model": "gpt-4o-mini", "..." : "..." },
      "eval": { "enabled": true, "..." : "..." }
    }
  }
}
```

Override via CLI flag: `--config path/to/custom.json`

## Development

```bash
# Run tests
uv run pytest tests/test_skill_flow/ -v

# Lint
uv run ruff check skill_flow/

# Type check
uv run mypy skill_flow/
```

## Requirements

- Python 3.12+
- CUDA GPU recommended for index building (CPU works but slower)
- Corpus data in `../skill-crawler/data/skills/`

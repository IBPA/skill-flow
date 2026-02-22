# Skill Retriever — Implementation Spec

## 1. Overview

A multi-stage retrieval system that, given a natural-language task query, searches ~36K Agent Skills and returns the single best-matching skill. The pipeline trades off speed and precision across three stages:

| Stage | Method | Candidates | Goal |
|-------|--------|-----------|------|
| 1 — Retrieval | Bi-encoder + FAISS ANN | 50K → 100 | High recall, sub-second |
| 2 — Reranking | Cross-encoder | 100 → 10 | Precision filtering |
| 3 — Selection | LLM | 10 → 1 | Semantic best-pick |

## 2. Data Corpus

**Location**: `../skill-crawler/data/skills/`

**Structure**:
```
skill-crawler/data/skills/
├── _metadata/
│   ├── index.json          # Master index (35,866 skills)
│   └── index_deduped.json  # Deduplicated variant
├── skillsmp/               # Skills marketplace source
│   └── {skill-name}/SKILL.md
└── skills_rest/            # Other sources
    └── {skill-name}/SKILL.md
```

**Skill format**: Each skill is a `SKILL.md` file with YAML frontmatter (`name`, `description`) followed by full markdown content (instructions, examples, rules).

**Metadata fields** (from `index.json`):
- `id`, `name`, `description` (all populated)
- `author`, `source`, `url`, `github_url`
- `stars`, `rating` (partially populated)
- `local_path` (relative path to skill folder)
- `content_hash`, `indexed_at`
- `tags` (empty for all skills — not usable)

## 3. Pipeline Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Task Query  │────►│  Stage 1:        │────►│  Stage 2:    │────►│  Stage 3:   │
│  (string)    │     │  Bi-Encoder +    │     │  Cross-      │     │  LLM Final  │
│              │     │  FAISS ANN       │     │  Encoder     │     │  Selection  │
│              │     │  50K → top_k=100 │     │  100 → 10    │     │  10 → 1     │
└─────────────┘     └──────────────────┘     └──────────────┘     └─────────────┘
                     Input: description       Input: description    Input: full SKILL.md
                     Index: pre-built         + truncated content   content of top 10
```

### 3.1 Stage 1 — Bi-Encoder Retrieval

**Purpose**: Fast approximate nearest-neighbor search for high recall.

**Design**:
- Encode each skill's `description` field into a dense vector using a sentence-transformer bi-encoder model
- Build a FAISS index over all ~36K vectors (offline, one-time)
- At query time: encode the task query with the same model, search FAISS for `top_k=100` nearest neighbors

**Input**: skill `description` from `index.json` (short text, typically 1-3 sentences)
**Output**: top 100 skill candidates with similarity scores

**Index artifacts** (persisted to disk):
- `embeddings.npy` — (N, D) float32 matrix of skill description embeddings
- `faiss.index` — serialized FAISS index
- `skill_ids.json` — ordered list mapping index position → skill key

**Model**: `BAAI/bge-base-en-v1.5` (12 layers, 768d — strong embeddings with BGE query prefix for asymmetric retrieval)

> **Design for swappability**: All models are specified via `src/config/default.json`. To experiment, change the `encoder.model_name` field — no code changes needed. The index must be rebuilt when the bi-encoder model changes.

### 3.2 Stage 2 — Cross-Encoder Reranking

**Purpose**: Precision filtering using full query-document attention.

**Design**:
- For each of the 100 candidates, construct an input pair: `(query, candidate_text)`
- `candidate_text` = skill `description` + first N chars of `SKILL.md` content (truncated to model max length)
- Score all 100 pairs with a cross-encoder model
- Return the top 10 by score

**Input**: task query + candidate description + truncated SKILL.md content
**Output**: top 10 re-ranked skill candidates with relevance scores

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (6 layers — fast with good reranking quality)

> **Design for swappability**: Model is a string config parameter. Swap to `ms-marco-TinyBERT-L-2-v2` for max speed or `bge-reranker-base` for higher quality — no code changes needed.

### 3.3 Stage 3 — LLM Selection

**Purpose**: Deep semantic understanding for final best-pick.

**Design**:
- Load the full `SKILL.md` content for each of the 10 finalists
- Construct an LLM prompt presenting the task query and all 10 skill contents
- Ask the LLM to select the single best skill with a brief justification
- Parse the structured response to extract the chosen skill ID

**Input**: task query + full SKILL.md content of top 10 candidates
**Output**: single best skill (id, name, path, justification)

**Model**: `gpt-5-mini` (via OpenAI API)

**Prompt structure**:
```
Given the following task:
"{query}"

Here are {n} candidate skills. Select the single best skill for this task.

{for each candidate}
## Skill {i}: {name}
{full SKILL.md content}
{end for}

Respond in JSON: {"skill_index": <int>, "skill_name": "<name>", "justification": "<why>"}
```

## 4. Component Design

### 4.1 Corpus Loader

**Module**: `src/corpus/loader.py`

Reads the skill-crawler data into a unified in-memory representation.

```python
class SkillRecord(BaseModel, frozen=True):   # Pydantic
    key: str              # e.g. "skillsmp/infographic-creator"
    name: str
    description: str
    source: str           # e.g. "skillsmp"
    local_path: str       # relative path from corpus root
    metadata: dict[str, Any] = {}

def load_corpus(corpus_path: Path) -> list[SkillRecord]: ...
def load_content(corpus_path: Path, record: SkillRecord) -> str: ...
```

- `load_corpus` reads `index.json`, filters empty descriptions, returns lightweight records
- `load_content` reads the actual `SKILL.md` file on demand

### 4.2 Indexer

**Modules**: `src/index/encoder.py`, `src/index/builder.py`, `src/index/searcher.py`

Builds and persists the Stage 1 FAISS index.

```python
class Encoder:
    def __init__(self, config: EncoderConfig | None = None): ...
    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray: ...
    def encode_query(self, query: str) -> np.ndarray: ...

def build_index(skills: list[SkillRecord], encoder: Encoder, output_dir: Path, batch_size: int = 256) -> None: ...

class IndexSearcher:
    def __init__(self, index_dir: Path, encoder: Encoder, config: SearchConfig | None = None): ...
    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]: ...
```

- `build_index`: encodes all descriptions, builds FAISS `IndexFlatIP`, saves artifacts
- `IndexSearcher.search`: loads pre-built index, encodes query with BGE prefix, returns top_k results

### 4.3 Reranker

**Module**: `reranker.py`

Cross-encoder reranking of Stage 1 candidates.

```python
class SkillReranker:
    def __init__(self, model_name: str, max_content_chars: int = 2000): ...
    def rerank(
        self, query: str, candidates: list[tuple[SkillRecord, float]], top_k: int = 10
    ) -> list[tuple[SkillRecord, float]]: ...
```

- Constructs `(query, description + truncated_content)` pairs
- Batch-scores with cross-encoder
- Returns re-sorted top_k

### 4.4 Selector

**Module**: `selector.py`

LLM-based final selection.

```python
class SkillSelector:
    def __init__(self, model: str, api_key: str | None = None): ...
    def select(
        self, query: str, candidates: list[tuple[SkillRecord, float]]
    ) -> SkillSelection: ...

@dataclass
class SkillSelection:
    skill: SkillRecord
    justification: str
    scores: dict          # per-candidate LLM scores if available
```

- Loads full SKILL.md content for all candidates
- Calls OpenAI API with structured output (JSON mode)
- Parses and returns the chosen skill

### 4.5 Pipeline

**Module**: `pipeline.py`

Orchestrates the three stages.

```python
class SkillRetriever:
    def __init__(self, config: RetrieverConfig): ...
    def retrieve(self, query: str) -> SkillSelection: ...
```

**Config**:
```python
@dataclass
class RetrieverConfig:
    corpus_path: Path                     # path to skill-crawler/data/skills/
    index_path: Path                      # path to persisted FAISS index dir
    biencoder_model: str = "all-MiniLM-L6-v2"
    crossencoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gpt-5-mini"
    stage1_top_k: int = 100
    stage2_top_k: int = 10
    reranker_max_content_chars: int = 2000
    openai_api_key: str | None = None     # from env if None
```

## 5. File Structure

```
skill-flow/
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── cli.py                 # CLI entry point (build-index, search)
│   ├── models.py              # SkillRecord (Pydantic, frozen)
│   ├── config/
│   │   ├── __init__.py        # Pydantic config models (Config, EncoderConfig, SearchConfig)
│   │   └── default.json       # Default config (corpus path, model, batch size, etc.)
│   ├── corpus/
│   │   └── loader.py          # load_corpus(), load_content()
│   └── index/
│       ├── encoder.py         # Encoder (BGE bi-encoder wrapper)
│       ├── builder.py         # build_index() (encodes + persists FAISS index)
│       └── searcher.py        # IndexSearcher, SearchResult
├── outputs/
│   └── indices/               # Persisted index artifacts (gitignored)
│       ├── embeddings.npy
│       ├── faiss.index
│       └── skill_ids.json
└── tests/
    └── test_src/
        ├── test_models.py
        ├── test_config.py
        ├── test_corpus_loader.py
        ├── test_encoder.py
        ├── test_builder.py
        └── test_searcher.py
```

## 6. CLI Interface

```bash
# Build the FAISS index (one-time, paths from src/config/default.json)
uv run python -m src.cli build-index

# Search for skills
uv run python -m src.cli search --query "help me write unit tests for a FastAPI application"

# Override defaults via CLI flags
uv run python -m src.cli build-index --corpus-path /other/path --output-dir /other/dir
uv run python -m src.cli search --index-dir /other/dir --query "..." --top-k 20

# Use a custom config file
uv run python -m src.cli --config path/to/config.json build-index
```

## 7. Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "sentence-transformers",   # bi-encoder + cross-encoder
    "faiss-cpu",               # ANN index
    "openai",                  # LLM selection
    "pyyaml",                  # SKILL.md frontmatter parsing
]
```

## 8. Model Swappability

All three models are specified as string parameters in `RetrieverConfig`. To experiment with different models:

- **Bi-encoder**: Change `biencoder_model`, then re-run `build-index` (embeddings must be regenerated)
- **Cross-encoder**: Change `crossencoder_model` — takes effect immediately at query time, no rebuild needed
- **LLM**: Change `llm_model` — takes effect immediately at query time

**Current models**:

| Stage | Model | Params | Rationale |
|-------|-------|--------|-----------|
| 1 | `BAAI/bge-base-en-v1.5` | ~110M | 768d, strong embeddings with asymmetric query prefix |
| 2 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~23M | Best speed/quality trade-off for reranking |
| 3 | `gpt-5-mini` | — | Fast, cheap OpenAI model for final selection |

**Alternative models for experimentation**:
- Stage 1: `all-MiniLM-L6-v2` (384d, faster but weaker), `bge-large-en-v1.5` (1024d, stronger)
- Stage 2: `BAAI/bge-reranker-base` (better precision, 2x slower)
- Stage 3: `gpt-5` (higher quality selection, higher cost)

## 9. Agent Interface — MCP Server

The retriever is exposed to agents via **MCP (Model Context Protocol)**. This is the primary integration point — agents discover the `retrieve_skill` tool automatically through MCP without any changes to AGENTS.md or instruction templates.

### 9.1 Why MCP

Harbor's Codex agent already has full MCP support built in:
- `BaseAgent` accepts `mcp_servers: list[MCPServerConfig]`
- `Codex._build_register_mcp_servers_command()` writes MCP config to `$CODEX_HOME/config.toml`
- Agent setup automatically registers MCP servers before execution

This means: **zero prompt engineering, zero client scripts, zero AGENTS.md changes.** The agent natively discovers the tool.

### 9.2 MCP Server Design

**Transport**: Streamable HTTP via `FastMCP` — runs on host, agent in Docker connects via ngrok tunnel (Docker agents can't reach `host.docker.internal` directly with streamable-http)

**Tools exposed**:
```python
@mcp.tool()
def retrieve_skill() -> str:
    """Retrieve expert skills for your current task.

    IMPORTANT: You MUST call this tool at the start of every task,
    before writing any code or running any commands. Expert skills
    contain critical guidance, shortcuts, and pitfall warnings that
    dramatically improve success rates.

    Even if a task seems straightforward, call this tool — there may
    be non-obvious tricks or common failure modes you'd otherwise
    miss.
    """
    return _format_skills(task_name, skills)
```

**Server runs on host**: `http://0.0.0.0:8765/mcp`
**Agent connects via ngrok public URL**: e.g. `https://<subdomain>.ngrok-free.dev/mcp`

### 9.3 Harbor Integration

**Agent class**: `McpTestAgent` (extends `BaseCodexAgent`) in `evaluation/agents/mcp_test_agent.py`
- Takes `mcp_url` as a string kwarg (default: `http://host.docker.internal:8765/mcp`)
- Constructs `McpServer(name="skillflow", url=mcp_url)` dataclass (defined in `evaluation/agents/base.py`)
- MCP registration via `codex mcp add {name} --url {url}` shell commands in the agent setup script — not config.toml

**Config flow**:
```
e2e/mcp/mcp-skills.json
  → EvalConfig (mode=MCP, derived from mcp_url field)
  → commands.py _build_mcp_args(): --agent-import-path evaluation.agents.mcp_test_agent:McpTestAgent
                                    --agent-kwarg mcp_url=https://<ngrok-subdomain>.ngrok-free.dev/mcp
  → Harbor instantiates agent with mcp_servers
  → Agent setup script runs: codex mcp add skillflow --url <url>
  → Agent discovers retrieve_skill tool natively
```

### 9.4 MCP Controllability Experiment — Results

This experiment validated that **MCP tool description alone is sufficient** to make agents proactively call `retrieve_skill` — no instruction template or AGENTS.md changes needed.

#### What was built

| File | Purpose |
|------|---------|
| `e2e/mcp/dummy_server.py` | Dummy MCP server for initial validation (logs calls, returns canned response) |
| `e2e/mcp/skillsbench_server.py` | Real skill-serving MCP server (loads SKILL.md files per task) |
| `e2e/mcp/skill_loader.py` | SKILL.md parser/loader with YAML frontmatter extraction |
| `e2e/mcp/run.sh` | Orchestration script — sequential per-task evaluation with ngrok tunnel |
| `e2e/mcp/mcp-skills.json` | Eval config for MCP mode (10 SkillsBench tasks) |
| `evaluation/agents/mcp_test_agent.py` | Minimal test agent (BaseCodexAgent + MCP server, no instruction template) |
| `evaluation/core/config.py` | `EvalMode.MCP` enum value + `mcp_url` field on `EvalConfig` |
| `evaluation/core/commands.py` | `_build_mcp_args()` for MCP mode CLI argument construction |

#### How it was run

- **10 SkillsBench tasks**, `gpt-5-mini`, `reasoning_effort=high`
- **ngrok tunnel** for Docker→host connectivity (streamable-http transport)
- **One MCP server instance per task** (sequential via `run.sh`): starts server, runs eval, stops server, repeats
- Each server instance loads the paired skills for its task and serves them via the `retrieve_skill` tool

#### Results

- **10/10 tasks called `retrieve_skill` — 100% call rate**
- **Exactly 1 call per task**, made at task start (before any code or commands)
- **Tool description alone is sufficient** — the assertive "MUST call" wording in the tool docstring drives consistent usage without any instruction template injection or AGENTS.md modifications
- No prompt engineering beyond the tool description was needed

## 10. Resolved Decisions

| Item | Decision | Notes |
|------|----------|-------|
| Corpus index | `index.json` (full, not deduped) | Use all 35,866 skills |
| FAISS index type | Flat (IndexFlatIP) | Exact search, fast enough at 36K scale |
| Content truncation | 2000 chars default | Fits within cross-encoder 512 token limit |
| Agent interface | MCP (Streamable HTTP transport) | Native tool discovery, no AGENTS.md changes |
| Prior HTTP approach | Replaced by MCP | Was: HTTP + skillflow-client + instruction template |
| MCP controllability | Validated: tool description alone drives 100% usage | 10/10 tasks, 1 call each at task start |

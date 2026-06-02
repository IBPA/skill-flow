"""Late-fusion searcher over three per-section FAISS indices.

Each section index is a self-contained dense retriever built and queried by
its own encoder, so the encoder used for the ``code`` section may differ
from the one used for ``yaml``/``prose``. Per-section rankings are fused at
the score level — never across embedding spaces — via one of ``rrf``,
``max``, ``mean``, or ``sum_norm``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from skill_flow.config import RetrieverConfig, SectionConfig
from skill_flow.corpus.splitter import safe_section_text, split_skill_sections
from skill_flow.index.encoder import Encoder
from skill_flow.reranker.reranker import aggregate_scores
from skill_flow.retriever.retriever import IndexSearcher, SearchResult

logger = logging.getLogger(__name__)

_SECTIONS: tuple[str, str, str] = ("yaml", "prose", "code")
# Score assigned to a key that didn't appear in a section's top pool. Cosine
# similarity on L2-normalised vectors is in [-1, 1]; -1 is the floor and
# pushes the key to the bottom of that section's ranking when fusing.
_MISSING_SCORE = -1.0


def _section_config(config: RetrieverConfig, name: str) -> SectionConfig:
    if not config.sections or name not in config.sections:
        msg = f"section {name!r} missing from RetrieverConfig.sections"
        raise ValueError(msg)
    return config.sections[name]


def _to_retriever_config(sec: SectionConfig) -> RetrieverConfig:
    return RetrieverConfig(
        model_name=sec.model_name,
        query_prompt=sec.query_prompt,
        doc_prompt=sec.doc_prompt,
        batch_size=sec.batch_size,
        revision=sec.revision,
        max_seq_length=sec.max_seq_length,
    )


class SectionSearcher:
    """Searcher over (yaml, prose, code) sub-indices with late score fusion."""

    def __init__(
        self,
        sub_searchers: dict[str, IndexSearcher],
        descriptions: dict[str, str],
        contents: dict[str, str],
        aggregation: str = "rrf",
        pool_size: int = 1000,
    ) -> None:
        self._sub: dict[str, IndexSearcher] = sub_searchers
        self._descriptions = dict(descriptions)
        self._contents = dict(contents)
        self._aggregation = aggregation
        self._pool_size = pool_size

    @classmethod
    def from_config(cls, config: RetrieverConfig) -> SectionSearcher:
        """Construct from a ``RetrieverConfig`` with ``retriever_type='section'``."""
        sub: dict[str, IndexSearcher] = {}
        descriptions: dict[str, str] = {}
        contents: dict[str, str] = {}
        parent_dir: Path | None = None
        for name in _SECTIONS:
            sec = _section_config(config, name)
            rc = _to_retriever_config(sec)
            encoder = Encoder(rc)
            sub_dir = Path(sec.index_dir)
            sub[name] = IndexSearcher(sub_dir, encoder, rc)
            if parent_dir is None:
                parent_dir = sub_dir.parent
        if parent_dir is not None:
            descriptions = _maybe_load_json(parent_dir / "skill_descriptions.json")
            contents = _maybe_load_json(parent_dir / "skill_contents.json")
        return cls(
            sub_searchers=sub,
            descriptions=descriptions,
            contents=contents,
            aggregation=config.section_aggregation,
            pool_size=config.section_pool_size,
        )

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search each sub-index and fuse the rankings."""
        per_section: dict[str, dict[str, float]] = {}
        for name, searcher in self._sub.items():
            results = searcher.search(query, top_k=self._pool_size)
            per_section[name] = {r.key: r.score for r in results}

        fused = _fuse(per_section, self._aggregation)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        k = top_k if top_k is not None else len(ordered)
        ordered = ordered[:k]
        return [
            SearchResult(
                key=key,
                score=score,
                description=self._descriptions.get(key, ""),
                content=self._contents.get(key, ""),
            )
            for key, score in ordered
        ]

    def augment(self, keys: list[str], descriptions: list[str]) -> None:
        """No-op: actual augmentation happens in :meth:`add_contents`.

        The eval runner calls ``augment`` with descriptions only, but section
        retrieval needs the full SKILL.md content (to split into 3 sections).
        We defer the work to ``add_contents``, which receives the full text
        immediately afterwards in :func:`skill_flow.eval.runner._augment_searcher`.
        """

    def add_descriptions(self, descriptions: dict[str, str]) -> None:
        self._descriptions.update(descriptions)

    def add_contents(self, contents: dict[str, str]) -> None:
        self._contents.update(contents)
        for key, content in contents.items():
            yaml, prose, code = split_skill_sections(content)
            section_text = {"yaml": yaml, "prose": prose, "code": code}
            for name, searcher in self._sub.items():
                searcher.augment([key], [safe_section_text(section_text[name])])


def _maybe_load_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _fuse(
    per_section: dict[str, dict[str, float]],
    aggregation: str,
) -> dict[str, float]:
    """Combine per-section score dicts into one score per key.

    Builds aligned per-key score vectors (one entry per section) with missing
    entries filled by ``_MISSING_SCORE``, then delegates to
    :func:`skill_flow.reranker.reranker.aggregate_scores` for ``max``,
    ``mean``, ``rrf``. ``sum_norm`` (min-max normalise per section, sum
    across) is computed locally.
    """
    section_names = list(per_section.keys())
    all_keys: set[str] = set()
    for d in per_section.values():
        all_keys.update(d)

    if aggregation == "sum_norm":
        return _sum_norm(per_section, all_keys, section_names)

    per_key_scores: dict[str, list[float]] = {
        key: [per_section[name].get(key, _MISSING_SCORE) for name in section_names]
        for key in all_keys
    }
    return aggregate_scores(per_key_scores, aggregation)


def _sum_norm(
    per_section: dict[str, dict[str, float]],
    all_keys: set[str],
    section_names: list[str],
) -> dict[str, float]:
    """Min-max normalise each section's scores to ``[0, 1]``, then sum."""
    normalised: dict[str, dict[str, float]] = {}
    for name in section_names:
        scores = per_section[name]
        if not scores:
            normalised[name] = {}
            continue
        lo = min(scores.values())
        hi = max(scores.values())
        span = hi - lo
        if span <= 0:
            normalised[name] = dict.fromkeys(scores, 0.0)
        else:
            normalised[name] = {k: (v - lo) / span for k, v in scores.items()}
    fused: dict[str, float] = {}
    for key in all_keys:
        fused[key] = sum(normalised[name].get(key, 0.0) for name in section_names)
    return fused

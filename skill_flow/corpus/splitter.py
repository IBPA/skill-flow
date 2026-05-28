"""Split a SKILL.md into (yaml, prose, code) sections for structure-aware retrieval.

The three sections are encoded independently and their rankings are late-fused
by :class:`skill_flow.retriever.section_searcher.SectionSearcher`. Empty
sections are returned as the empty string; callers decide on a sentinel
(typically encoding the empty string, which yields a deterministic vector
under sentence-transformers).
"""

from __future__ import annotations

import re

# Frontmatter is fenced by ``---`` lines per Anthropic SKILL.md convention.
# DOTALL so ``.`` matches the YAML body's newlines; non-greedy so the second
# fence stops at the first occurrence.
# Sentinel returned by :func:`safe_section_text` for empty sections. Causal-LM
# encoders (e.g. bge-code-v1, Qwen2 backbone) crash with ``cannot reshape
# tensor of 0 elements`` when the tokenizer produces a zero-length sequence
# for an empty string, so callers must substitute this before encoding.
EMPTY_SECTION_SENTINEL = "[empty]"


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)

# Fenced code blocks: opening ``` plus an optional language tag, then body,
# then closing ```. Language tag is consumed but discarded.
_CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9+_.-]*\n(.*?)```", re.DOTALL)


def split_skill_sections(md: str) -> tuple[str, str, str]:
    """Return ``(yaml, prose, code)`` for a SKILL.md string.

    * ``yaml`` — raw YAML frontmatter body (between the ``---`` fences), or
      ``""`` if absent or malformed.
    * ``prose`` — markdown body with fenced code blocks removed, whitespace
      collapsed.
    * ``code`` — concatenation of every fenced code block body in document
      order, separated by blank lines. Fence markers and language tags are
      dropped.
    """
    m = _FRONTMATTER_RE.match(md)
    if m:
        yaml = m.group(1).strip()
        body = m.group(2)
    else:
        yaml = ""
        body = md

    code_blocks = _CODE_BLOCK_RE.findall(body)
    code = "\n\n".join(b.strip() for b in code_blocks if b.strip())

    prose = _CODE_BLOCK_RE.sub("", body)
    prose = re.sub(r"\n{3,}", "\n\n", prose).strip()

    return yaml, prose, code


def safe_section_text(text: str) -> str:
    """Return *text* or :data:`EMPTY_SECTION_SENTINEL` if empty after strip."""
    return text if text.strip() else EMPTY_SECTION_SENTINEL

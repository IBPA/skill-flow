"""LLM judge for the corpus functionality audit (gpt-4o-mini by default).

Given a skill's SKILL.md plus the list of files it bundles, the judge rates
three axes in context: code-sound, no-missing-files, and purpose-aligned.
Judging in context (with the bundle listing) is what lets the model tell a
genuinely missing bundled file from a reference to the user's own project,
which a regex over the corpus cannot do. Results are cached by skill name so
re-runs do not re-call the API.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from openai import OpenAI

from analysis.results.utils.functional_utils import JudgeVerdict

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_CHARS = 8000  # SKILL.md truncation to bound per-call cost

_SYSTEM = """You audit agent skills. A skill is a SKILL.md file, sometimes \
shipping bundled scripts/references. Given the SKILL.md and the list of files \
it bundles, judge whether it is FUNCTIONAL. Definitions:
- code_sound: the runnable code the skill provides (bundled scripts and/or \
fenced code blocks) is complete and executable, not a stub, placeholder, \
pseudocode, or truncated fragment. If the skill provides no code at all (pure \
prose, or only illustrative snippets), set code_sound=false.
- missing_files: set true ONLY if the SKILL.md explicitly tells the agent to \
run or read a SPECIFIC file in the skill's OWN bundle (a scripts/, references/, \
or assets/ path, or a {baseDir}/ path) that is ABSENT from the Bundled files \
list. If the skill bundles nothing and names no specific bundled file, OR all \
named bundled files are present, set missing_files=false. References to the \
USER'S project files (paths expected in the user's codebase, not shipped by \
the skill) NEVER count as missing.
- purpose_aligned: the provided code/instructions plausibly accomplish what \
the description claims.
- tier: "functional" = ships complete runnable code (a bundled script or a \
complete fenced code block / command sequence) that matches its purpose, with \
no missing bundled files; "reference_only" = a prose / instructional / \
reference document not intended to ship runnable code (it guides the agent in \
natural language or shows only illustrative snippets) -- most skills with no \
bundled scripts and only descriptive prose are reference_only; "partial" = it \
clearly intends to ship runnable code but that code is incomplete, stubbed, or \
misaligned, OR it references missing bundled files.
Return ONLY a JSON object with keys code_sound (bool), missing_files (bool), \
purpose_aligned (bool), tier (string), reason (one short sentence)."""


def build_user_prompt(skill_dir: Path) -> str:
    """Render the user message: bundle listing + (truncated) SKILL.md."""
    files = sorted(
        str(f.relative_to(skill_dir))
        for f in skill_dir.rglob("*")
        if f.is_file() and f.name != "SKILL.md"
    )
    listing = "\n".join(files) if files else "(none)"
    text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="ignore")
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n...[truncated]"
    return (
        f"Skill: {skill_dir.name}\n\n"
        f"Bundled files (besides SKILL.md):\n{listing}\n\n"
        f"SKILL.md:\n{text}"
    )


def _parse(name: str, content: str) -> JudgeVerdict | None:
    """Parse a JSON verdict; None on malformed output."""
    try:
        d = json.loads(content)
        return JudgeVerdict(
            name=name,
            code_sound=bool(d["code_sound"]),
            missing_files=bool(d["missing_files"]),
            purpose_aligned=bool(d["purpose_aligned"]),
            tier=str(d["tier"]),
            reason=str(d.get("reason", ""))[:300],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("judge parse failed for %s: %s", name, exc)
        return None


def _load_cache(path: Path) -> dict[str, dict[str, object]]:
    if path.exists():
        data: dict[str, dict[str, object]] = json.loads(
            path.read_text(encoding="utf-8")
        )
        return data
    return {}


def judge_sample(
    skill_dirs: list[Path],
    *,
    model: str = "gpt-4o-mini",
    cache_path: Path,
) -> list[JudgeVerdict]:
    """Judge each skill dir, caching verdicts by name. Returns parsed verdicts."""
    load_dotenv()
    client = OpenAI()
    cache = _load_cache(cache_path)
    verdicts: list[JudgeVerdict] = []
    for i, d in enumerate(skill_dirs):
        if d.name in cache:
            v = _parse(d.name, json.dumps(cache[d.name]))
            if v is not None:
                verdicts.append(v)
                continue
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": build_user_prompt(d)},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or "{}"
        v = _parse(d.name, content)
        if v is not None:
            verdicts.append(v)
            cache[d.name] = v.model_dump(exclude={"name"})
        if (i + 1) % 25 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
            logger.info("judged %d/%d", i + 1, len(skill_dirs))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return verdicts

"""Corpus functionality audit (community skills): distribution + sampled judge.

Reports the full-corpus structural distribution (objective) and a sampled
LLM-judged functional fraction (code-bearing AND code-sound AND
no-missing-files, judged in context). Runtime correctness is out of scope.

Writes ``paper/tables/19_functional_audit.tex`` plus a JSON record. The judge
step calls the OpenAI API (needs OPENAI_API_KEY and network); pass
``--no-judge`` to compute only the free structural distribution.

Usage::

    uv run python -m analysis.results.t19_generate_functional_audit
    uv run python -m analysis.results.t19_generate_functional_audit --no-judge
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from analysis.results.utils.functional_judge import judge_sample
from analysis.results.utils.functional_utils import (
    AuditSummary,
    CorpusDistribution,
    SkillStructural,
    audit_summary,
    scan_corpus,
    stratified_sample,
    structural_distribution,
)
from analysis.results.utils.latex_utils import table_env, write_or_print

logger = logging.getLogger(__name__)


def _load_or_scan(corpus_dir: Path, cache: Path) -> list[SkillStructural]:
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return [SkillStructural.model_validate(r) for r in raw]
    rows = scan_corpus(corpus_dir)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([r.model_dump() for r in rows]), encoding="utf-8")
    return rows


def render_latex(dist: CorpusDistribution, summary: AuditSummary) -> list[str]:
    """Metric/value table body (tabular only; wrapper lives in main.tex)."""
    top, bot = table_env("lr", r"\textbf{Metric} & \textbf{Value}")

    def pct(x: float, d: int = 1) -> str:
        return f"{x * 100:.{d}f}\\%"  # escape % for LaTeX

    body = [
        f"  Code-bearing (corpus, n={dist.n}) & {pct(dist.code_bearing)} \\\\",
        f"  Bundles executable scripts & {pct(dist.has_scripts)} \\\\",
        f"  Has fenced code block & {pct(dist.has_code_block)} \\\\",
        f"  Has any bundled file & {pct(dist.has_bundled_file)} \\\\",
        r"  \midrule",
    ]
    if summary.n_judged:
        lo, hi = summary.functional_ci
        body += [
            f"  Judged functional (sample, n={summary.n_judged}) & "
            f"{pct(summary.functional_fraction)} "
            f"{{\\scriptsize~[{pct(lo, 0)}, {pct(hi, 0)}]}} \\\\",
            f"  \\quad tier: functional & {summary.tier_functional} \\\\",
            f"  \\quad tier: partial & {summary.tier_partial} \\\\",
            f"  \\quad tier: reference-only & {summary.tier_reference_only} \\\\",
        ]
    return [*top, *body, *bot]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, default=Path("data/skills/skillsmp"))
    ap.add_argument("--sample-size", type=int, default=300)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument(
        "--structural-cache",
        type=Path,
        default=Path("outputs/analysis/corpus_structural.json"),
    )
    ap.add_argument(
        "--judge-cache",
        type=Path,
        default=Path("outputs/analysis/functional_judge_cache.json"),
    )
    ap.add_argument(
        "--latex-out", type=Path, default=Path("paper/tables/19_functional_audit.tex")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("outputs/analysis/functional_audit.json")
    )
    args = ap.parse_args()

    rows = _load_or_scan(args.corpus_dir, args.structural_cache)
    dist = structural_distribution(rows)
    logger.info("corpus: n=%d  code-bearing=%.1f%%", dist.n, dist.code_bearing * 100)

    summary = audit_summary({}, [])
    verdict_dump: list[dict[str, object]] = []
    if not args.no_judge:
        sample = stratified_sample(rows, args.sample_size)
        dirs = [args.corpus_dir / r.name for r in sample]
        verdicts = judge_sample(dirs, model=args.model, cache_path=args.judge_cache)
        structural_map = {r.name: r for r in sample}
        summary = audit_summary(structural_map, verdicts)
        verdict_dump = [v.model_dump() for v in verdicts]

    print("## Full-corpus structural distribution")
    print(f"  code_bearing: {dist.code_bearing:.1%}")
    print(f"  has_scripts: {dist.has_scripts:.1%}")
    print(f"  has_code_block: {dist.has_code_block:.1%}")
    print(f"  has_bundled_file: {dist.has_bundled_file:.1%}")
    if summary.n_judged:
        lo, hi = summary.functional_ci
        print(f"\n## Judged sample (n={summary.n_judged})")
        print(
            f"  functional fraction: {summary.functional_fraction:.1%} "
            f"[{lo:.1%}, {hi:.1%}]"
        )
        print(
            f"  tiers: functional={summary.tier_functional} "
            f"partial={summary.tier_partial} "
            f"reference_only={summary.tier_reference_only}"
        )

    write_or_print(render_latex(dist, summary), args.latex_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "distribution": dist.model_dump(),
                "audit": summary.model_dump(),
                "verdicts": verdict_dump,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate case study pipeline progression tables (T13--T15).

Each case study table is written to paper/tables/ as a standalone .tex file
(13_case_study_1.tex, 14_case_study_2.tex, 15_case_study_3.tex).
The narrative text lives directly in main.tex.

Usage::

    uv run python -m analysis.results.t13_14_15_generate_case_studies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from analysis.results.utils.latex_utils import write_or_print

_DEFAULT_PIPELINE_DIR = Path(
    "outputs/pipeline/skillsbench/specificity-v3.0",
)
_DEFAULT_TASK_IDS = [
    "invoice-fraud-detection",
    "court-form-filling",
    "gravitational-wave-detection",
]

_STAGE_FILES = {
    "retriever": "pipeline-stage1-retriever.json",
    "reranker": "pipeline-stage2-reranker.json",
    "deep_reranker": "pipeline-stage3-deep_reranker.json",
    "selector": "pipeline-stage4-selector.json",
}

_STAGE_LABELS = {
    "retriever": "Stage 1: Dense Retriever",
    "reranker": "Stage 2: Cross-Encoder",
    "deep_reranker": "Stage 3: Deep Reranker",
    "selector": "Stage 4: LLM Selector",
}

TOP_K = 5


def load_pipeline_stages(
    pipeline_dir: Path,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Load all pipeline stage results, keyed by stage then task_id."""
    stages: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for stage, filename in _STAGE_FILES.items():
        path = pipeline_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        task_map: dict[str, list[dict[str, Any]]] = {}
        for tr in data["task_results"]:
            task_map[tr["task_id"]] = tr["skills"][:TOP_K]
        stages[stage] = task_map
    return stages


def load_descriptions(path: Path) -> dict[str, str]:
    """Load skill descriptions keyed by skill ID."""
    if not path.exists():
        return {}
    result: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _escape_latex(text: str) -> str:
    for old, new in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
    ]:
        text = text.replace(old, new)
    return text


def _skill_short_name(key: str) -> str:
    parts = key.split("/")
    return parts[-1] if len(parts) > 1 else key


def _is_gt_skill(key: str, task_id: str) -> bool:
    return key.startswith(f"skillsbench/{task_id}/")


def _format_score(score: Any) -> str:
    if isinstance(score, dict):
        if score.get("relevancy", 0) == 1 and score.get("specificity", 0) == 1:
            return "relevant"
        return "filtered"
    if isinstance(score, float):
        return f"{score:.3f}"
    return str(score)


def _build_table_lines(
    task_id: str,
    stages: dict[str, dict[str, list[dict[str, Any]]]],
    _case_num: int,
) -> list[str]:
    """Build tabular content for a case study (no table wrapper)."""
    lines = [
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrl}",
        r"  \toprule",
        r"  \textbf{Stage} & \textbf{Skill} & \textbf{Score} & \textbf{GT?} \\",
        r"  \midrule",
    ]
    stage_keys = ["retriever", "reranker", "deep_reranker", "selector"]
    for si, stage_key in enumerate(stage_keys):
        task_skills = stages.get(stage_key, {}).get(task_id, [])
        label = _STAGE_LABELS[stage_key]
        first = True
        for skill in task_skills[:TOP_K]:
            key = skill["key"]
            score = _format_score(skill["score"])
            gt = "\\checkmark" if _is_gt_skill(key, task_id) else ""
            short = _escape_latex(_skill_short_name(key))
            stage_col = _escape_latex(label) if first else ""
            lines.append(f"  {stage_col} & \\texttt{{{short}}} & {score} & {gt} \\\\")
            first = False
        if si < len(stage_keys) - 1:
            lines.append(r"  \midrule")
    lines.extend([r"  \bottomrule", r"\end{tabular}%", r"}"])
    return lines


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Generate case study tables")
    ap.add_argument("--pipeline-dir", type=Path, default=_DEFAULT_PIPELINE_DIR)
    ap.add_argument("--task-ids", nargs="+", default=_DEFAULT_TASK_IDS)
    ap.add_argument("--output-dir", type=Path, default=Path("paper/tables"))
    args = ap.parse_args()

    stages = load_pipeline_stages(args.pipeline_dir)

    for i, task_id in enumerate(args.task_ids, 1):
        table_num = 12 + i
        table_lines = _build_table_lines(task_id, stages, i)
        out = args.output_dir / f"{table_num}_case_study_{i}.tex"
        write_or_print(table_lines, out)

    return 0


if __name__ == "__main__":
    sys.exit(main())

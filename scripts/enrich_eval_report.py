"""Enrich a slim evaluation report with full SKILL.md content.

Usage:
    uv run python scripts/enrich_eval_report.py REPORT_PATH
    uv run python scripts/enrich_eval_report.py REPORT_PATH -o enriched.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from skill_flow.eval.ground_truth import load_content_map, load_ground_truth
from skill_flow.eval.models import EvalReport, RetrievedSkill

logger = logging.getLogger(__name__)


def enrich_report(
    report: EvalReport,
    content_map: dict[str, str],
) -> EvalReport:
    """Return a copy of the report with skill content filled in."""
    enriched_results = [
        task_result.model_copy(
            update={
                "retrieved_skills": [
                    RetrievedSkill(
                        key=s.key,
                        score=s.score,
                        description=s.description,
                        content=content_map.get(s.key, s.content),
                    )
                    for s in task_result.retrieved_skills
                ],
            },
        )
        for task_result in report.task_results
    ]
    return report.model_copy(update={"task_results": enriched_results})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enrich eval report with full SKILL.md content",
    )
    parser.add_argument("report", type=Path, help="Slim eval report JSON")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path (default: <report>-enriched.json)",
    )
    args = parser.parse_args()

    if not args.report.exists():
        logger.error("Report not found: %s", args.report)
        return 1

    report = EvalReport.model_validate_json(
        args.report.read_text(encoding="utf-8"),
    )

    index_dir: Path = report.config.index_dir  # type: ignore[union-attr]
    tasks_dir: Path = report.config.tasks_dir  # type: ignore[union-attr]

    _, injected_skills, _ = load_ground_truth(tasks_dir)
    content_map = load_content_map(index_dir, injected_skills)
    enriched = enrich_report(report, content_map)

    output = args.output or args.report.with_stem(
        args.report.stem + "-enriched",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(enriched.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    logger.info("Enriched report written to %s", output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

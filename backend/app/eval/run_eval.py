"""Evaluate retrieval quality with the reusable ``relevance_eval`` skill.

This runner wires the repo's hybrid ``RetrievalService`` into the shared skill
(installed via the ``[eval]`` extra) and writes a Precision@k / MRR@k / nDCG@k
report. The retrieval logic and the metrics both live elsewhere — here we only
load judgments + thresholds, inject the search adapter, and gate.

Usage (from ``backend/``)::

    python -m app.eval.run_eval                       # uses committed example files
    python -m app.eval.run_eval --judgments j.json --thresholds t.json

Requires a live retrieval stack (Postgres + Qdrant + TEI), so this is an
integration-time command — run it locally against a populated index. The unit
tests exercise the same code paths offline with a fake service.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relevance_eval import (
    evaluate_thresholds,
    load_thresholds,
    run_evaluation,
    to_json,
    to_markdown,
)

from backend.app.dependencies import build_retrieval_service
from backend.app.eval.skill_adapter import STRATEGIES, make_search_fn

HERE = Path(__file__).resolve().parent
DEFAULT_JUDGMENTS = HERE / "judgments.example.json"
DEFAULT_THRESHOLDS = HERE / "thresholds.example.json"
REPO_ROOT = HERE.parents[3]
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"

KS = (1, 3, 5, 10)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
    thresholds = load_thresholds(str(args.thresholds))

    service = build_retrieval_service()
    search_fn = make_search_fn(service, limit=args.limit)

    report = run_evaluation(judgments, search_fn, list(STRATEGIES), ks=KS)
    gate = evaluate_thresholds(report, thresholds)
    markdown = to_markdown(report, gate, title="Retrieval evaluation (hybrid)")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "retrieval_eval.json").write_text(to_json(report), encoding="utf-8")
    (args.report_dir / "retrieval_eval.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

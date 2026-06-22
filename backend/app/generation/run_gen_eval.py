"""Faithfulness GATE for the grounded-generation path.

Mirrors ``app/eval/run_eval.py`` exactly — the relevance gate — but gates
*generation* on faithfulness instead of *retrieval* on relevance:

    load fixture + thresholds -> run_eval -> evaluate_thresholds ->
    JSON + Markdown via to_json / to_markdown -> exit 0 (pass) / 1 (regression).

Fully offline: the deterministic provider drafts grounded answers over the
committed fixture's canned chunks, so this runs in CI with no live ES / Qdrant /
TEI / Ollama. The runner exits non-zero on regression, exactly like the
relevance gate — that is the headline: generation is gated on faithfulness the
same way search is gated on relevance.

Run (from ``backend/``)::

    python -m app.generation.run_gen_eval     # uses committed example files
    python -m app.generation.run_gen_eval --fixture f.json --thresholds t.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from relevance_eval import evaluate_thresholds, load_thresholds, to_json, to_markdown

from backend.app.generation.eval import DEFAULT_FIXTURE, run_eval

HERE = Path(__file__).resolve().parent
DEFAULT_THRESHOLDS = HERE / "thresholds.example.json"
REPO_ROOT = HERE.parents[3]
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    report = asyncio.run(run_eval(args.fixture))
    thresholds = load_thresholds(str(args.thresholds))
    gate = evaluate_thresholds(report, thresholds)
    markdown = to_markdown(report, gate, title="Generation faithfulness gate")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "generation_eval.json").write_text(to_json(report), encoding="utf-8")
    (args.report_dir / "generation_eval.md").write_text(markdown, encoding="utf-8")

    # Print defensively: the shared renderer emits ✅/❌, which a non-UTF-8
    # console (Windows cp1252) cannot encode. The report files (written above)
    # and the exit code are the contract; the stdout echo must not crash the gate.
    try:
        print(markdown)
    except UnicodeEncodeError:
        sys.stdout.reconfigure(errors="replace")
        print(markdown)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

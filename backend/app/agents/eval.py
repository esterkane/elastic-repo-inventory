"""Offline eval for the research-briefing supervisor.

Measures two numbers on a small FIXED, committed fixture (questions + canned
retrieved chunks), with no live ES / Qdrant / TEI / Ollama:

* **citation-accuracy** — fraction of briefing claims whose attached
  ``source_url`` is in the retained (actually-retrieved) set; and
* **answer-faithfulness** — fraction of claims token-supported by the retrieved
  chunk texts, via ``relevance_eval.faithfulness_score`` (the shared 5.2 skill).

Both come straight from the :class:`VerifierAgent`, which is the same gate the
supervisor uses, so the eval measures exactly what production enforces. The
fixture is replayed through a fake retrieval service, so the deterministic
provider produces a grounded briefing and the eval runs fully offline.

Run (from ``backend/``)::

    python -m app.agents.eval        # prints a report; writes reports/agents_eval.{json,md}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend.app.agents.models import SubQuestionEvidence
from backend.app.agents.providers import DeterministicProvider
from backend.app.agents.supervisor import SupervisorAgent
from backend.app.ingest.metadata import normalize_metadata
from backend.app.retrieval.service import RankedHit

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "fixtures" / "briefing_eval.json"
REPO_ROOT = HERE.parents[3]
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"


class _FixtureRetrievalService:
    """Async ``retrieve`` that returns the case's canned chunks for any query.

    The same chunks are returned for every sub-question of a case, so the
    deterministic provider grounds the briefing in exactly the retained set the
    eval scores against — no live backend needed.
    """

    def __init__(self, chunks: list[RankedHit]) -> None:
        self.chunks = chunks

    async def retrieve(self, query, limit=10, filters=None, boosts=None):
        return {"hits": self.chunks[:limit], "warnings": [], "degraded": False}


def _chunk_to_hit(chunk: dict[str, Any], rank: int) -> RankedHit:
    metadata = normalize_metadata(
        {
            "title": chunk.get("title"),
            "heading_path": chunk.get("heading_path"),
            "content_type": chunk.get("content_type"),
            "license_family": chunk.get("license_family"),
            "final_rank": rank,
        },
        source_url=str(chunk["source_url"]),
        repo=str(chunk.get("repo") or ""),
        path=str(chunk.get("path") or ""),
    )
    return RankedHit(
        id=str(chunk["id"]),
        score=1.0 - rank * 0.01,
        metadata=metadata,
        source_url=str(chunk["source_url"]),
        text=str(chunk.get("text") or ""),
    )


async def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    chunks = [_chunk_to_hit(chunk, rank) for rank, chunk in enumerate(case["chunks"], start=1)]
    service = _FixtureRetrievalService(chunks)
    supervisor = SupervisorAgent(service, provider=DeterministicProvider())
    verified = await supervisor.run(case["query"])
    verification = verified.verification
    return {
        "id": case["id"],
        "query": case["query"],
        "claims": len(verification.claims),
        "citation_accuracy": round(verification.citation_accuracy, 4),
        "faithfulness": round(verification.faithfulness, 4),
        "verified": verified.verified,
        "unsupported_claims": [c.text for c in verification.unsupported_claims],
    }


async def run_eval(fixture_path: Path) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    case_reports = [await evaluate_case(case) for case in fixture["cases"]]
    n = len(case_reports) or 1
    return {
        "cases": case_reports,
        "summary": {
            "case_count": len(case_reports),
            "mean_citation_accuracy": round(sum(c["citation_accuracy"] for c in case_reports) / n, 4),
            "mean_faithfulness": round(sum(c["faithfulness"] for c in case_reports) / n, 4),
            "all_verified": all(c["verified"] for c in case_reports),
        },
    }


def to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Research-briefing supervisor eval",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Mean citation-accuracy: {summary['mean_citation_accuracy']}",
        f"- Mean answer-faithfulness: {summary['mean_faithfulness']}",
        f"- All briefings verified: {summary['all_verified']}",
        "",
        "| Case | Claims | Citation-accuracy | Faithfulness | Verified |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for case in report["cases"]:
        lines.append(
            f"| {case['id']} | {case['claims']} | {case['citation_accuracy']} | "
            f"{case['faithfulness']} | {'yes' if case['verified'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args(argv)

    report = asyncio.run(run_eval(args.fixture))
    markdown = to_markdown(report)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "agents_eval.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.report_dir / "agents_eval.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    return 0 if report["summary"]["all_verified"] else 1


if __name__ == "__main__":
    sys.exit(main())

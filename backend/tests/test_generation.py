"""Offline tests for the grounded-generation path and its faithfulness gate.

No live Postgres / Qdrant / TEI / Ollama. A ``FakeRetrievalService`` (async
``retrieve`` returning canned ``RankedHit``s, the pattern used across
``test_agents.py`` / ``test_eval_skill_integration.py``) drives generation, and
the deterministic provider drafts grounded answers.

Covered:
* ``GENERATION_ENABLED`` is off by default (opt-in) and accepts the usual truthy
  values — proving the default deterministic ``/answer`` path is untouched.
* ``answer()`` returns the ``{text, citations[]}`` shape with a ``source_url`` on
  every citation, grounded only in retrieved chunks.
* The eval CATCHES an intentionally-unsupported answer (faithfulness < 1.0) and
  the GATE FAILS (exit non-zero) when thresholds require full faithfulness; a
  fully-grounded fixture PASSES (exit 0).
* Citation-accuracy / faithfulness / correctness numbers match hand-computed
  expectations on the committed fixtures.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from backend.app.agents.providers import DeterministicProvider
from backend.app.generation.answer import answer, generation_enabled
from backend.app.generation.eval import (
    DEFAULT_FIXTURE,
    build_report,
    correctness_f1,
    evaluate_case,
    run_eval,
)
from backend.app.generation import run_gen_eval
from backend.app.ingest.metadata import normalize_metadata
from backend.app.retrieval.service import RankedHit

UNSUPPORTED_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "generation"
    / "fixtures"
    / "generation_eval_unsupported.json"
)

GROUNDED_TEXT = (
    "ES|QL adds a new LOOKUP JOIN command that enriches rows from an index by "
    "joining on a matched field. The LOOKUP JOIN command improves query latency "
    "for enrichment workloads compared to the older enrich pipeline approach."
)
GROUNDED_URL = "https://github.com/elastic/docs-content/blob/main/release-notes/esql.md#lookup-join"


def _hit(chunk_id: str, source_url: str, text: str, *, rank: int = 1) -> RankedHit:
    metadata = normalize_metadata(
        {"title": "Doc", "heading_path": "Guide > X", "content_type": "release_note", "final_rank": rank},
        source_url=source_url,
        repo="elastic/docs-content",
        path="release-notes/x.md",
    )
    return RankedHit(id=chunk_id, score=1.0 - rank * 0.01, metadata=metadata, source_url=source_url, text=text)


class FakeRetrievalService:
    def __init__(self, hits, warnings=None, degraded=False) -> None:
        self.hits = hits
        self.warnings = warnings or []
        self.degraded = degraded

    async def retrieve(self, query, limit=10, filters=None, boosts=None):
        return {
            "hits": self.hits[:limit],
            "recommendation_categories": [],
            "warnings": self.warnings,
            "degraded": self.degraded,
        }


# --- GENERATION_ENABLED off-by-default (opt-in proof) ------------------------


def test_generation_disabled_by_default(monkeypatch):
    monkeypatch.delenv("GENERATION_ENABLED", raising=False)
    assert generation_enabled() is False


def test_generation_flag_truthy_values(monkeypatch):
    for value in ("true", "1", "yes", "on", "TRUE"):
        monkeypatch.setenv("GENERATION_ENABLED", value)
        assert generation_enabled() is True
    for value in ("false", "0", "no", "", "off"):
        monkeypatch.setenv("GENERATION_ENABLED", value)
        assert generation_enabled() is False


def test_default_answer_path_unchanged_when_generation_off(monkeypatch):
    """With GENERATION_ENABLED off, the deterministic /answer synthesis is the
    default and is byte-for-byte unchanged — the generation package does not
    touch it. We assert the existing synthesis still produces its grounded
    answer independently of any generation code path."""
    monkeypatch.delenv("GENERATION_ENABLED", raising=False)
    from backend.app.api.search import answer_evidence, synthesize_answer_model

    hits = [_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)]
    evidence = answer_evidence("What is new in ES|QL for join queries?", hits, limit=4)
    model = synthesize_answer_model("What is new in ES|QL for join queries?", evidence)
    assert model["direct_answer"]
    assert model["provenance"][0].source_url == GROUNDED_URL
    assert generation_enabled() is False


# --- answer() contract: {text, citations[]} with source_url per claim --------


def test_answer_returns_text_and_citations_with_source_url():
    service = FakeRetrievalService(hits=[_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)])
    result = asyncio.run(
        answer("What is new in ES|QL for join queries?", retrieval_service=service, provider=DeterministicProvider())
    )

    assert isinstance(result["text"], str) and result["text"]
    assert isinstance(result["citations"], list) and result["citations"]
    # Every citation carries a source_url, and it is one of the retrieved chunks.
    for citation in result["citations"]:
        assert citation["source_url"]
        assert "claim" in citation
        assert citation["source_url"] in result["retained_source_urls"]
    # Grounded => nothing dropped, perfect provenance.
    assert result["dropped_claims"] == []
    assert result["citation_accuracy"] == 1.0
    assert result["faithfulness"] == 1.0


def test_answer_grounded_only_in_retrieved_chunks():
    """A claim may only cite a retrieved source_url; the deterministic provider
    grounds every claim, so no citation references an un-retrieved URL."""
    service = FakeRetrievalService(hits=[_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)])
    result = asyncio.run(answer("ES|QL join", retrieval_service=service, provider=DeterministicProvider()))
    retained = set(result["retained_source_urls"])
    assert retained == {GROUNDED_URL}
    assert all(c["source_url"] in retained for c in result["citations"])


# --- The headline: eval CATCHES an unsupported answer; gate FAILS ------------


def test_eval_catches_unsupported_answer():
    """The injected answer makes a claim its sources do not support, so the eval
    scores faithfulness < 1.0 for that case and flags the offending claim."""
    case = asyncio.run(_load_case(UNSUPPORTED_FIXTURE))
    report = asyncio.run(evaluate_case(case))
    assert report["faithfulness"] < 1.0
    # It still CITES a retrieved source (citation-accuracy high) — the failure is
    # faithfulness, not citation: the answer is grounded-looking but unfaithful.
    assert report["citation_accuracy"] == 1.0
    assert report["unsupported_claims"]
    assert any("quantum reranker" in claim for claim in report["unsupported_claims"])


def test_gate_fails_on_unfaithful_fixture():
    """The runner exits non-zero when thresholds require full faithfulness and a
    case is unfaithful — generation is gated on faithfulness like search on
    relevance."""
    exit_code = run_gen_eval.main(["--fixture", str(UNSUPPORTED_FIXTURE)])
    assert exit_code == 1


def test_gate_passes_on_grounded_fixture():
    """A fully-grounded fixture passes the gate (exit 0)."""
    exit_code = run_gen_eval.main(["--fixture", str(DEFAULT_FIXTURE)])
    assert exit_code == 0


# --- Hand-computed metric numbers --------------------------------------------


def test_grounded_fixture_metrics_match_expectations():
    report = asyncio.run(run_eval(DEFAULT_FIXTURE))
    summary = report["summary"]
    # Both grounded cases: every claim cites a retained source AND is token
    # supported, so citation-accuracy and faithfulness are a perfect 1.0.
    assert summary["mean_citation_accuracy"] == 1.0
    assert summary["mean_faithfulness"] == 1.0
    for case in report["cases"]:
        assert case["citation_accuracy"] == 1.0
        assert case["faithfulness"] == 1.0
        # Correctness vs gold is high (token-overlap F1) but not required to be 1.0.
        assert case["correctness"] > 0.5


def test_unsupported_case_faithfulness_is_one_half():
    """Two sentences in the injected answer: one grounded, one hallucinated, so
    faithfulness is exactly 1/2."""
    case = asyncio.run(_load_case(UNSUPPORTED_FIXTURE))
    report = asyncio.run(evaluate_case(case))
    assert report["faithfulness"] == 0.5


def test_correctness_is_one_for_identical_answer():
    text = "Vector search improves recall for filtered kNN queries."
    assert correctness_f1(text, text) == 1.0


def test_build_report_shapes_relevance_schema():
    """The report slots into the relevance_eval schema so the shared gate +
    renderer work unchanged: one strategy, metrics carried at @1."""
    cases = [
        {"id": "a", "query": "q", "citation_accuracy": 1.0, "faithfulness": 1.0, "correctness": 0.8, "unsupported_claims": []},
        {"id": "b", "query": "q", "citation_accuracy": 1.0, "faithfulness": 0.5, "correctness": 0.6, "unsupported_claims": ["x"]},
    ]
    report = build_report(cases)
    assert report["strategies"]["generation"]["metrics"]["faithfulness"]["1"] == 0.75
    assert report["ks"] == [1]
    assert set(report["metrics"]) == {"citation_accuracy", "faithfulness", "correctness"}


async def _load_case(fixture_path: Path):
    import json

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    return fixture["cases"][0]

"""Offline tests for the research-briefing multi-agent supervisor.

No live Postgres / Qdrant / TEI / Ollama is required. A ``FakeRetrievalService``
(async ``retrieve`` returning canned ``RankedHit``s, mirroring the pattern in
``test_mcp_tools.py`` / ``test_eval_skill_integration.py``) drives the agents.

Covered:
* VerifierAgent catches a known-UNSUPPORTED claim and passes a grounded one.
* Planner decomposition is deterministic.
* Supervisor degrades gracefully (a degraded retrieval still yields a briefing
  carrying the warning).
* Feature-flag-off isolation (``AGENTS_ENABLED`` default false; existing
  ``/answer`` deterministic path unchanged).
* Provider swappability (a fake provider is injected and used).
* Eval numbers (citation-accuracy + faithfulness) match hand-computed values.
"""

from __future__ import annotations

import asyncio

from backend.app.agents.agents import VerifierAgent
from backend.app.agents.eval import run_eval
from backend.app.agents.models import (
    Briefing,
    BriefingClaim,
    SubQuestion,
    SubQuestionEvidence,
)
from backend.app.agents.providers import DeterministicProvider, build_provider
from backend.app.agents.supervisor import SupervisorAgent, agents_enabled
from backend.app.agents.eval import DEFAULT_FIXTURE
from backend.app.ingest.metadata import normalize_metadata
from backend.app.retrieval.service import RankedHit, RetrievalWarning


# --- fakes -------------------------------------------------------------------


def _hit(chunk_id: str, source_url: str, text: str, *, rank: int = 1) -> RankedHit:
    metadata = normalize_metadata(
        {"title": "Doc", "heading_path": "Guide > X", "content_type": "release_note", "final_rank": rank},
        source_url=source_url,
        repo="elastic/docs-content",
        path="release-notes/x.md",
    )
    return RankedHit(
        id=chunk_id,
        score=1.0 - rank * 0.01,
        metadata=metadata,
        source_url=source_url,
        text=text,
    )


class FakeRetrievalService:
    """Async ``retrieve`` returning canned hits, with optional warnings/degraded."""

    def __init__(self, hits, warnings=None, degraded=False) -> None:
        self.hits = hits
        self.warnings = warnings or []
        self.degraded = degraded
        self.calls: list[dict] = []

    async def retrieve(self, query, limit=10, filters=None, boosts=None):
        self.calls.append({"query": query, "limit": limit, "filters": filters})
        return {
            "hits": self.hits[:limit],
            "recommendation_categories": [],
            "warnings": self.warnings,
            "degraded": self.degraded,
        }


GROUNDED_TEXT = (
    "ES|QL adds a new LOOKUP JOIN command that enriches rows from an index by "
    "joining on a matched field. The LOOKUP JOIN command improves query latency "
    "for enrichment workloads compared to the older enrich pipeline approach."
)
GROUNDED_URL = "https://github.com/elastic/docs-content/blob/main/release-notes/esql.md#lookup-join"


# --- VerifierAgent: the safety mechanism -------------------------------------


def _evidence_for(hits) -> list[SubQuestionEvidence]:
    return [
        SubQuestionEvidence(
            sub_question=SubQuestion(text="q"),
            hits=hits,
            warnings=[],
            degraded=False,
        )
    ]


def test_verifier_catches_unsupported_claim():
    """A claim whose source_url is NOT in the retained set must be flagged."""
    hits = [_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)]
    evidence = _evidence_for(hits)

    grounded = BriefingClaim(
        text="ES|QL adds a new LOOKUP JOIN command that enriches rows from an index by joining on a matched field.",
        source_url=GROUNDED_URL,
        chunk_id="chunk-1",
        section="answer",
    )
    fabricated = BriefingClaim(
        text="Elasticsearch now ships a built-in quantum reranker that triples recall overnight.",
        source_url="https://example.test/not-retrieved",
        chunk_id="ghost",
        section="answer",
    )
    briefing = Briefing(
        query="q",
        answer="ans",
        what_new=None,
        why_it_matters=None,
        claims=[grounded, fabricated],
        sub_questions=[SubQuestion(text="q")],
        retained_source_urls=[GROUNDED_URL],
    )

    result = VerifierAgent().verify(briefing, evidence)

    assert result.passed is False
    assert len(result.unsupported_claims) == 1
    flagged = result.unsupported_claims[0]
    assert flagged.text == fabricated.text
    assert flagged.source_retained is False


def test_verifier_catches_retained_url_but_no_token_support():
    """Even with a retained source_url, a claim with no token support fails."""
    hits = [_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)]
    evidence = _evidence_for(hits)
    # Cites a real retained url, but the sentence shares no meaningful tokens.
    claim = BriefingClaim(
        text="Sharks predate dinosaurs by hundreds of millions of years entirely.",
        source_url=GROUNDED_URL,
        chunk_id="chunk-1",
        section="answer",
    )
    briefing = Briefing(
        query="q",
        answer="ans",
        what_new=None,
        why_it_matters=None,
        claims=[claim],
        sub_questions=[SubQuestion(text="q")],
        retained_source_urls=[GROUNDED_URL],
    )

    result = VerifierAgent().verify(briefing, evidence)
    assert result.passed is False
    assert result.unsupported_claims[0].source_retained is True
    assert result.unsupported_claims[0].supported is False


def test_verifier_passes_fully_grounded_briefing():
    hits = [_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)]
    evidence = _evidence_for(hits)
    claim = BriefingClaim(
        text="ES|QL adds a new LOOKUP JOIN command that enriches rows from an index by joining on a matched field.",
        source_url=GROUNDED_URL,
        chunk_id="chunk-1",
        section="answer",
    )
    briefing = Briefing(
        query="q",
        answer="ans",
        what_new=None,
        why_it_matters=None,
        claims=[claim],
        sub_questions=[SubQuestion(text="q")],
        retained_source_urls=[GROUNDED_URL],
    )
    result = VerifierAgent().verify(briefing, evidence)
    assert result.passed is True
    assert result.unsupported_claims == []
    assert result.citation_accuracy == 1.0
    assert result.faithfulness == 1.0


# --- Planner decomposition (deterministic) -----------------------------------


def test_planner_decomposition_is_deterministic():
    provider = DeterministicProvider()
    subs_a = provider.plan("What is new in ES|QL joins?")
    subs_b = provider.plan("What is new in ES|QL joins?")
    assert [s.text for s in subs_a] == [s.text for s in subs_b]
    # primary question first, then facets
    assert subs_a[0].text == "What is new in ES|QL joins?"
    assert len(subs_a) >= 2


# --- Supervisor degrade-gracefully -------------------------------------------


def test_supervisor_degrades_gracefully_and_still_briefs():
    warning = RetrievalWarning(
        code="semantic_unavailable",
        message="Semantic retrieval unavailable; showing lexical results when available.",
        stage="semantic_retrieval",
    )
    service = FakeRetrievalService(
        hits=[_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)],
        warnings=[warning],
        degraded=True,
    )
    supervisor = SupervisorAgent(service, provider=DeterministicProvider())
    verified = asyncio.run(supervisor.run("What is new in ES|QL for join queries?"))

    # A briefing is still produced...
    assert verified.briefing.answer
    # ...and it carries the degradation warning forward.
    assert verified.briefing.degraded is True
    assert any(w.code == "semantic_unavailable" for w in verified.briefing.warnings)
    # grounded => verified.
    assert verified.verified is True


def test_supervisor_marks_unverified_when_provider_fabricates():
    """If a (fake) provider emits an unsupported claim, the supervisor returns it
    marked verified=False rather than silently emitting it."""

    class FabricatingProvider:
        name = "fabricating"

        def plan(self, query):
            return [SubQuestion(text=query)]

        def write(self, query, evidence):
            hits = [h for ev in evidence for h in ev.hits]
            retained = [h.source_url for h in hits]
            return Briefing(
                query=query,
                answer="fabricated",
                what_new=None,
                why_it_matters=None,
                claims=[
                    BriefingClaim(
                        text="A totally invented capability that appears in no chunk whatsoever.",
                        source_url="https://example.test/hallucinated",
                        chunk_id="ghost",
                        section="answer",
                    )
                ],
                sub_questions=[SubQuestion(text=query)],
                retained_source_urls=retained,
            )

    service = FakeRetrievalService(hits=[_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)])
    supervisor = SupervisorAgent(service, provider=FabricatingProvider())
    verified = asyncio.run(supervisor.run("q"))
    assert verified.verified is False
    assert verified.verification.unsupported_claims


# --- Feature flag isolation --------------------------------------------------


def test_agents_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENTS_ENABLED", raising=False)
    assert agents_enabled() is False


def test_agents_flag_truthy_values(monkeypatch):
    for value in ("true", "1", "yes", "on", "TRUE"):
        monkeypatch.setenv("AGENTS_ENABLED", value)
        assert agents_enabled() is True
    for value in ("false", "0", "no", "", "off"):
        monkeypatch.setenv("AGENTS_ENABLED", value)
        assert agents_enabled() is False


def test_answer_route_unchanged_when_flag_off(monkeypatch):
    """With the flag off, the deterministic /answer synthesis is byte-for-byte
    unchanged — the agents package does not touch the existing route. We assert
    the existing synthesis still produces its grounded answer independently of
    any agent code path."""
    monkeypatch.delenv("AGENTS_ENABLED", raising=False)
    from backend.app.api.search import answer_evidence, synthesize_answer_model

    hits = [_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)]
    evidence = answer_evidence("What is new in ES|QL for join queries?", hits, limit=4)
    model = synthesize_answer_model("What is new in ES|QL for join queries?", evidence)
    assert model["direct_answer"]
    assert model["provenance"][0].source_url == GROUNDED_URL
    assert agents_enabled() is False


# --- Provider swappability ---------------------------------------------------


def test_provider_swappable_via_injection():
    """A fake provider injected into the supervisor is the one that runs."""
    calls: list[str] = []

    class SpyProvider:
        name = "spy"

        def plan(self, query):
            calls.append("plan")
            return [SubQuestion(text=query)]

        def write(self, query, evidence):
            calls.append("write")
            hits = [h for ev in evidence for h in ev.hits]
            return Briefing(
                query=query,
                answer="spy answer",
                what_new=None,
                why_it_matters=None,
                claims=[
                    BriefingClaim(
                        text=hits[0].text,
                        source_url=hits[0].source_url,
                        chunk_id=hits[0].id,
                        section="answer",
                    )
                ],
                sub_questions=[SubQuestion(text=query)],
                retained_source_urls=[h.source_url for h in hits],
            )

    service = FakeRetrievalService(hits=[_hit("chunk-1", GROUNDED_URL, GROUNDED_TEXT)])
    supervisor = SupervisorAgent(service, provider=SpyProvider())
    verified = asyncio.run(supervisor.run("q"))
    assert calls == ["plan", "write"]
    assert verified.briefing.answer == "spy answer"


def test_build_provider_defaults_deterministic(monkeypatch):
    monkeypatch.delenv("AGENTS_PROVIDER", raising=False)
    assert build_provider().name == "deterministic"
    monkeypatch.setenv("AGENTS_PROVIDER", "ollama")
    assert build_provider().name == "ollama"


# --- Eval numbers (hand-computed) --------------------------------------------


def test_eval_numbers_match_expectations():
    report = asyncio.run(run_eval(DEFAULT_FIXTURE))
    summary = report["summary"]
    # Grounded fixture => every claim cites a retained source and is token
    # supported, so both metrics are a perfect 1.0 and all briefings verify.
    assert summary["mean_citation_accuracy"] == 1.0
    assert summary["mean_faithfulness"] == 1.0
    assert summary["all_verified"] is True
    for case in report["cases"]:
        assert case["citation_accuracy"] == 1.0
        assert case["faithfulness"] == 1.0
        assert case["claims"] >= 1
        assert case["verified"] is True

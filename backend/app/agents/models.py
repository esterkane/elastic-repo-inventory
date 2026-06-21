"""Dataclasses shared across the agent pipeline.

These are plain frozen dataclasses (not Pydantic) so the agents stay light and
testable; the API layer maps them to Pydantic response models when a route
exposes them. Every evidence- or claim-bearing structure carries ``source_url``
so the provenance contract is enforceable end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.app.retrieval.service import RankedHit, RetrievalWarning


@dataclass(frozen=True)
class SubQuestion:
    """One decomposed facet of the research question."""

    text: str
    rationale: str = ""


@dataclass(frozen=True)
class SubQuestionEvidence:
    """Evidence retrieved (and optionally reranked) for a single sub-question.

    ``hits`` are the retained chunks — the *only* chunks a claim may cite. The
    retrieval warnings and ``degraded`` flag for this sub-question are carried
    forward so the supervisor can surface graceful degradation.
    """

    sub_question: SubQuestion
    hits: list[RankedHit]
    warnings: list[RetrievalWarning] = field(default_factory=list)
    degraded: bool = False
    reranked: bool = False


@dataclass(frozen=True)
class BriefingClaim:
    """A single claim in the briefing, tagged with the chunk it traces to.

    ``source_url`` is the canonical GitHub URL of the retrieved chunk the writer
    grounded this claim in. ``chunk_id`` is the retained chunk's deterministic
    id. ``section`` records which part of the briefing the claim belongs to
    (answer / what's new / why it matters).
    """

    text: str
    source_url: str
    chunk_id: str
    section: str = "answer"


@dataclass(frozen=True)
class Briefing:
    """The drafted research briefing, before or after verification."""

    query: str
    answer: str
    what_new: str | None
    why_it_matters: str | None
    claims: list[BriefingClaim]
    sub_questions: list[SubQuestion]
    retained_source_urls: list[str]
    warnings: list[RetrievalWarning] = field(default_factory=list)
    degraded: bool = False


@dataclass(frozen=True)
class ClaimVerification:
    """Per-claim verification outcome."""

    text: str
    source_url: str
    section: str
    supported: bool
    support_score: float
    source_retained: bool

    @property
    def traces(self) -> bool:
        """A claim *traces* iff it is token-supported AND its source is retained."""
        return self.supported and self.source_retained


@dataclass(frozen=True)
class VerificationResult:
    """Structured verifier output. ``passed`` is False if any claim fails."""

    passed: bool
    faithfulness: float
    citation_accuracy: float
    claims: list[ClaimVerification]
    unsupported_claims: list[ClaimVerification] = field(default_factory=list)


@dataclass(frozen=True)
class VerifiedBriefing:
    """A briefing plus its verification verdict. ``verified`` mirrors
    ``verification.passed`` and is the single field callers gate on."""

    briefing: Briefing
    verification: VerificationResult

    @property
    def verified(self) -> bool:
        return self.verification.passed

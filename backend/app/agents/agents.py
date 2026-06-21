"""The individual agents: planner, retrieval, rerank, writer, verifier.

None of these implement retrieval, fusion, or synthesis logic of their own —
they wrap the existing functions exposed as :mod:`backend.app.agents.tools` and
the swappable :mod:`backend.app.agents.providers`. The VerifierAgent is the
safety mechanism that enforces the provenance contract.
"""

from __future__ import annotations

import re

from relevance_eval import faithfulness_score

from backend.app.agents.models import (
    Briefing,
    ClaimVerification,
    SubQuestion,
    SubQuestionEvidence,
    VerificationResult,
)
from backend.app.agents.providers import AgentLlmProvider
from backend.app.agents.tools import (
    RetrievalToolResult,
    SupportsRerank,
    SupportsRetrieve,
    retrieve_tool,
    rerank_tool,
)

# Token-support threshold for a claim to count as grounded in a retained chunk.
SUPPORT_THRESHOLD = 0.6


class PlannerAgent:
    """Decompose a release-intelligence question into sub-questions.

    Deterministic by default (rules in the provider); an LLM provider can
    override the decomposition. No retrieval happens here.
    """

    def __init__(self, provider: AgentLlmProvider) -> None:
        self.provider = provider

    def plan(self, query: str) -> list[SubQuestion]:
        return self.provider.plan(query)


class RetrievalAgent:
    """Run hybrid retrieval for a sub-question via the existing service."""

    def __init__(self, service: SupportsRetrieve, *, limit: int = 8) -> None:
        self.service = service
        self.limit = limit

    async def retrieve(self, sub_question: SubQuestion, *, filters: dict | None = None) -> RetrievalToolResult:
        return await retrieve_tool(self.service, sub_question.text, limit=self.limit, filters=filters)


class RerankAgent:
    """Re-score retained hits with the existing TEI reranker (optional)."""

    def __init__(self, reranker: SupportsRerank | None) -> None:
        self.reranker = reranker

    async def rerank(self, sub_question: SubQuestion, result: RetrievalToolResult) -> RetrievalToolResult:
        return await rerank_tool(self.reranker, sub_question.text, result)


class WriterAgent:
    """Draft the briefing grounded only in retrieved chunks, via the provider."""

    def __init__(self, provider: AgentLlmProvider) -> None:
        self.provider = provider

    def write(self, query: str, evidence: list[SubQuestionEvidence]) -> Briefing:
        return self.provider.write(query, evidence)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class VerifierAgent:
    """THE safety mechanism: enforce that every claim traces to a retained chunk.

    For each claim we check two things:

    1. **Token support** — the claim is supported by the text of the retrieved
       chunks, scored with ``relevance_eval.faithfulness_score`` (the shared
       5.2 skill's ``DeterministicClaimExtractor`` + ``TokenOverlapScorer``).
    2. **Source retained** — the ``source_url`` the writer attached to the claim
       is in the retained set (a chunk that was actually retrieved).

    A claim *traces* only if BOTH hold. The briefing FAILS verification if any
    claim does not trace; the offending claims are returned so the supervisor
    can surface them instead of silently emitting an unsupported briefing.
    """

    def __init__(self, support_threshold: float = SUPPORT_THRESHOLD) -> None:
        self.support_threshold = support_threshold

    def verify(self, briefing: Briefing, evidence: list[SubQuestionEvidence]) -> VerificationResult:
        retained_urls = set(briefing.retained_source_urls)
        # source_url -> concatenated retained chunk text, for per-claim scoring.
        url_to_text: dict[str, str] = {}
        all_texts: list[str] = []
        for item in evidence:
            for hit in item.hits:
                url_to_text.setdefault(hit.source_url, "")
                url_to_text[hit.source_url] += " " + hit.text
                all_texts.append(hit.text)

        verifications: list[ClaimVerification] = []
        unsupported: list[ClaimVerification] = []
        supported_count = 0
        cited_ok = 0

        for claim in briefing.claims:
            source_retained = claim.source_url in retained_urls
            # Score the claim against its OWN cited chunk text when retained, so
            # citation accuracy and token support are aligned; if the cited url
            # is not retained, score against all retained text (still likely to
            # fail, which is the point).
            scoring_sources = [url_to_text[claim.source_url]] if source_retained else all_texts
            faith = faithfulness_score(
                claim.text,
                scoring_sources or [""],
                support_threshold=self.support_threshold,
            )
            # faithfulness_score returns score==1.0 with total==0 for claims with
            # too few tokens to extract; treat a no-claim sentence as supported
            # only if it is also retained (it still must cite a real source).
            support_score = _claim_support_score(faith)
            token_supported = support_score >= self.support_threshold
            supported = token_supported and source_retained

            verification = ClaimVerification(
                text=claim.text,
                source_url=claim.source_url,
                section=claim.section,
                supported=token_supported,
                support_score=support_score,
                source_retained=source_retained,
            )
            verifications.append(verification)
            if source_retained:
                cited_ok += 1
            if supported:
                supported_count += 1
            else:
                unsupported.append(verification)

        total = len(briefing.claims)
        faithfulness = 1.0 if total == 0 else supported_count / total
        citation_accuracy = 1.0 if total == 0 else cited_ok / total
        passed = total > 0 and not unsupported

        return VerificationResult(
            passed=passed,
            faithfulness=faithfulness,
            citation_accuracy=citation_accuracy,
            claims=verifications,
            unsupported_claims=unsupported,
        )


def _claim_support_score(faith: dict) -> float:
    """Best per-claim support score from a faithfulness_score result.

    ``faithfulness_score`` runs its own claim extraction over the sentence; we
    take the max support over the (usually one) extracted claim. If nothing was
    extractable (too few tokens), the support is 0.0 so a contentless sentence
    does not get a free pass on token support.
    """
    claims = faith.get("claims") or []
    if not claims:
        return 0.0
    return max(float(entry.get("support_score") or 0.0) for entry in claims)

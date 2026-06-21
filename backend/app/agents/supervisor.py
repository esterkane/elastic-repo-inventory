"""SupervisorAgent — orchestrates the multi-agent research-briefing pipeline.

    planner -> (retrieval + rerank per sub-question) -> writer -> verifier

Degrades gracefully: retrieval warnings / ``degraded`` are propagated, a degraded
stage does not crash the run, and if verification fails the briefing is returned
marked ``verified=False`` with the offending claims rather than silently emitted.

Feature-flagged by ``AGENTS_ENABLED`` (env, default ``false``). When off, the
deterministic ``/answer`` synthesis path stays the default and no agent route or
job is active — :func:`agents_enabled` is the single gate the API consults.
"""

from __future__ import annotations

import os

from backend.app.agents.agents import (
    PlannerAgent,
    RerankAgent,
    RetrievalAgent,
    VerifierAgent,
    WriterAgent,
)
from backend.app.agents.models import (
    SubQuestionEvidence,
    VerifiedBriefing,
)
from backend.app.agents.providers import AgentLlmProvider, DeterministicProvider
from backend.app.agents.tools import SupportsRerank, SupportsRetrieve


def agents_enabled() -> bool:
    """Read the ``AGENTS_ENABLED`` flag the ``os.getenv`` way (default false)."""
    return os.getenv("AGENTS_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


class SupervisorAgent:
    """Coordinate the agents over the existing hybrid retrieval core."""

    def __init__(
        self,
        service: SupportsRetrieve,
        *,
        provider: AgentLlmProvider | None = None,
        reranker: SupportsRerank | None = None,
        retrieval_limit: int = 8,
    ) -> None:
        self.provider = provider or DeterministicProvider()
        self.planner = PlannerAgent(self.provider)
        self.retrieval = RetrievalAgent(service, limit=retrieval_limit)
        self.rerank = RerankAgent(reranker)
        self.writer = WriterAgent(self.provider)
        self.verifier = VerifierAgent()

    async def run(self, query: str, *, filters: dict | None = None) -> VerifiedBriefing:
        sub_questions = self.planner.plan(query)

        evidence: list[SubQuestionEvidence] = []
        for sub_question in sub_questions:
            result = await self.retrieval.retrieve(sub_question, filters=filters)
            # Rerank if available; honours the reliability contract internally.
            result = await self.rerank.rerank(sub_question, result)
            evidence.append(
                SubQuestionEvidence(
                    sub_question=sub_question,
                    hits=result.hits,
                    warnings=result.warnings,
                    degraded=result.degraded,
                    reranked=result.reranked,
                )
            )

        briefing = self.writer.write(query, evidence)
        verification = self.verifier.verify(briefing, evidence)
        return VerifiedBriefing(briefing=briefing, verification=verification)

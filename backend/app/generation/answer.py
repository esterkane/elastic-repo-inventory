"""Grounded-generation answer path — opt-in, gated on faithfulness.

The headline framing of this package: **generation is gated on faithfulness the
same way search is gated on relevance.** :func:`answer` is an *optional* answer
path (``GENERATION_ENABLED``, default false). When the flag is off the app's
default answer path is the existing deterministic ``/answer`` synthesis,
byte-for-byte unchanged — nothing here runs.

When on, :func:`answer`:

1. **retrieves** evidence via the existing async ``RetrievalService`` (hybrid
   BM25 + vector, reliability contract built in);
2. **drafts** an answer grounded ONLY in the retrieved chunks via the swappable
   provider (``build_provider`` / the agents' :class:`WriterAgent`), which
   attaches a retained ``source_url`` to every claim; and
3. **verifies** with the agents' :class:`VerifierAgent` — the SAME gate the
   research-briefing supervisor uses and the SAME measurement the generation
   eval scores — keeping only claims that *trace* (token-supported by their
   cited chunk AND that chunk was actually retrieved) and dropping the rest.

So we never duplicate the provider, writer, or verifier: the generation path is
the existing grounded-briefing machinery, reshaped into the ``{text,
citations[]}`` contract with a ``source_url`` on every citation.

The default provider is :class:`DeterministicProvider` (offline; the test/CI
default). A real run uses the wired local Ollama provider
(``AGENTS_PROVIDER=ollama``, ``OLLAMA_URL`` / ``OLLAMA_MODEL``) — documented,
never required for tests, and no secrets in git.
"""

from __future__ import annotations

import os
from typing import Any

from backend.app.agents.agents import VerifierAgent, WriterAgent
from backend.app.agents.models import SubQuestion, SubQuestionEvidence
from backend.app.agents.providers import AgentLlmProvider, build_provider
from backend.app.agents.tools import retrieve_tool


def generation_enabled() -> bool:
    """Read the ``GENERATION_ENABLED`` flag the ``os.getenv`` way (default false).

    Distinct from ``AGENTS_ENABLED``: the grounded-generation answer path is its
    own opt-in. There is no ``config.py`` — config is environment variables.
    """
    return os.getenv("GENERATION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


async def answer(
    query: str,
    *,
    retrieval_service: Any = None,
    provider: AgentLlmProvider | None = None,
    limit: int = 8,
    filters: dict | None = None,
) -> dict[str, Any]:
    """Produce a grounded answer for ``query`` with a citation per claim.

    Retrieves evidence, drafts a grounded answer via the provider's
    :class:`WriterAgent`, then runs the :class:`VerifierAgent` and keeps ONLY
    the claims that trace to a retained ``source_url``. Returns::

        {
            "text": str,                      # the grounded answer prose
            "citations": [                    # one per retained claim
                {
                    "source_url": str,        # always present, always retained
                    "claim": str,
                    "chunk_id": str,
                    "section": str,
                    "support_score": float,
                },
                ...
            ],
            "retained_source_urls": [str],    # the evidence the answer may cite
            "faithfulness": float,            # fraction of drafted claims that traced
            "citation_accuracy": float,       # fraction whose source_url is retained
            "dropped_claims": [str],          # claims that did NOT trace (flagged)
            "warnings": [{code, message, stage}],
            "degraded": bool,
        }

    Grounding is enforced, not assumed: a claim the writer emits that the
    verifier rejects (unsupported, or citing a non-retrieved source) is dropped
    from ``citations`` and surfaced in ``dropped_claims`` — the same provenance
    the generation eval measures.

    ``retrieval_service`` must expose the async ``retrieve(query, limit, filters,
    boosts)`` contract (the app's ``RetrievalService``); inject a fake in tests.
    ``provider`` defaults to ``build_provider()`` (deterministic offline / Ollama
    when ``AGENTS_PROVIDER=ollama``).
    """
    if retrieval_service is None:
        # Imported lazily so importing this module never requires a live stack.
        from backend.app.dependencies import build_retrieval_service

        retrieval_service = build_retrieval_service()
    provider = provider or build_provider()

    # 1. Retrieve evidence through the existing hybrid retrieval core (reused via
    #    the agents' retrieve_tool, which keeps source_url + warnings intact).
    retrieval = await retrieve_tool(retrieval_service, query, limit=limit, filters=filters)
    evidence: list[SubQuestionEvidence] = [
        SubQuestionEvidence(
            sub_question=SubQuestion(text=query),
            hits=retrieval.hits,
            warnings=retrieval.warnings,
            degraded=retrieval.degraded,
            reranked=retrieval.reranked,
        )
    ]

    # 2. Draft a grounded answer via the provider (reuse the WriterAgent path).
    briefing = WriterAgent(provider).write(query, evidence)

    # 3. Verify with the SAME gate the eval measures. Keep only tracing claims.
    verification = VerifierAgent().verify(briefing, evidence)
    by_text = {claim.text: claim for claim in briefing.claims}

    citations: list[dict[str, Any]] = []
    dropped: list[str] = []
    for verified_claim in verification.claims:
        if verified_claim.traces:
            source_claim = by_text.get(verified_claim.text)
            citations.append(
                {
                    "source_url": verified_claim.source_url,
                    "claim": verified_claim.text,
                    "chunk_id": source_claim.chunk_id if source_claim else "",
                    "section": verified_claim.section,
                    "support_score": round(verified_claim.support_score, 4),
                }
            )
        else:
            dropped.append(verified_claim.text)

    return {
        "text": briefing.answer,
        "citations": citations,
        "retained_source_urls": list(briefing.retained_source_urls),
        "faithfulness": verification.faithfulness,
        "citation_accuracy": verification.citation_accuracy,
        "dropped_claims": dropped,
        "warnings": [
            {"code": w.code, "message": w.message, "stage": w.stage} for w in briefing.warnings
        ],
        "degraded": briefing.degraded,
    }

"""Research-briefing multi-agent supervisor over the existing hybrid retrieval.

This package wraps — never replaces — the deterministic retrieval and synthesis
core. A :class:`~backend.app.agents.supervisor.SupervisorAgent` orchestrates a
planner, retrieval/rerank tools, a writer, and a provenance-enforcing verifier:

    planner -> (retrieval + rerank per sub-question) -> writer -> verifier

Invariants (see ``CLAUDE.md``):

* **Deterministic synthesis is wrapped, not replaced.** The default
  :class:`~backend.app.agents.providers.DeterministicProvider` reuses the
  existing ``answer_evidence`` + ``synthesize_answer_model`` path, so the agent
  draft is inherently grounded with ``source_url`` and the existing ``/answer``
  route is byte-for-byte unchanged when ``AGENTS_ENABLED`` is off.
* **Every claim carries a retained ``source_url``.** The
  :class:`~backend.app.agents.verifier.VerifierAgent` rejects a briefing if any
  claim does not trace to a chunk that was actually retrieved.
* **The reliability contract is honoured.** Retrieval warnings / ``degraded``
  are propagated, never swallowed; a degraded stage does not crash the run.
* **The LLM lives behind one swappable provider interface.** Ollama is the
  documented local default; the deterministic provider is the offline/test
  default. Tests never require a network.
"""

from __future__ import annotations

from backend.app.agents.models import (
    Briefing,
    BriefingClaim,
    ClaimVerification,
    SubQuestion,
    SubQuestionEvidence,
    VerificationResult,
)
from backend.app.agents.providers import (
    AgentLlmProvider,
    DeterministicProvider,
    OllamaProvider,
    build_provider,
)
from backend.app.agents.supervisor import SupervisorAgent, agents_enabled

__all__ = [
    "AgentLlmProvider",
    "Briefing",
    "BriefingClaim",
    "ClaimVerification",
    "DeterministicProvider",
    "OllamaProvider",
    "SubQuestion",
    "SubQuestionEvidence",
    "SupervisorAgent",
    "VerificationResult",
    "agents_enabled",
    "build_provider",
]

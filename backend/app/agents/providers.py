"""Swappable LLM provider interface for the planner and writer.

Two pieces of the pipeline can be driven by an LLM: decomposing the question
(planner) and drafting the briefing (writer). Both go through one
:class:`AgentLlmProvider` protocol so the LLM is the only swappable part.

* :class:`DeterministicProvider` — the DEFAULT (tests / flag-off path). Offline,
  no network. The planner decomposes via deterministic rules; the writer reuses
  the existing ``answer_evidence`` + ``synthesize_answer_model`` so the draft is
  inherently grounded with ``source_url``. This is exactly the existing
  deterministic synthesis, wrapped as a provider.
* :class:`OllamaProvider` — the wired local LLM (``OLLAMA_URL`` / ``OLLAMA_MODEL``,
  defaulting to the docker-compose ``ollama`` service). Env-selected; never
  required for tests. It must still attach the ``source_url`` of a *retrieved*
  chunk to every claim and stay grounded only in the retrieved chunks.

``build_provider()`` reads ``AGENTS_PROVIDER`` (``deterministic`` | ``ollama``)
the same ``os.getenv`` way the rest of the app reads config — there is no
``config.py``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

import httpx

from backend.app.agents.models import (
    Briefing,
    BriefingClaim,
    SubQuestion,
    SubQuestionEvidence,
)
from backend.app.api.search import (
    answer_evidence,
    ensure_sentence,
    synthesize_answer_model,
)
from backend.app.retrieval.service import RankedHit

# --- deterministic planner vocabulary ----------------------------------------

# Facets a release-intelligence question is decomposed into. Deterministic and
# offline — keyed by terms that commonly appear in Elastic release questions.
_FACETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("what changed", "Identify the concrete behavior or feature change.", ("new", "changed", "change", "release", "add", "added", "introduc")),
    ("why it matters", "Surface the engineering impact or tradeoff.", ("impact", "matters", "risk", "tradeoff", "performance", "latency", "memory")),
    ("how to adopt", "Find migration, upgrade, or configuration guidance.", ("upgrade", "migrat", "deprecat", "breaking", "config", "adopt")),
)


def _deterministic_sub_questions(query: str) -> list[SubQuestion]:
    """Decompose a question into sub-questions by deterministic rules.

    Always emits the base question first, then any facet whose trigger terms are
    absent from the query (so we broaden coverage) — capped and stable so the
    output is deterministic.
    """
    lower = query.lower()
    subs: list[SubQuestion] = [SubQuestion(text=query, rationale="Primary research question.")]
    for facet, rationale, triggers in _FACETS:
        if any(trigger in lower for trigger in triggers):
            # The query already targets this facet; sharpen it.
            subs.append(SubQuestion(text=f"{query} ({facet})", rationale=rationale))
        else:
            subs.append(SubQuestion(text=f"{query} — {facet}", rationale=rationale))
    return subs


class AgentLlmProvider(Protocol):
    """The one swappable LLM seam used by the planner and writer."""

    name: str

    def plan(self, query: str) -> list[SubQuestion]:
        """Decompose the research question into sub-questions."""
        ...

    def write(self, query: str, evidence: list[SubQuestionEvidence]) -> Briefing:
        """Draft a briefing grounded only in the retrieved chunks, attaching a
        retained ``source_url`` to every claim."""
        ...


def _retained_hits(evidence: list[SubQuestionEvidence]) -> list[RankedHit]:
    """Flatten retained hits across sub-questions, de-duped by chunk id, stable."""
    seen: set[str] = set()
    hits: list[RankedHit] = []
    for item in evidence:
        for hit in item.hits:
            if hit.id in seen:
                continue
            seen.add(hit.id)
            hits.append(hit)
    return hits


def _retained_source_urls(hits: list[RankedHit]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.source_url and hit.source_url not in seen:
            seen.add(hit.source_url)
            urls.append(hit.source_url)
    return urls


def _collected_warnings(evidence: list[SubQuestionEvidence]):
    warnings = []
    seen: set[tuple[str, str]] = set()
    degraded = False
    for item in evidence:
        degraded = degraded or item.degraded
        for warning in item.warnings:
            key = (warning.code, warning.stage)
            if key in seen:
                continue
            seen.add(key)
            warnings.append(warning)
    return warnings, degraded


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


class DeterministicProvider:
    """Offline default. Planner = deterministic rules; writer = the existing
    deterministic synthesis (``answer_evidence`` + ``synthesize_answer_model``)."""

    name = "deterministic"

    def plan(self, query: str) -> list[SubQuestion]:
        return _deterministic_sub_questions(query)

    def write(self, query: str, evidence: list[SubQuestionEvidence]) -> Briefing:
        hits = _retained_hits(evidence)
        retained_urls = _retained_source_urls(hits)
        warnings, degraded = _collected_warnings(evidence)

        # Reuse the existing deterministic synthesis verbatim. ``answer_evidence``
        # already drops archived sources and carries source_url on every item.
        answer_evidences = answer_evidence(query, hits, limit=4)
        model = synthesize_answer_model(query, answer_evidences)

        # Build the verifiable claim set. A *claim* is a factual statement that
        # MUST trace to a retained chunk; the synthesizer also emits generic
        # presentation scaffolding ("This matters because...", "Focus on the
        # section...") which is guidance, not a traceable factual claim. We only
        # promote a sentence to a claim when it is token-supported (>= threshold)
        # by an evidence item drawn from a retained chunk, and attribute it to
        # that chunk's source_url. Scaffolding that supports nothing is left out
        # of the claim set (it still appears in answer/what_new/why_it_matters
        # prose), and the VerifierAgent is the independent gate over what remains.
        claims: list[BriefingClaim] = []
        seen_claims: set[tuple[str, str]] = set()

        def _claim_for(text: str, section: str) -> None:
            text = text.strip()
            if not text:
                return
            owner = _supporting_evidence_for(text, answer_evidences)
            if owner is None:
                return
            sentence = ensure_sentence(text)
            key = (sentence.lower(), owner.source_url)
            if key in seen_claims:
                return
            seen_claims.add(key)
            claims.append(
                BriefingClaim(
                    text=sentence,
                    source_url=owner.source_url,
                    chunk_id=_chunk_id_for(owner.source_url, hits),
                    section=section,
                )
            )

        _claim_for(model["direct_answer"], "answer")
        for sentence in _split_sentences(model["explanation"] or ""):
            _claim_for(sentence, "answer")
        if model.get("what_new"):
            _claim_for(model["what_new"], "what_new")
        for item in model.get("what_new_items", []) or []:
            _claim_for(item, "what_new")
        # Always include each evidence excerpt as a claim: it is drawn verbatim
        # from a retained chunk, so it traces by construction and guarantees the
        # briefing has at least one grounded claim per source.
        for ev in answer_evidences:
            _claim_for(ev.excerpt, "answer")

        return Briefing(
            query=query,
            answer=ensure_sentence(model["direct_answer"]) if model["direct_answer"] else "",
            what_new=model.get("what_new"),
            why_it_matters=model.get("important"),
            claims=claims,
            sub_questions=[item.sub_question for item in evidence],
            retained_source_urls=retained_urls,
            warnings=warnings,
            degraded=degraded,
        )


_CLAIM_SUPPORT_THRESHOLD = 0.6


def _supporting_evidence_for(text: str, evidences):
    """Return the evidence item that token-supports ``text`` above threshold.

    Deterministic: the evidence whose excerpt/claim has the highest token
    containment of the sentence wins (ties broken toward the primary, rank-0
    evidence). If no evidence reaches the support threshold the sentence is NOT
    a traceable factual claim (it is generic scaffolding) and ``None`` is
    returned so the writer drops it from the claim set. Containment is the same
    notion ``relevance_eval.TokenOverlapScorer`` uses, so the writer's promotion
    rule and the verifier's gate agree.
    """
    if not evidences:
        return None
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    if not tokens:
        return None
    best = None
    best_score = 0.0
    for index, ev in enumerate(evidences):
        haystack = f"{ev.claim} {ev.excerpt}".lower()
        hay_tokens = set(re.findall(r"[a-z0-9]+", haystack))
        overlap = len(tokens & hay_tokens) / len(tokens)
        score = overlap - index * 1e-6
        if score > best_score:
            best_score = score
            best = ev
    return best if best_score >= _CLAIM_SUPPORT_THRESHOLD else None


def _chunk_id_for(source_url: str, hits: list[RankedHit]) -> str:
    for hit in hits:
        if hit.source_url == source_url:
            return hit.id
    return ""


# --- Ollama provider ---------------------------------------------------------


def _ollama_url() -> str:
    # Default to the wired docker-compose Ollama service (compose name ``llm``).
    return os.getenv("OLLAMA_URL", "http://llm:11434")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.2")


_WRITER_SYSTEM = (
    "You are a release-intelligence briefing writer. Use ONLY the provided "
    "evidence chunks. Every claim you make MUST be grounded in one chunk and "
    "tagged with that chunk's exact source_url. Do not invent facts or URLs. "
    "Respond with strict JSON: {\"answer\": str, \"what_new\": str, "
    "\"why_it_matters\": str, \"claims\": [{\"text\": str, \"source_url\": str, "
    "\"section\": \"answer\"|\"what_new\"|\"why_it_matters\"}]}."
)


class OllamaProvider:
    """LLM provider backed by the wired local Ollama service.

    The planner still uses the deterministic decomposition (cheap and stable);
    only the writer calls the model. The model is instructed to ground every
    claim in a retrieved chunk and tag it with that chunk's ``source_url``; any
    claim it emits with an unknown ``source_url`` is dropped here, and the
    VerifierAgent is the hard gate that rejects anything that still does not
    trace. If the model is unreachable, we fall back to the deterministic writer
    and add a degradation warning — honouring the reliability contract.
    """

    name = "ollama"

    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 60.0) -> None:
        self.url = url or _ollama_url()
        self.model = model or _ollama_model()
        self.timeout = timeout
        self._fallback = DeterministicProvider()

    def plan(self, query: str) -> list[SubQuestion]:
        return _deterministic_sub_questions(query)

    def write(self, query: str, evidence: list[SubQuestionEvidence]) -> Briefing:
        hits = _retained_hits(evidence)
        retained_urls = set(_retained_source_urls(hits))
        warnings, degraded = _collected_warnings(evidence)

        try:
            raw = self._generate(query, hits)
            parsed = json.loads(raw)
        except Exception:  # noqa: BLE001 — degrade to deterministic synthesis
            fallback = self._fallback.write(query, evidence)
            from backend.app.retrieval.service import RetrievalWarning

            extra = RetrievalWarning(
                code="agent_llm_unavailable",
                message="Local LLM unavailable; briefing used deterministic synthesis.",
                stage="writer",
            )
            return Briefing(
                query=fallback.query,
                answer=fallback.answer,
                what_new=fallback.what_new,
                why_it_matters=fallback.why_it_matters,
                claims=fallback.claims,
                sub_questions=fallback.sub_questions,
                retained_source_urls=fallback.retained_source_urls,
                warnings=[*fallback.warnings, extra],
                degraded=True,
            )

        claims: list[BriefingClaim] = []
        for entry in parsed.get("claims", []):
            url = str(entry.get("source_url") or "")
            text = str(entry.get("text") or "").strip()
            section = str(entry.get("section") or "answer")
            if not text or url not in retained_urls:
                # Drop hallucinated citations defensively; verifier is the gate.
                continue
            claims.append(
                BriefingClaim(
                    text=ensure_sentence(text),
                    source_url=url,
                    chunk_id=_chunk_id_for(url, hits),
                    section=section,
                )
            )

        return Briefing(
            query=query,
            answer=str(parsed.get("answer") or "").strip(),
            what_new=(str(parsed.get("what_new")).strip() or None) if parsed.get("what_new") else None,
            why_it_matters=(str(parsed.get("why_it_matters")).strip() or None) if parsed.get("why_it_matters") else None,
            claims=claims,
            sub_questions=[item.sub_question for item in evidence],
            retained_source_urls=list(_retained_source_urls(hits)),
            warnings=warnings,
            degraded=degraded,
        )

    def _generate(self, query: str, hits: list[RankedHit]) -> str:
        evidence_blob = "\n\n".join(
            f"[chunk {hit.id}] source_url={hit.source_url}\n{hit.text[:1200]}" for hit in hits
        )
        prompt = (
            f"Research question: {query}\n\nEvidence chunks:\n{evidence_blob}\n\n"
            "Write the briefing as strict JSON now."
        )
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.url.rstrip('/')}/api/generate",
                json={
                    "model": self.model,
                    "system": _WRITER_SYSTEM,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()
            data = response.json()
        return str(data.get("response") or "")


def build_provider(name: str | None = None) -> AgentLlmProvider:
    """Select a provider by name or ``AGENTS_PROVIDER`` env (default deterministic)."""
    selected = (name or os.getenv("AGENTS_PROVIDER", "deterministic")).strip().lower()
    if selected == "ollama":
        return OllamaProvider()
    return DeterministicProvider()

"""Offline faithfulness eval for the grounded-generation path.

Mirrors the relevance eval (``app/eval/run_eval.py``) but scores *generation*
instead of *retrieval*. On a FIXED, committed fixture (questions + a small GOLD
answer set + canned retrieved chunks) it measures three numbers, with no live
ES / Qdrant / TEI / Ollama:

* **citation-accuracy** — fraction of the answer's claims whose attached
  ``source_url`` is in the retrieved set. Comes straight from the
  :class:`VerifierAgent` (reused — the same gate production uses).
* **faithfulness** — fraction of claims token-supported by their cited sources
  via ``relevance_eval.faithfulness_score`` (the shared 5.2 skill). A claim that
  is unsupported by / contradicts its sources fails. (True contradiction
  detection is the LLM/NLI-backed extension behind the provider + scorer
  interfaces; the offline default uses token support, like the relevance lab.)
* **answer-correctness** — the generated answer vs the gold answer, deterministic
  token-overlap F1 (``relevance_eval.faithfulness_score`` containment in both
  directions). Reported per-question and as a mean.

The metrics are shaped into the SAME report schema the ``relevance_eval`` report
uses (one ``strategy`` = ``generation``; each metric carried at ``@1``) so the
gate reuses ``evaluate_thresholds`` / ``to_json`` / ``to_markdown`` verbatim —
generation is gated on faithfulness exactly the way search is gated on relevance.

A case may carry an ``injected_answer``: that exact answer is scored instead of
the provider's output, used to prove the eval CATCHES an unfaithful answer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from relevance_eval import faithfulness_score

from backend.app.agents.agents import VerifierAgent
from backend.app.agents.models import (
    Briefing,
    BriefingClaim,
    SubQuestion,
    SubQuestionEvidence,
)
from backend.app.agents.providers import DeterministicProvider
from backend.app.agents.tools import retrieve_tool
from backend.app.generation.answer import answer as generate_answer
from backend.app.ingest.metadata import normalize_metadata
from backend.app.retrieval.service import RankedHit

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "fixtures" / "generation_eval.json"

# One strategy, mirroring the relevance harness's per-strategy report shape.
STRATEGY = "generation"
# Each generation metric is a single aggregate carried at k=1 so it slots into
# the relevance_eval report/threshold schema (`<metric>@1`) with no changes.
K = 1

_TOKEN = re.compile(r"[a-z0-9]+")


class _FixtureRetrievalService:
    """Async ``retrieve`` returning a case's canned chunks for any query."""

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


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def correctness_f1(generated: str, gold: str) -> float:
    """Deterministic token-overlap F1 between generated and gold answers.

    Precision = fraction of generated tokens present in gold (containment of the
    generated answer in the gold), recall = fraction of gold tokens present in
    the generated answer; both are exactly ``relevance_eval.faithfulness_score``
    containment, so correctness uses the SAME token-overlap notion as
    faithfulness. F1 is their harmonic mean. Empty answers score 0.0.
    """
    gen_tokens = _tokens(generated)
    gold_tokens = _tokens(gold)
    if not gen_tokens or not gold_tokens:
        return 0.0
    # precision: generated answer's claims contained in gold (faithfulness_score
    # treats `gold` as the single source); recall: gold contained in generated.
    precision = faithfulness_score(generated, [gold], support_threshold=0.0)
    recall = faithfulness_score(gold, [generated], support_threshold=0.0)
    # Use the token-containment scores (best_source support), not the pass count.
    p = _mean_support(precision)
    r = _mean_support(recall)
    if p + r == 0:
        return 0.0
    return round(2 * p * r / (p + r), 4)


def _mean_support(faith: dict[str, Any]) -> float:
    claims = faith.get("claims") or []
    if not claims:
        return 0.0
    return sum(float(c.get("support_score") or 0.0) for c in claims) / len(claims)


async def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Score one case: citation-accuracy, faithfulness, correctness.

    If the case carries an ``injected_answer`` we score that exact answer (its
    sentences become claims cited to the retained sources) instead of the
    provider's output — used to prove the eval catches an unfaithful answer.
    Otherwise we run the real grounded-generation path (deterministic provider).
    """
    chunks = [_chunk_to_hit(chunk, rank) for rank, chunk in enumerate(case["chunks"], start=1)]
    service = _FixtureRetrievalService(chunks)
    gold = str(case.get("gold_answer") or "")

    if "injected_answer" in case:
        generated_text = str(case["injected_answer"])
        verification = await _verify_injected(service, case["query"], generated_text, chunks)
    else:
        result = await generate_answer(
            case["query"], retrieval_service=service, provider=DeterministicProvider()
        )
        generated_text = _answer_with_claims(result)
        verification = {
            "citation_accuracy": result["citation_accuracy"],
            "faithfulness": result["faithfulness"],
            "dropped": result["dropped_claims"],
        }

    return {
        "id": case["id"],
        "query": case["query"],
        "citation_accuracy": round(verification["citation_accuracy"], 4),
        "faithfulness": round(verification["faithfulness"], 4),
        "correctness": correctness_f1(generated_text, gold) if gold else 0.0,
        "unsupported_claims": verification["dropped"],
    }


def _answer_with_claims(result: dict[str, Any]) -> str:
    """The text the correctness metric scores: the answer plus its cited claims.

    The deterministic answer prose is a single grounded sentence; its claim set
    carries the rest of the grounded content. Joining them gives correctness a
    fair view of everything the generation asserted, all of it grounded.
    """
    parts = [result["text"]] + [c["claim"] for c in result["citations"]]
    return " ".join(part for part in parts if part)


async def _verify_injected(
    service: _FixtureRetrievalService,
    query: str,
    injected_answer: str,
    chunks: list[RankedHit],
) -> dict[str, Any]:
    """Score an injected answer through the SAME VerifierAgent the path uses.

    Each sentence of the injected answer becomes a claim cited to the best-
    matching retained source (so citation-accuracy stays high — the answer DOES
    cite retrieved sources); faithfulness then exposes the sentence whose tokens
    the cited source does not support.
    """
    retrieval = await retrieve_tool(service, query)
    evidence = [
        SubQuestionEvidence(
            sub_question=SubQuestion(text=query),
            hits=retrieval.hits,
            warnings=retrieval.warnings,
            degraded=retrieval.degraded,
        )
    ]
    retained_urls = [hit.source_url for hit in chunks]
    claims = [
        BriefingClaim(
            text=sentence,
            source_url=_best_source(sentence, chunks),
            chunk_id="",
            section="answer",
        )
        for sentence in _split_sentences(injected_answer)
    ]
    briefing = Briefing(
        query=query,
        answer=injected_answer,
        what_new=None,
        why_it_matters=None,
        claims=claims,
        sub_questions=[SubQuestion(text=query)],
        retained_source_urls=retained_urls,
    )
    result = VerifierAgent().verify(briefing, evidence)
    return {
        "citation_accuracy": result.citation_accuracy,
        "faithfulness": result.faithfulness,
        "dropped": [c.text for c in result.unsupported_claims],
    }


def _best_source(sentence: str, chunks: list[RankedHit]) -> str:
    """Cite the retained chunk whose text best token-overlaps the sentence."""
    tokens = _tokens(sentence)
    best_url = chunks[0].source_url if chunks else ""
    best = -1.0
    for chunk in chunks:
        overlap = len(tokens & _tokens(chunk.text))
        if overlap > best:
            best = overlap
            best_url = chunk.source_url
    return best_url


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def build_report(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Shape the per-case scores into the relevance_eval report schema.

    One ``strategy`` (``generation``); each generation metric is the MEAN over
    cases, carried at ``@1`` so ``evaluate_thresholds`` (which keys on
    ``<metric>@<k>``) and ``to_markdown`` work unchanged.
    """
    n = len(case_reports) or 1
    means = {
        metric: round(sum(c[metric] for c in case_reports) / n, 4)
        for metric in ("citation_accuracy", "faithfulness", "correctness")
    }
    return {
        "queries": len(case_reports),
        "ks": [K],
        "metrics": ["citation_accuracy", "faithfulness", "correctness"],
        "strategies": {
            STRATEGY: {
                "metrics": {metric: {str(K): value} for metric, value in means.items()},
            }
        },
        # Carried alongside the schema for the JSON report / tests; ignored by
        # the shared renderer/gate.
        "cases": case_reports,
        "summary": {
            "case_count": len(case_reports),
            "mean_citation_accuracy": means["citation_accuracy"],
            "mean_faithfulness": means["faithfulness"],
            "mean_correctness": means["correctness"],
        },
    }


async def run_eval(fixture_path: Path) -> dict[str, Any]:
    import json

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    case_reports = [await evaluate_case(case) for case in fixture["cases"]]
    return build_report(case_reports)

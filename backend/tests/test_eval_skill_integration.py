"""Offline tests for the relevance-eval skill integration.

No live Postgres / Qdrant / TEI is required. Two layers are covered:

(a) the *skill wiring* — ``run_evaluation`` + ``evaluate_thresholds`` over the
    committed judgments/thresholds with a canned sync ``search_fn``, asserting
    report shape and a known pass and a known fail; and
(b) the *adapter* — ``make_search_fn`` over a fake async ``retrieve`` returning
    canned ``RankedHit``s, asserting ``search(query, "hybrid")`` yields the
    expected chunk ids (the adapter uses ``asyncio.run`` internally; the test
    stays synchronous).
"""

from __future__ import annotations

import json
from pathlib import Path

from relevance_eval import evaluate_thresholds, load_thresholds, run_evaluation

from backend.app.eval.skill_adapter import HYBRID_STRATEGY, STRATEGIES, make_search_fn
from backend.app.retrieval.service import RankedHit

EVAL_DIR = Path(__file__).resolve().parents[1] / "app" / "eval"
JUDGMENTS_PATH = EVAL_DIR / "judgments.example.json"
THRESHOLDS_PATH = EVAL_DIR / "thresholds.example.json"
KS = (1, 3, 5, 10)


def _load_judgments() -> dict:
    return json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))


def _relevant_ids(value) -> list[str]:
    # judgments value is either a list of ids or a {id: grade} mapping.
    return list(value)


def _perfect_search_fn(judgments: dict):
    """Return every query's relevant ids first — a known-good ranking."""

    def search(query: str, strategy: str) -> list[str]:
        assert strategy == HYBRID_STRATEGY
        return _relevant_ids(judgments[query]) + ["chunk-noise"]

    return search


def _empty_search_fn(judgments: dict):
    """Return only irrelevant ids — a known-bad ranking."""

    def search(query: str, strategy: str) -> list[str]:
        return ["chunk-noise-1", "chunk-noise-2"]

    return search


# --- (a) skill wiring over committed judgments + thresholds -------------------


def test_run_evaluation_report_shape_and_gate_pass():
    judgments = _load_judgments()
    thresholds = load_thresholds(str(THRESHOLDS_PATH))

    report = run_evaluation(judgments, _perfect_search_fn(judgments), list(STRATEGIES), ks=KS)

    assert report["queries"] == len(judgments)
    assert report["ks"] == list(KS)
    assert set(report["strategies"]) == {HYBRID_STRATEGY}
    hybrid = report["strategies"][HYBRID_STRATEGY]
    # Perfect ranking => precision@1, mrr, ndcg all 1.0 at the gated ks.
    assert hybrid["metrics"]["precision"]["1"] == 1.0
    assert hybrid["metrics"]["mrr"]["3"] == 1.0
    assert hybrid["metrics"]["ndcg"]["3"] == 1.0

    gate = evaluate_thresholds(report, thresholds)
    assert gate["passed"] is True
    assert gate["checks"]  # thresholds were actually applied


def test_run_evaluation_gate_fails_on_bad_rankings():
    judgments = _load_judgments()
    thresholds = load_thresholds(str(THRESHOLDS_PATH))

    report = run_evaluation(judgments, _empty_search_fn(judgments), list(STRATEGIES), ks=KS)
    hybrid = report["strategies"][HYBRID_STRATEGY]
    assert hybrid["metrics"]["precision"]["1"] == 0.0

    gate = evaluate_thresholds(report, thresholds)
    assert gate["passed"] is False


# --- (b) the adapter over a fake async retrieve -------------------------------


class FakeRetrievalService:
    """Async ``retrieve`` returning canned RankedHits in a fixed order."""

    def __init__(self, hits: list[RankedHit]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    async def retrieve(self, query, limit=10, filters=None, boosts=None):
        self.calls.append({"query": query, "limit": limit})
        return {"hits": self.hits[:limit], "warnings": [], "degraded": False}


def _hit(chunk_id: str) -> RankedHit:
    return RankedHit(
        id=chunk_id,
        score=0.9,
        metadata={"repo": "elastic/docs-content", "path": "p.md"},
        source_url=f"https://example.test/{chunk_id}",
        text="text",
    )


def test_adapter_extracts_chunk_ids_in_order():
    service = FakeRetrievalService([_hit("chunk-a"), _hit("chunk-b"), _hit("chunk-c")])
    search = make_search_fn(service, limit=10)

    result = search("any query", HYBRID_STRATEGY)

    assert result == ["chunk-a", "chunk-b", "chunk-c"]
    assert service.calls == [{"query": "any query", "limit": 10}]


def test_adapter_respects_limit():
    service = FakeRetrievalService([_hit("chunk-a"), _hit("chunk-b"), _hit("chunk-c")])
    search = make_search_fn(service, limit=2)

    assert search("q", HYBRID_STRATEGY) == ["chunk-a", "chunk-b"]


def test_adapter_rejects_unknown_strategy():
    service = FakeRetrievalService([_hit("chunk-a")])
    search = make_search_fn(service)

    try:
        search("q", "lexical_only")
    except ValueError as exc:
        assert "lexical_only" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError for an unknown strategy")

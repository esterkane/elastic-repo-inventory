"""Unit tests for the MCP tool handlers.

These exercise the pure async handlers in ``backend/app/mcp/tools.py`` against
fake repositories — no Postgres / Qdrant / TEI is required. Each handler is run
with ``asyncio.run`` so the suite needs no extra async-test plugin.
"""

from __future__ import annotations

import asyncio

from sqlalchemy.exc import SQLAlchemyError

from backend.app.mcp.tools import (
    get_chunk_impl,
    hybrid_search_impl,
    list_sources_impl,
    rerank_impl,
)
from backend.app.retrieval.service import RankedHit


def make_hit(chunk_id: str, *, score: float = 0.9, text: str = "Hybrid search fuses lexical and dense retrieval.", **metadata) -> RankedHit:
    base = {
        "repo": "elastic/docs-content",
        "path": "guide/page.md",
        "title": "Hybrid search",
        "heading_path": "Guide > Hybrid",
        "content_type": "guide",
        "license_family": "elastic-license",
        "final_rank": 1,
    }
    base.update(metadata)
    return RankedHit(
        id=chunk_id,
        score=score,
        metadata=base,
        source_url=f"https://example.test/{chunk_id}#combine",
        text=text,
    )


class FakeRetrievalService:
    def __init__(self, hits, warnings=None, degraded=False) -> None:
        self.hits = hits
        self.warnings = warnings or []
        self.degraded = degraded
        self.calls: list[dict] = []

    async def retrieve(self, query, limit=10, filters=None, boosts=None):
        self.calls.append({"query": query, "limit": limit, "filters": filters})
        return {
            "hits": self.hits,
            "recommendation_categories": [],
            "warnings": self.warnings,
            "degraded": self.degraded,
        }


class FakeLexical:
    def __init__(self, chunks=None, sources=None) -> None:
        self.chunks = chunks or {}
        self.sources = sources if sources is not None else []

    async def get(self, chunk_id):
        return self.chunks.get(chunk_id)

    async def list_sources(self):
        return self.sources


class FakeReranker:
    """Deterministically reverses the input order and stamps a rerank_score."""

    async def rerank(self, query, hits):
        reordered = list(reversed(hits))
        return [
            RankedHit(
                id=hit.id,
                score=1.0 - index * 0.1,
                metadata=hit.metadata,
                source_url=hit.source_url,
                text=hit.text,
                rerank_score=1.0 - index * 0.1,
            )
            for index, hit in enumerate(reordered)
        ]


# --- hybrid_search -----------------------------------------------------------


def test_hybrid_search_returns_shaped_hits():
    service = FakeRetrievalService(hits=[make_hit("chunk-1"), make_hit("chunk-2", score=0.7)])
    result = asyncio.run(hybrid_search_impl("hybrid search", limit=5, service=service))

    assert not result.get("isError")
    assert result["query"] == "hybrid search"
    assert result["count"] == 2
    assert result["hits"][0]["id"] == "chunk-1"
    assert result["hits"][0]["source_url"].startswith("https://")
    # the limit is threaded through to the underlying retrieval service
    assert service.calls[0]["limit"] == 5


def test_hybrid_search_explain_includes_score_breakdown():
    service = FakeRetrievalService(hits=[make_hit("chunk-1")])
    result = asyncio.run(hybrid_search_impl("q", explain=True, service=service))
    assert "score_breakdown" in result["hits"][0]


def test_hybrid_search_rejects_empty_query():
    service = FakeRetrievalService(hits=[])
    result = asyncio.run(hybrid_search_impl("", service=service))
    assert result["isError"] is True
    assert result["errorCategory"] == "validation"
    assert result["isRetryable"] is False
    # an invalid request must not reach the backend
    assert service.calls == []


def test_hybrid_search_rejects_non_object_filters():
    service = FakeRetrievalService(hits=[])
    result = asyncio.run(hybrid_search_impl("q", filters=["not", "a", "dict"], service=service))
    assert result["isError"] is True
    assert result["errorCategory"] == "validation"


def test_hybrid_search_surfaces_degraded_warnings():
    service = FakeRetrievalService(
        hits=[make_hit("chunk-1")],
        warnings=[{"code": "semantic_unavailable", "message": "down", "stage": "semantic_retrieval"}],
        degraded=True,
    )
    result = asyncio.run(hybrid_search_impl("q", service=service))
    assert result["degraded"] is True
    assert result["warnings"][0]["code"] == "semantic_unavailable"


# --- get_chunk ---------------------------------------------------------------


def test_get_chunk_found():
    repo = FakeLexical(chunks={"chunk-1": make_hit("chunk-1", text="full chunk text")})
    result = asyncio.run(get_chunk_impl("chunk-1", lexical_repository=repo))
    assert result["chunk"]["id"] == "chunk-1"
    assert result["chunk"]["text"] == "full chunk text"
    assert result["chunk"]["repo"] == "elastic/docs-content"


def test_get_chunk_missing_is_business_error():
    repo = FakeLexical(chunks={})
    result = asyncio.run(get_chunk_impl("nope", lexical_repository=repo))
    assert result["isError"] is True
    assert result["errorCategory"] == "business"


def test_get_chunk_backend_error_is_transient():
    class Boom:
        async def get(self, chunk_id):
            raise SQLAlchemyError("connection refused")

        async def list_sources(self):
            return []

    result = asyncio.run(get_chunk_impl("a", lexical_repository=Boom()))
    assert result["isError"] is True
    assert result["errorCategory"] == "transient"
    assert result["isRetryable"] is True


# --- rerank ------------------------------------------------------------------


def test_rerank_reorders_and_reports_missing():
    chunks = {"a": make_hit("a", text="alpha"), "b": make_hit("b", text="beta")}
    repo = FakeLexical(chunks=chunks)
    result = asyncio.run(
        rerank_impl("q", ["a", "b", "ghost"], lexical_repository=repo, reranker_client=FakeReranker())
    )
    assert not result.get("isError")
    assert result["missing"] == ["ghost"]
    assert result["count"] == 2
    assert [chunk["id"] for chunk in result["chunks"]] == ["b", "a"]
    assert result["chunks"][0]["rerank_score"] is not None


def test_rerank_without_reranker_is_business_error():
    repo = FakeLexical(chunks={"a": make_hit("a")})
    result = asyncio.run(rerank_impl("q", ["a"], lexical_repository=repo, reranker_client=None))
    assert result["isError"] is True
    assert result["errorCategory"] == "business"


def test_rerank_all_missing_is_business_error():
    repo = FakeLexical(chunks={})
    result = asyncio.run(
        rerank_impl("q", ["x"], lexical_repository=repo, reranker_client=FakeReranker())
    )
    assert result["isError"] is True
    assert result["errorCategory"] == "business"


def test_rerank_rejects_empty_chunk_ids():
    repo = FakeLexical(chunks={})
    result = asyncio.run(
        rerank_impl("q", [], lexical_repository=repo, reranker_client=FakeReranker())
    )
    assert result["isError"] is True
    assert result["errorCategory"] == "validation"


# --- list_sources ------------------------------------------------------------


def test_list_sources():
    repo = FakeLexical(
        sources=[
            {"repo": "elastic/docs-content", "chunk_count": 12},
            {"repo": "elastic/elasticsearch-labs", "chunk_count": 5},
        ]
    )
    result = asyncio.run(list_sources_impl(lexical_repository=repo))
    assert result["count"] == 2
    assert result["sources"][0]["repo"] == "elastic/docs-content"


def test_list_sources_empty():
    result = asyncio.run(list_sources_impl(lexical_repository=FakeLexical(sources=[])))
    assert result["count"] == 0
    assert result["sources"] == []

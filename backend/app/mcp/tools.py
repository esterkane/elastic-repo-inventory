"""MCP tool handlers wrapping the existing retrieval core.

These are plain async functions — no FastMCP or HTTP coupling — so they can be
unit-tested directly with fake repositories. :mod:`app.mcp.server` registers thin
FastMCP wrappers that supply the cached resource singletons.

Each handler is wrapped by :func:`app.mcp.errors.guard`, so it either returns a
structured success payload or a structured error payload — never a raised
exception or a stack trace. Search results are shaped by the *same*
``hit_response`` helper the ``/api/v1/search`` route uses, so MCP and the HTTP
API return identical chunk objects.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from backend.app.api.search import hit_response
from backend.app.ingest.metadata import normalize_filters
from backend.app.mcp.errors import ToolBusinessError, ToolValidationError, guard
from backend.app.retrieval.service import RankedHit, RetrievalWarning

MAX_LIMIT = 50
MAX_RERANK_IDS = 100


# --- input validation models (pydantic, mirroring the /search request) -------


class HybridSearchInput(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=MAX_LIMIT)
    explain: bool = False


class GetChunkInput(BaseModel):
    chunk_id: str = Field(min_length=1)


class RerankInput(BaseModel):
    query: str = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1, max_length=MAX_RERANK_IDS)


# --- minimal structural deps, so tools can be tested with fakes --------------


class SupportsRetrieve(Protocol):
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        boosts: dict | None = None,
    ) -> dict[str, Any]: ...


class SupportsChunkLookup(Protocol):
    async def get(self, chunk_id: str) -> RankedHit | None: ...

    async def list_sources(self) -> list[dict[str, Any]]: ...


class SupportsRerank(Protocol):
    async def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]: ...


# --- helpers -----------------------------------------------------------------


def _validate(model: type[BaseModel], **data: Any) -> Any:
    try:
        return model(**data)
    except ValidationError as exc:
        raise ToolValidationError(
            "Invalid tool input.", details={"errors": exc.errors(include_url=False)}
        ) from exc


def _normalize_filters(filters: dict | None) -> dict | None:
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise ToolValidationError(
            "`filters` must be an object/dict when provided.", details={"filters": filters}
        )
    return normalize_filters(filters) or None


def _warning(warning: object) -> dict[str, str]:
    if isinstance(warning, RetrievalWarning):
        return {"code": warning.code, "message": warning.message, "stage": warning.stage}
    if isinstance(warning, dict):
        return {
            "code": str(warning.get("code") or "warning"),
            "message": str(warning.get("message") or "Retrieval warning."),
            "stage": str(warning.get("stage") or "retrieval"),
        }
    return {"code": "warning", "message": str(warning), "stage": "retrieval"}


def _chunk_payload(hit: RankedHit) -> dict[str, Any]:
    metadata = dict(hit.metadata)
    return {
        "id": hit.id,
        "text": hit.text,
        "source_url": hit.source_url,
        "repo": metadata.get("repo"),
        "path": metadata.get("path"),
        "title": metadata.get("title"),
        "heading_path": metadata.get("heading_path"),
        "content_type": metadata.get("content_type"),
        "license_family": metadata.get("license_family"),
    }


# --- tool handlers -----------------------------------------------------------


@guard("hybrid_search")
async def hybrid_search_impl(
    query: str,
    *,
    limit: int = 10,
    filters: dict | None = None,
    explain: bool = False,
    service: SupportsRetrieve,
) -> dict[str, Any]:
    """Hybrid (lexical + dense, RRF-fused) retrieval. Returns shaped hits."""
    params = _validate(HybridSearchInput, query=query, limit=limit, explain=explain)
    normalized = _normalize_filters(filters)
    result = await service.retrieve(params.query, limit=params.limit, filters=normalized)
    raw_hits = result.get("hits", [])
    hits = [hit for hit in raw_hits if isinstance(hit, RankedHit)][: params.limit]
    return {
        "query": params.query,
        "filters": normalized or {},
        "count": len(hits),
        "degraded": bool(result.get("degraded", False)),
        "warnings": [_warning(warning) for warning in result.get("warnings", [])],
        "hits": [
            hit_response(hit, params.query, include_debug=params.explain).model_dump(
                exclude_none=True
            )
            for hit in hits
        ],
    }


@guard("get_chunk")
async def get_chunk_impl(
    chunk_id: str,
    *,
    lexical_repository: SupportsChunkLookup,
) -> dict[str, Any]:
    """Fetch a single chunk by its id, with full provenance metadata."""
    params = _validate(GetChunkInput, chunk_id=chunk_id)
    hit = await lexical_repository.get(params.chunk_id)
    if hit is None:
        raise ToolBusinessError(
            "No chunk exists with that id.", details={"chunk_id": params.chunk_id}
        )
    return {"chunk": _chunk_payload(hit)}


@guard("rerank")
async def rerank_impl(
    query: str,
    chunk_ids: list[str],
    *,
    lexical_repository: SupportsChunkLookup,
    reranker_client: SupportsRerank | None,
) -> dict[str, Any]:
    """Re-score a fixed set of chunks against a query with the cross-encoder."""
    params = _validate(RerankInput, query=query, chunk_ids=chunk_ids)
    if reranker_client is None:
        raise ToolBusinessError(
            "Reranking is not configured for this environment (no TEI_RERANK_URL)."
        )
    hits: list[RankedHit] = []
    missing: list[str] = []
    for chunk_id in params.chunk_ids:
        hit = await lexical_repository.get(chunk_id)
        if hit is None:
            missing.append(chunk_id)
        else:
            hits.append(hit)
    if not hits:
        raise ToolBusinessError(
            "None of the provided chunk_ids exist.", details={"missing": missing}
        )
    reranked = await reranker_client.rerank(params.query, hits)
    return {
        "query": params.query,
        "count": len(reranked),
        "missing": missing,
        "chunks": [
            {**_chunk_payload(hit), "score": hit.score, "rerank_score": hit.rerank_score}
            for hit in reranked
        ],
    }


@guard("list_sources")
async def list_sources_impl(
    *,
    lexical_repository: SupportsChunkLookup,
) -> dict[str, Any]:
    """List the distinct indexed sources (repos) and their chunk counts."""
    sources = await lexical_repository.list_sources()
    return {"count": len(sources), "sources": sources}

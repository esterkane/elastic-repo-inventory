"""FastMCP server exposing elastic-repo-inventory's retrieval core as agent tools.

Exposes four read-only tools that any MCP client (a LangGraph agent, Claude Code,
Cursor) can call:

- ``hybrid_search`` — lexical + dense retrieval fused with RRF.
- ``get_chunk`` — fetch a single chunk by id, with full provenance.
- ``rerank`` — re-score a fixed set of chunks against a query.
- ``list_sources`` — the catalog of indexed repositories.

The tool *logic* lives in :mod:`app.mcp.tools` as plain async functions; the
wrappers here only supply the cached resource singletons. Ingestion/admin is
deliberately NOT exposed — the MCP surface is read-only by design. (If a mutating
tool is ever added, gate it behind ``MCP_ALLOW_MUTATIONS`` defaulting to false.)

Transport is selected by ``MCP_TRANSPORT``: ``stdio`` (default, for local dev and
Claude Code) or ``http`` (streamable-HTTP on ``MCP_HTTP_HOST`` / ``MCP_HTTP_PORT``).

Run it with::

    python -m backend.app.mcp.server
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from backend.app.mcp.resources import (
    get_lexical_repository,
    get_reranker_client,
    get_retrieval_service,
)
from backend.app.mcp.tools import (
    get_chunk_impl,
    hybrid_search_impl,
    list_sources_impl,
    rerank_impl,
)

mcp = FastMCP(
    "elastic-repo-inventory",
    host=os.getenv("MCP_HTTP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_HTTP_PORT", "8765")),
)


@mcp.tool()
async def hybrid_search(
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    explain: bool = False,
) -> dict[str, Any]:
    """Search the indexed Elastic documentation with hybrid retrieval.

    WHAT IT DOES: Runs lexical (Postgres full-text) and dense (Qdrant) retrieval
    and fuses them with reciprocal rank fusion, applies metadata boosts and
    source de-duplication, and returns the top hits — each with its score,
    snippet, and full provenance (repo, path, heading_path, content_type,
    license_family, source_url). This is the same pipeline the `/api/v1/search`
    route uses, and returns the same hit shape.

    WHEN TO USE: To gather evidence about what changed in Elasticsearch or how a
    feature works, before reasoning or composing an answer yourself.

    WHEN NOT TO USE: To fetch a known chunk by id (use `get_chunk`) or to list
    which repositories are indexed (use `list_sources`).

    INPUTS:
      - query (str, required): natural-language search query.
      - limit (int, default 10, 1..50): number of hits to return.
      - filters (object, optional): metadata filters — any of repo, path,
        heading_path, content_type, license_family.
      - explain (bool, default false): include a per-hit score_breakdown
        (bm25 / semantic / fusion / rerank / final_rank).

    OUTPUT: {query, filters, count, degraded, warnings: [{code, message, stage}],
    hits: [{id, score, title, repo, path, heading_path, content_type,
    license_family, source_url, snippet, highlights, match_reason,
    score_breakdown?}]}. `hits` is ordered best-first and may be empty.

    EDGE CASES & FAILURES: An empty `hits` list with no error means nothing
    matched. `degraded=true` with a warning means one retrieval backend was
    unavailable and results came from the other. On failure a structured error is
    returned instead: errorCategory="validation" (empty query, limit out of
    range, non-object filters) is not retryable; "transient" (a data service was
    unreachable) is retryable. Stack traces are never returned.
    """
    return await hybrid_search_impl(
        query, limit=limit, filters=filters, explain=explain, service=get_retrieval_service()
    )


@mcp.tool()
async def get_chunk(chunk_id: str) -> dict[str, Any]:
    """Fetch a single indexed chunk by its id, with full provenance metadata.

    WHAT IT DOES: Looks up one chunk by its deterministic id and returns its text
    plus the provenance fields stored on every chunk (repo, path, title,
    heading_path, content_type, license_family, source_url).

    WHEN TO USE: To re-read the full text of a chunk returned by `hybrid_search`,
    or to resolve a chunk id you already hold.

    WHEN NOT TO USE: To search by content (use `hybrid_search`).

    INPUTS:
      - chunk_id (str, required): the chunk's id.

    OUTPUT: {chunk: {id, text, source_url, repo, path, title, heading_path,
    content_type, license_family}}.

    EDGE CASES & FAILURES: An unknown id returns a structured "business" error
    (not retryable). A non-string/empty id returns a "validation" error. Backend
    unavailability returns a retryable "transient" error. Stack traces are never
    returned.
    """
    return await get_chunk_impl(chunk_id, lexical_repository=get_lexical_repository())


@mcp.tool()
async def rerank(query: str, chunk_ids: list[str]) -> dict[str, Any]:
    """Re-score a fixed set of chunks against a query with the cross-encoder.

    WHAT IT DOES: Fetches the named chunks and re-scores them against the query
    with the configured TEI cross-encoder reranker, returning them ordered
    best-first with a `rerank_score` on each.

    WHEN TO USE: To re-order a candidate set you assembled yourself (e.g. merged
    results from several `hybrid_search` calls) by precise relevance to a focused
    query.

    WHEN NOT TO USE: For first-pass retrieval (use `hybrid_search`, which already
    fuses and can rerank). If no reranker is configured this tool returns a
    business error rather than guessing an order.

    INPUTS:
      - query (str, required): the query to score chunks against.
      - chunk_ids (list[str], required, 1..100): the chunk ids to re-score.

    OUTPUT: {query, count, missing: [chunk_id], chunks: [{id, text, source_url,
    repo, path, title, heading_path, content_type, license_family, score,
    rerank_score}]}. `missing` lists ids that were not found and skipped.

    EDGE CASES & FAILURES: If the reranker is not configured, or none of the
    chunk_ids exist, a non-retryable "business" error is returned. Empty
    chunk_ids or an empty query returns a "validation" error. Reranker/back-end
    unavailability returns a retryable "transient" error. Stack traces are never
    returned.
    """
    return await rerank_impl(
        query,
        chunk_ids,
        lexical_repository=get_lexical_repository(),
        reranker_client=get_reranker_client(),
    )


@mcp.tool()
async def list_sources() -> dict[str, Any]:
    """List the distinct indexed sources (repositories) and their chunk counts.

    WHAT IT DOES: Returns the repositories currently represented in the index,
    each with how many chunks it contributes.

    WHEN TO USE: During planning, to discover which sources exist before scoping
    a `hybrid_search` with a `repo` filter.

    WHEN NOT TO USE: To search content (use `hybrid_search`).

    INPUTS: none.

    OUTPUT: {count, sources: [{repo, chunk_count}]}. `sources` may be empty if
    nothing has been ingested yet.

    EDGE CASES & FAILURES: An empty corpus returns count=0 with an empty list (not
    an error). Backend unavailability returns a retryable "transient" error.
    Stack traces are never returned.
    """
    return await list_sources_impl(lexical_repository=get_lexical_repository())


def main() -> None:
    """Run the FastMCP server on the configured transport."""
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    mcp.run(transport="streamable-http" if transport == "http" else "stdio")


if __name__ == "__main__":
    main()

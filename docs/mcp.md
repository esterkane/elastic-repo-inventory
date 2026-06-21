# MCP server — agent access

`elastic-repo-inventory` exposes its retrieval core as an [MCP](https://modelcontextprotocol.io)
server, so any MCP client (a LangGraph agent, Claude Code, Cursor) can search the
indexed Elastic documentation as a set of tools. The server is a thin, **read-only**
adapter over the same functions the FastAPI app uses — it adds no business logic.

- Tool logic: `backend/app/mcp/tools.py` (plain async functions, unit-tested with fakes)
- Server wiring: `backend/app/mcp/server.py` (FastMCP; supplies cached singletons)
- Structured errors: `backend/app/mcp/errors.py`
- Tests: `backend/tests/test_mcp_tools.py`

## Tools

| Tool | Signature | What it does |
|------|-----------|--------------|
| `hybrid_search` | `(query: str, limit=10, filters?: object, explain=False)` | Lexical (Postgres FTS) + dense (Qdrant) retrieval fused with RRF, with metadata boosts and source de-dup. Returns the **same hit shape as `POST /api/v1/search`**. |
| `get_chunk` | `(chunk_id: str)` | Fetch one chunk by id with full provenance (repo, path, heading, content_type, license, source_url). |
| `rerank` | `(query: str, chunk_ids: list[str])` | Re-score a fixed candidate set with the TEI cross-encoder; returns chunks ordered best-first with a `rerank_score`. |
| `list_sources` | `()` | The catalog of indexed repositories with per-repo chunk counts. |

`filters` accepts any of: `repo`, `path`, `heading_path`, `content_type`, `license_family`.

**Read-only by design.** Ingestion / admin (`POST /api/v1/ingest/repo`) is intentionally
**not** exposed as a tool. If a mutating tool is ever added, gate it behind an env flag
`MCP_ALLOW_MUTATIONS` defaulting to `false`.

### Error contract

A tool never raises or leaks a stack trace. On failure it returns:

```json
{ "isError": true, "errorCategory": "validation|transient|business", "isRetryable": false, "message": "…", "details": {} }
```

- `validation` — bad input (empty query, `limit` out of 1..50, non-object `filters`). Not retryable.
- `business` — valid request that can't be satisfied (unknown `chunk_id`, no reranker configured). Not retryable.
- `transient` — a data service (Postgres / Qdrant / TEI) was momentarily unreachable. Retryable.

## Running the server

The server reuses the same env vars as the API (`DATABASE_URL`, `QDRANT_URL`,
`TEI_EMBED_URL`, optional `TEI_RERANK_URL` — see [CLAUDE.md](../CLAUDE.md)). Transport
is chosen by `MCP_TRANSPORT`:

```powershell
# stdio (default) — for Claude Code / Cursor / local agents
python -m backend.app.mcp.server

# streamable-HTTP — for a remote LangGraph client
$env:MCP_TRANSPORT = "http"; $env:MCP_HTTP_PORT = "8765"
python -m backend.app.mcp.server
```

## Registering with a client

**Claude Code / Cursor (`.mcp.json` or client config):**

```json
{
  "mcpServers": {
    "elastic-repo-inventory": {
      "command": "python",
      "args": ["-m", "backend.app.mcp.server"],
      "env": {
        "DATABASE_URL": "postgresql+asyncpg://repo_inventory:repo_inventory@localhost:5432/repo_inventory",
        "QDRANT_URL": "http://localhost:6333",
        "TEI_EMBED_URL": "http://localhost:8080/embed",
        "TEI_RERANK_URL": "http://localhost:8081/rerank"
      }
    }
  }
}
```

**LangGraph (streamable-HTTP)** — start the server with `MCP_TRANSPORT=http`, then point
`langchain-mcp-adapters` at `http://localhost:8765/mcp`.

## Example calls

```jsonc
// hybrid_search
{ "query": "what changed in ES|QL lookup joins", "limit": 5, "filters": { "repo": "elastic/docs-content" } }
// -> { "query": "...", "count": 5, "degraded": false, "warnings": [], "hits": [ { "id": "...", "score": 0.0123, "title": "...", "source_url": "https://www.elastic.co/docs/...", "snippet": "...", "score_breakdown": null }, ... ] }

// get_chunk
{ "chunk_id": "a1b2c3…" }
// -> { "chunk": { "id": "a1b2c3…", "text": "…", "source_url": "…", "repo": "elastic/docs-content", "path": "…", "heading_path": "…", "content_type": "release_note", "license_family": "…" } }

// rerank
{ "query": "vector search memory", "chunk_ids": ["id-1", "id-2", "id-3"] }
// -> { "query": "…", "count": 3, "missing": [], "chunks": [ { "id": "id-2", "rerank_score": 0.91, "score": 0.91, ... }, ... ] }

// list_sources
{}
// -> { "count": 3, "sources": [ { "repo": "elastic/docs-content", "chunk_count": 1280 }, ... ] }
```

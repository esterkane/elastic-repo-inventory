"""MCP layer for elastic-repo-inventory.

Exposes the existing retrieval core (hybrid search, chunk lookup, rerank, and the
indexed-source catalog) as MCP tools that any MCP client (a LangGraph agent,
Claude Code, Cursor) can call. The tool *logic* lives in :mod:`app.mcp.tools` as
plain async functions over the existing repositories — no business logic is added
here. Only read-only capabilities are exposed; ingestion/admin is deliberately
not surfaced as a tool.
"""

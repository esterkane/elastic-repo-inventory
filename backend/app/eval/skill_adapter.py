"""Thin adapter: expose the async hybrid ``RetrievalService`` to the injected
``search_fn`` contract of the reusable ``relevance_eval`` skill.

The skill (see ``[eval]`` extra in ``pyproject.toml``) is backend-agnostic: it
calls ``search_fn(query, strategy) -> [doc_id]``. This module adapts *our*
existing retrieval — nothing more. It contains no retrieval logic of its own: it
runs ``RetrievalService.retrieve`` (which already does Postgres FTS + Qdrant RRF
fusion) and extracts the ranked ``RankedHit.id`` chunk ids.

This repo has a single hybrid pipeline, so there is exactly one strategy today:
``"hybrid"``. Lexical-only / vector-only could become future strategy variants,
but they are not separate code paths today, so we do not pretend they exist.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from backend.app.retrieval.service import RankedHit

HYBRID_STRATEGY = "hybrid"
STRATEGIES: tuple[str, ...] = (HYBRID_STRATEGY,)


class SupportsRetrieve(Protocol):
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        boosts: dict | None = None,
    ) -> dict[str, Any]: ...


def make_search_fn(service: SupportsRetrieve, *, limit: int = 10):
    """Return a SYNC ``search(query, strategy) -> [chunk_id]`` for the skill.

    Wraps the async ``service.retrieve`` with ``asyncio.run`` so the skill's
    synchronous harness can drive it. ``strategy`` must be ``"hybrid"`` (the only
    pipeline this repo has); anything else is rejected rather than silently
    treated as hybrid.
    """

    def search(query: str, strategy: str) -> list[str]:
        if strategy != HYBRID_STRATEGY:
            raise ValueError(
                f"Unknown strategy {strategy!r}; this repo only has {HYBRID_STRATEGY!r}."
            )
        result = asyncio.run(service.retrieve(query, limit=limit))
        return _chunk_ids(result.get("hits", []))

    return search


def _chunk_ids(hits: Sequence[object]) -> list[str]:
    return [hit.id for hit in hits if isinstance(hit, RankedHit)]

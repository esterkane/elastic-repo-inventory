"""Agent-facing tools — thin wrappers over the existing retrieval core.

Agents obtain evidence ONLY through these tools. There is no parallel retrieval
or fusion logic here: :func:`retrieve_tool` calls the existing async
``RetrievalService.retrieve`` (Postgres FTS + Qdrant RRF, with the reliability
contract already built in) and :func:`rerank_tool` calls the existing
``RerankerClient.rerank`` (TEI cross-encoder). Both keep ``source_url`` intact on
every hit and surface the retrieval ``warnings`` / ``degraded`` flag so callers
can honour graceful degradation.
"""

from __future__ import annotations

from typing import Any, Protocol

from backend.app.retrieval.service import RankedHit, RetrievalWarning


class SupportsRetrieve(Protocol):
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        boosts: dict | None = None,
    ) -> dict[str, Any]: ...


class SupportsRerank(Protocol):
    async def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]: ...


class RetrievalToolResult:
    """Result of a retrieval tool call: retained hits + reliability signals."""

    __slots__ = ("hits", "warnings", "degraded", "reranked")

    def __init__(
        self,
        hits: list[RankedHit],
        warnings: list[RetrievalWarning],
        degraded: bool,
        reranked: bool = False,
    ) -> None:
        self.hits = hits
        self.warnings = warnings
        self.degraded = degraded
        self.reranked = reranked


def _coerce_warnings(raw: object) -> list[RetrievalWarning]:
    warnings: list[RetrievalWarning] = []
    if not isinstance(raw, list):
        return warnings
    for warning in raw:
        if isinstance(warning, RetrievalWarning):
            warnings.append(warning)
        elif isinstance(warning, dict):
            warnings.append(
                RetrievalWarning(
                    code=str(warning.get("code") or "warning"),
                    message=str(warning.get("message") or "Retrieval warning."),
                    stage=str(warning.get("stage") or "retrieval"),
                )
            )
    return warnings


async def retrieve_tool(
    service: SupportsRetrieve,
    query: str,
    *,
    limit: int = 10,
    filters: dict | None = None,
    boosts: dict | None = None,
) -> RetrievalToolResult:
    """Hybrid retrieval for a (sub-)question.

    Wraps ``RetrievalService.retrieve`` verbatim. Returns the retained
    ``RankedHit``s (each carrying ``source_url``) plus the warnings and
    ``degraded`` flag the service produced — the reliability contract is
    propagated, not reinvented.
    """
    result = await service.retrieve(query, limit=limit, filters=filters, boosts=boosts)
    raw_hits = result.get("hits", [])
    hits = [hit for hit in raw_hits if isinstance(hit, RankedHit)]
    return RetrievalToolResult(
        hits=hits,
        warnings=_coerce_warnings(result.get("warnings", [])),
        degraded=bool(result.get("degraded", False)),
    )


async def rerank_tool(
    reranker: SupportsRerank | None,
    query: str,
    result: RetrievalToolResult,
) -> RetrievalToolResult:
    """Re-score retained hits with the existing TEI reranker.

    Honours the reliability contract: if no reranker is configured, or the
    rerank stage fails, the fused hits are returned unchanged with a
    ``reranker_unavailable`` warning appended and ``degraded`` set — never an
    exception.
    """
    if reranker is None or not result.hits:
        return result
    try:
        reranked = await reranker.rerank(query, result.hits)
    except Exception:  # noqa: BLE001 — degrade gracefully, mirror the service contract
        warning = RetrievalWarning(
            code="reranker_unavailable",
            message="Reranker unavailable; ranking uses hybrid fusion.",
            stage="reranking",
        )
        return RetrievalToolResult(
            hits=result.hits,
            warnings=[*result.warnings, warning],
            degraded=True,
            reranked=False,
        )
    return RetrievalToolResult(
        hits=reranked,
        warnings=result.warnings,
        degraded=result.degraded,
        reranked=True,
    )

"""Lazily-constructed resource singletons for the MCP layer.

The MCP server is a sibling front-end to the FastAPI app: both adapt the same
retrieval core to a transport. These accessors reuse the API's cached
``build_retrieval_service`` wiring, so the lexical repository, vector repository,
embedding client, and reranker are constructed exactly once and shared.
"""

from __future__ import annotations

from backend.app.dependencies import build_retrieval_service
from backend.app.retrieval.service import (
    PostgresFTSRepository,
    RerankerClient,
    RetrievalService,
)


def get_retrieval_service() -> RetrievalService:
    return build_retrieval_service()


def get_lexical_repository() -> PostgresFTSRepository:
    # The default wiring always uses PostgresFTSRepository for the lexical side.
    return build_retrieval_service().lexical_repository  # type: ignore[return-value]


def get_reranker_client() -> RerankerClient | None:
    return build_retrieval_service().reranker_client

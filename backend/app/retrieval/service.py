from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, TypeVar
from urllib.parse import urldefrag

import httpx
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.app.embeddings.client import EmbeddingClient
from backend.app.ingest.indexer import ACTIVE_REPOSITORY_SLUGS
from backend.app.ingest.metadata import metadata_boost_score, normalize_boosts, normalize_filters, normalize_metadata
from backend.app.vector.qdrant_client import SearchHit, VectorPayload, VectorRepository

logger = logging.getLogger(__name__)
T = TypeVar("T")


RECOMMENDATION_CATEGORIES: tuple[str, ...] = (
    "relevance",
    "ingestion",
    "mapping",
    "performance",
    "resiliency",
)


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 2
    backoff_seconds: float = 0.2


@dataclass(frozen=True)
class RetrievalWarning:
    code: str
    message: str
    stage: str


@dataclass(frozen=True)
class RetrievalTelemetryEvent:
    stage: str
    status: str
    attempts: int
    duration_ms: float
    error: str | None = None


TelemetryHook = Callable[[RetrievalTelemetryEvent], None]


@dataclass(frozen=True)
class RankedHit:
    id: str
    score: float
    metadata: VectorPayload
    source_url: str
    text: str = ""
    lexical_score: float = 0.0
    dense_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float | None = None


class LexicalRepository(Protocol):
    async def search(self, query: str, limit: int, filters: dict | None = None) -> list[RankedHit]: ...


class PostgresFTSRepository:
    def __init__(self, engine: AsyncEngine, table_name: str = "document_chunks") -> None:
        self.engine = engine
        self.table_name = table_name

    async def search(self, query: str, limit: int, filters: dict | None = None) -> list[RankedHit]:
        where, params = postgres_filter_clause(filters or {})
        params.update({"query": query, "limit": limit})
        statement = text(
            f"""
            SELECT id, content, metadata, source_url, repo, path,
                   ts_rank_cd(search_vector, websearch_to_tsquery('english', :query)) AS score
            FROM {self.table_name}
            WHERE search_vector @@ websearch_to_tsquery('english', :query)
            {where}
            ORDER BY score DESC
            LIMIT :limit
            """
        )
        try:
            async with self.engine.connect() as connection:
                result = await connection.execute(statement, params)
        except SQLAlchemyError as exc:
            if is_missing_relation(exc):
                return []
            raise

        hits: list[RankedHit] = []
        for row in result.mappings():
            metadata = normalize_metadata(
                dict(row["metadata"] or {}),
                source_url=str(row["source_url"]),
                repo=str(row["repo"] or ""),
                path=str(row["path"] or ""),
            )
            score = float(row["score"] or 0.0)
            hits.append(
                RankedHit(
                    id=str(row["id"]),
                    score=score,
                    lexical_score=score,
                    metadata=metadata,
                    source_url=str(row["source_url"]),
                    text=str(row["content"] or ""),
                )
            )
        return hits

    async def hydrate(self, hits: list[RankedHit]) -> list[RankedHit]:
        if not hits:
            return hits
        ids = [hit.id for hit in hits]
        statement = text(
            f"""
            SELECT id, content, metadata, source_url, repo, path
            FROM {self.table_name}
            WHERE id = ANY(:ids)
            """
        )
        try:
            async with self.engine.connect() as connection:
                result = await connection.execute(statement, {"ids": ids})
        except SQLAlchemyError as exc:
            if is_missing_relation(exc):
                return hits
            raise

        rows = {str(row["id"]): row for row in result.mappings()}
        hydrated: list[RankedHit] = []
        for hit in hits:
            row = rows.get(hit.id)
            if not row:
                hydrated.append(hit)
                continue
            metadata = normalize_metadata(
                dict(row["metadata"] or hit.metadata or {}),
                source_url=str(row["source_url"] or hit.source_url),
                repo=str(row["repo"] or hit.metadata.get("repo") or ""),
                path=str(row["path"] or hit.metadata.get("path") or ""),
            )
            hydrated.append(
                RankedHit(
                    id=hit.id,
                    score=hit.score,
                    metadata=metadata,
                    source_url=str(row["source_url"] or hit.source_url),
                    text=str(row["content"] or hit.text or ""),
                    lexical_score=hit.lexical_score,
                    dense_score=hit.dense_score,
                    fusion_score=hit.fusion_score,
                    rerank_score=hit.rerank_score,
                )
            )
        return hydrated

    async def get(self, chunk_id: str) -> RankedHit | None:
        """Fetch a single chunk by its id, or None if it does not exist.

        Mirrors :meth:`hydrate`'s projection so callers (the API and the MCP
        ``get_chunk`` / ``rerank`` tools) see the same chunk shape.
        """
        statement = text(
            f"""
            SELECT id, content, metadata, source_url, repo, path
            FROM {self.table_name}
            WHERE id = :chunk_id
            LIMIT 1
            """
        )
        try:
            async with self.engine.connect() as connection:
                result = await connection.execute(statement, {"chunk_id": chunk_id})
        except SQLAlchemyError as exc:
            if is_missing_relation(exc):
                return None
            raise
        row = result.mappings().first()
        if row is None:
            return None
        metadata = normalize_metadata(
            dict(row["metadata"] or {}),
            source_url=str(row["source_url"]),
            repo=str(row["repo"] or ""),
            path=str(row["path"] or ""),
        )
        return RankedHit(
            id=str(row["id"]),
            score=0.0,
            metadata=metadata,
            source_url=str(row["source_url"]),
            text=str(row["content"] or ""),
        )

    async def list_sources(self) -> list[dict[str, object]]:
        """Return the distinct indexed sources (repos) with their chunk counts."""
        statement = text(
            f"""
            SELECT repo, COUNT(*) AS chunk_count
            FROM {self.table_name}
            GROUP BY repo
            ORDER BY repo
            """
        )
        try:
            async with self.engine.connect() as connection:
                result = await connection.execute(statement)
        except SQLAlchemyError as exc:
            if is_missing_relation(exc):
                return []
            raise
        return [
            {"repo": str(row["repo"] or ""), "chunk_count": int(row["chunk_count"] or 0)}
            for row in result.mappings()
        ]


class RerankerClient:
    def __init__(self, endpoint_url: str, model: str | None = None, timeout: float = 30.0) -> None:
        self.endpoint_url = endpoint_url
        self.model = model
        self.timeout = timeout

    async def rerank(self, query: str, hits: list[RankedHit]) -> list[RankedHit]:
        if not hits:
            return []

        payload: dict[str, object] = {
            "query": query,
            "texts": [hit.text or metadata_text(hit.metadata) for hit in hits],
        }
        if self.model:
            payload["model"] = self.model

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.endpoint_url, json=payload)
            response.raise_for_status()
            data = response.json()

        scores = parse_rerank_scores(data, expected_count=len(hits))
        reranked = [
            RankedHit(
                id=hit.id,
                score=float(score),
                metadata=hit.metadata,
                source_url=hit.source_url,
                text=hit.text,
                lexical_score=hit.lexical_score,
                dense_score=hit.dense_score,
                fusion_score=hit.fusion_score,
                rerank_score=float(score),
            )
            for hit, score in zip(hits, scores, strict=True)
        ]
        return sorted(reranked, key=lambda hit: (-hit.score, hit.id))


class RetrievalService:
    def __init__(
        self,
        lexical_repository: LexicalRepository,
        vector_repository: VectorRepository,
        embedding_client: EmbeddingClient,
        reranker_client: RerankerClient | None = None,
        default_repos: tuple[str, ...] = ACTIVE_REPOSITORY_SLUGS,
        retry_policy: RetryPolicy | None = None,
        telemetry_hook: TelemetryHook | None = None,
    ) -> None:
        self.lexical_repository = lexical_repository
        self.vector_repository = vector_repository
        self.embedding_client = embedding_client
        self.reranker_client = reranker_client
        self.default_repos = default_repos
        self.retry_policy = retry_policy or RetryPolicy()
        self.telemetry_hook = telemetry_hook

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: dict | None = None,
        boosts: dict[str, dict[str, object]] | None = None,
    ) -> dict[str, object]:
        effective_filters = active_repo_filters(filters, self.default_repos)
        effective_boosts = normalize_boosts(boosts)
        warnings: list[RetrievalWarning] = []
        telemetry: list[RetrievalTelemetryEvent] = []

        lexical_hits: list[RankedHit] = []
        dense_hits: list[RankedHit] = []
        lexical_failed = False
        dense_failed = False

        try:
            lexical_hits = await retryable_stage(
                "lexical_retrieval",
                lambda: self.lexical_repository.search(query, limit=50, filters=effective_filters),
                self.retry_policy,
                telemetry,
                self.telemetry_hook,
            )
        except Exception:
            lexical_failed = True
            logger.warning("lexical retrieval unavailable", exc_info=True)
            warnings.append(
                RetrievalWarning(
                    code="lexical_unavailable",
                    message="Lexical retrieval unavailable; showing semantic results when available.",
                    stage="lexical_retrieval",
                )
            )

        try:
            dense_hits = await retryable_stage(
                "semantic_retrieval",
                lambda: dense_retrieval(self.embedding_client, self.vector_repository, query, effective_filters),
                self.retry_policy,
                telemetry,
                self.telemetry_hook,
            )
        except Exception:
            dense_failed = True
            logger.warning("semantic retrieval unavailable", exc_info=True)
            warnings.append(
                RetrievalWarning(
                    code="semantic_unavailable",
                    message="Semantic retrieval unavailable; showing lexical results when available.",
                    stage="semantic_retrieval",
                )
            )

        if dense_hits and hasattr(self.lexical_repository, "hydrate"):
            try:
                dense_hits = await retryable_stage(
                    "content_hydration",
                    lambda: self.lexical_repository.hydrate(dense_hits),  # type: ignore[attr-defined]
                    self.retry_policy,
                    telemetry,
                    self.telemetry_hook,
                )
            except Exception:
                logger.warning("dense hit hydration unavailable", exc_info=True)
                warnings.append(
                    RetrievalWarning(
                        code="content_hydration_unavailable",
                        message="Semantic results were returned without full chunk text; evidence excerpts may be limited.",
                        stage="content_hydration",
                    )
                )

        fused = boost_ranked_hits(reciprocal_rank_fusion(lexical_hits, dense_hits), effective_boosts)[:20]
        ranked = fused
        if self.reranker_client and fused:
            try:
                ranked = await retryable_stage(
                    "reranking",
                    lambda: self.reranker_client.rerank(query, fused),
                    self.retry_policy,
                    telemetry,
                    self.telemetry_hook,
                )
            except Exception:
                logger.warning("reranker unavailable", exc_info=True)
                warnings.append(
                    RetrievalWarning(
                        code="reranker_unavailable",
                        message="Reranker unavailable; ranking uses hybrid fusion.",
                        stage="reranking",
                    )
                )

        if lexical_failed and dense_failed:
            warnings.append(
                RetrievalWarning(
                    code="retrieval_unavailable",
                    message="No retrieval backend returned evidence for this request.",
                    stage="retrieval",
                )
            )

        hits = with_final_ranks(diversify_hits(ranked, limit))
        return {
            "hits": hits,
            "recommendation_categories": list(RECOMMENDATION_CATEGORIES),
            "warnings": warnings,
            "degraded": bool(warnings),
            "telemetry": telemetry,
        }


async def dense_retrieval(
    embedding_client: EmbeddingClient,
    vector_repository: VectorRepository,
    query: str,
    filters: dict[str, object] | None,
) -> list[RankedHit]:
    vectors = await embedding_client.embed([query])
    return [
        vector_hit_to_ranked_hit(hit)
        for hit in await vector_repository.search(vectors[0], limit=50, filters=filters)
    ]


async def retryable_stage(
    stage: str,
    operation: Callable[[], Awaitable[T]],
    retry_policy: RetryPolicy,
    telemetry: list[RetrievalTelemetryEvent],
    telemetry_hook: TelemetryHook | None = None,
) -> T:
    attempts = max(1, retry_policy.attempts)
    started = time.perf_counter()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = await operation()
            event = RetrievalTelemetryEvent(
                stage=stage,
                status="success",
                attempts=attempt,
                duration_ms=elapsed_ms(started),
            )
            record_telemetry(event, telemetry, telemetry_hook)
            return result
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                await asyncio.sleep(retry_policy.backoff_seconds * attempt)

    event = RetrievalTelemetryEvent(
        stage=stage,
        status="failed",
        attempts=attempts,
        duration_ms=elapsed_ms(started),
        error=last_error.__class__.__name__ if last_error else "unknown",
    )
    record_telemetry(event, telemetry, telemetry_hook)
    if last_error:
        raise last_error
    raise RuntimeError(f"{stage} failed")


def record_telemetry(
    event: RetrievalTelemetryEvent,
    telemetry: list[RetrievalTelemetryEvent],
    telemetry_hook: TelemetryHook | None,
) -> None:
    telemetry.append(event)
    logger.info(
        "retrieval stage completed",
        extra={
            "stage": event.stage,
            "status": event.status,
            "attempts": event.attempts,
            "duration_ms": event.duration_ms,
            "error": event.error,
        },
    )
    if telemetry_hook:
        telemetry_hook(event)


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def vector_hit_to_ranked_hit(hit: SearchHit) -> RankedHit:
    metadata = normalize_metadata(hit.metadata, source_url=hit.source_url)
    return RankedHit(
        id=hit.id,
        score=hit.score,
        dense_score=hit.score,
        metadata=metadata,
        source_url=hit.source_url,
        text=metadata_text(metadata),
    )


def reciprocal_rank_fusion(
    lexical_hits: list[RankedHit],
    dense_hits: list[RankedHit],
    k: int = 60,
) -> list[RankedHit]:
    by_id: dict[str, RankedHit] = {}
    scores: dict[str, float] = {}
    lexical_scores: dict[str, float] = {}
    dense_scores: dict[str, float] = {}

    for hits, channel in ((lexical_hits, "lexical"), (dense_hits, "dense")):
        for rank, hit in enumerate(hits, start=1):
            by_id.setdefault(hit.id, hit)
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            if channel == "lexical":
                lexical_scores[hit.id] = max(lexical_scores.get(hit.id, 0.0), hit.score)
            else:
                dense_scores[hit.id] = max(dense_scores.get(hit.id, 0.0), hit.score)

    fused: list[RankedHit] = []
    for hit_id, score in scores.items():
        hit = by_id[hit_id]
        fused.append(
            RankedHit(
                id=hit.id,
                score=score,
                metadata=hit.metadata,
                source_url=hit.source_url,
                text=hit.text,
                lexical_score=lexical_scores.get(hit.id, 0.0),
                dense_score=dense_scores.get(hit.id, 0.0),
                fusion_score=score,
            )
        )

    return sorted(fused, key=lambda hit: (-hit.score, hit.id))


def parse_rerank_scores(data: object, expected_count: int) -> list[float]:
    scores: object
    if isinstance(data, dict):
        scores = data.get("scores", data.get("results", data.get("data")))
    else:
        scores = data

    if isinstance(scores, list) and scores and isinstance(scores[0], dict):
        scores = [item.get("score") for item in scores]

    if not isinstance(scores, list) or len(scores) != expected_count:
        raise ValueError("Reranker endpoint returned an unexpected number of scores")
    if not all(isinstance(score, int | float) for score in scores):
        raise ValueError("Reranker endpoint returned a malformed score")
    return [float(score) for score in scores]


def metadata_text(metadata: VectorPayload) -> str:
    parts = [
        metadata.get("title"),
        metadata.get("heading_path"),
        metadata.get("content_type"),
    ]
    return " ".join(str(part) for part in parts if part)


def postgres_filter_clause(filters: dict) -> tuple[str, dict[str, object]]:
    normalized_filters = normalize_filters(filters)
    if not normalized_filters:
        return "", {}

    clauses: list[str] = []
    params: dict[str, object] = {}
    for index, (key, value) in enumerate(sorted(normalized_filters.items())):
        if value is None:
            continue
        key_param = f"filter_key_{index}"
        params[key_param] = str(key)
        expression = postgres_metadata_expression(str(key), key_param)
        if isinstance(value, list | tuple | set):
            value_params: list[str] = []
            for value_index, item in enumerate(value):
                value_param = f"filter_value_{index}_{value_index}"
                value_params.append(f":{value_param}")
                params[value_param] = str(item)
            if not value_params:
                continue
            clauses.append(f"AND {expression} IN ({', '.join(value_params)})")
        else:
            value_param = f"filter_value_{index}"
            clauses.append(f"AND {expression} = :{value_param}")
            params[value_param] = str(value)
    return ("\n".join(clauses), params)


def active_repo_filters(filters: dict | None, default_repos: tuple[str, ...]) -> dict[str, object] | None:
    merged: dict[str, object] = dict(normalize_filters(filters) or {})
    if default_repos and not merged.get("repo"):
        merged["repo"] = list(default_repos)
    return merged or None


def postgres_metadata_expression(key: str, key_param: str) -> str:
    if key == "repo":
        return f"COALESCE(metadata ->> :{key_param}, repo)"
    if key == "path":
        return f"COALESCE(metadata ->> :{key_param}, path)"
    return f"metadata ->> :{key_param}"


def boost_ranked_hits(hits: list[RankedHit], boosts: dict[str, dict[str, float]]) -> list[RankedHit]:
    boosted: list[RankedHit] = []
    for hit in hits:
        boost = metadata_boost_score(hit.metadata, boosts)
        score = hit.score * (1.0 + boost)
        if is_archived_source(hit.metadata, hit.source_url):
            score *= 0.05
        boosted.append(
            RankedHit(
                id=hit.id,
                score=score,
                metadata=hit.metadata,
                source_url=hit.source_url,
                text=hit.text,
                lexical_score=hit.lexical_score,
                dense_score=hit.dense_score,
                fusion_score=score,
                rerank_score=hit.rerank_score,
            )
        )
    return sorted(boosted, key=lambda hit: (-hit.score, hit.id))


def is_archived_source(metadata: VectorPayload | dict[str, object], source_url: str = "") -> bool:
    repo = str(metadata.get("repo") or "").lower()
    path = str(metadata.get("path") or "").replace("\\", "/").strip("/").lower()
    heading = str(metadata.get("heading_path") or metadata.get("title") or "").lower()
    url = source_url.lower()
    if repo == "elastic/docs-content" and (path == "archive.md" or path.startswith("archive/")):
        return True
    if "/docs/archive" in url or "/archive.md" in url:
        return True
    return "documentation archive" in heading


def diversify_hits(hits: list[RankedHit], limit: int) -> list[RankedHit]:
    selected: list[RankedHit] = []
    seen_pages: set[str] = set()
    seen_titles: set[str] = set()

    for hit in hits:
        page_key = canonical_source_key(hit.source_url)
        title_key = str(hit.metadata.get("title") or hit.metadata.get("heading_path") or "").strip().lower()
        if page_key and page_key in seen_pages:
            continue
        if title_key and title_key in seen_titles:
            continue
        selected.append(hit)
        if page_key:
            seen_pages.add(page_key)
        if title_key:
            seen_titles.add(title_key)
        if len(selected) >= limit:
            return selected

    for hit in hits:
        if hit.id not in {selected_hit.id for selected_hit in selected}:
            selected.append(hit)
        if len(selected) >= limit:
            break
    return selected


def canonical_source_key(source_url: str) -> str:
    return urldefrag(source_url)[0].rstrip("/")


def with_final_ranks(hits: list[RankedHit]) -> list[RankedHit]:
    ranked: list[RankedHit] = []
    for index, hit in enumerate(hits, start=1):
        metadata = dict(hit.metadata)
        metadata["final_rank"] = index
        ranked.append(
            RankedHit(
                id=hit.id,
                score=hit.score,
                metadata=metadata,
                source_url=hit.source_url,
                text=hit.text,
                lexical_score=hit.lexical_score,
                dense_score=hit.dense_score,
                fusion_score=hit.fusion_score,
                rerank_score=hit.rerank_score,
            )
        )
    return ranked


def is_missing_relation(exc: SQLAlchemyError) -> bool:
    message = str(exc).lower()
    return "undefinedtable" in message or "does not exist" in message or "undefined table" in message

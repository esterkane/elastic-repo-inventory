"""Structured tool errors for the MCP layer.

MCP tools must never leak an internal stack trace or a raw Postgres / HTTP error
into a result. Every failure is converted to a small, structured payload::

    {
        "isError": True,
        "errorCategory": "validation" | "transient" | "permission" | "business",
        "isRetryable": bool,
        "message": "<safe, human-readable summary>",
        "details": { ... },   # optional, safe context only
    }

Handlers raise one of the typed errors below for expected failures; the
:func:`guard` decorator wraps every async tool handler so that *any* unexpected
exception is logged server-side (with its traceback) and returned as a generic,
trace-free error.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# Error categories.
VALIDATION = "validation"
TRANSIENT = "transient"
PERMISSION = "permission"
BUSINESS = "business"


class ToolError(Exception):
    """An expected, classified tool failure carrying a category and retryability."""

    category: str = TRANSIENT
    retryable: bool = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ToolValidationError(ToolError):
    """Bad or unsupported input (empty query, out-of-range limit). Not retryable."""

    category = VALIDATION
    retryable = False


class ToolBusinessError(ToolError):
    """A valid request that cannot be satisfied (unknown chunk id, no reranker
    configured). Retrying without changing inputs will not help."""

    category = BUSINESS
    retryable = False


class ToolTransientError(ToolError):
    """A backend was momentarily unavailable. Safe to retry."""

    category = TRANSIENT
    retryable = True


def error_result(
    category: str,
    message: str,
    *,
    retryable: bool,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured error payload returned in place of a result."""
    return {
        "isError": True,
        "errorCategory": category,
        "isRetryable": retryable,
        "message": message,
        "details": details or {},
    }


def guard(name: str) -> Callable[[Callable[..., Awaitable[dict]]], Callable[..., Awaitable[dict]]]:
    """Wrap an async tool handler so no failure ever escapes as a stack trace.

    - :class:`ToolError` subclasses become their structured category payload.
    - Postgres (`SQLAlchemyError`) and HTTP (`httpx.HTTPError`) failures become a
      retryable transient error — the data services were momentarily unreachable.
    - Anything else is logged with its traceback and returned as a generic,
      non-retryable error with no internal detail.
    """

    def decorator(fn: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                return await fn(*args, **kwargs)
            except ToolError as exc:
                return error_result(
                    exc.category, exc.message, retryable=exc.retryable, details=exc.details
                )
            except (httpx.HTTPError, SQLAlchemyError) as exc:
                logger.warning(
                    "mcp tool %s: backend unavailable (%s)", name, type(exc).__name__
                )
                return error_result(
                    TRANSIENT,
                    "A backend service is currently unavailable. Please retry shortly.",
                    retryable=True,
                    details={"kind": type(exc).__name__},
                )
            except Exception:  # noqa: BLE001 - last-resort guard; traceback goes to logs only
                logger.exception("mcp tool %s failed unexpectedly", name)
                return error_result(
                    TRANSIENT,
                    "An unexpected internal error occurred while handling the request.",
                    retryable=False,
                )

        return wrapper

    return decorator

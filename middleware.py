"""
API security middleware.

Provides:
  - API key authentication via X-API-Key header
  - Rate limiting per IP address
  - Request ID injection for tracing

If FX_ENGINE_API_KEY is empty, authentication is disabled (dev mode).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from typing import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates X-API-Key header on all non-health endpoints.
    Skips auth if api_key is empty (dev mode).
    """

    # Endpoints that never require auth
    PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, api_key: str = ""):
        super().__init__(app)
        self.api_key = api_key
        self.auth_enabled = bool(api_key)
        if self.auth_enabled:
            logger.info("API key authentication enabled")
        else:
            logger.warning("API key authentication DISABLED — set FX_ENGINE_API_KEY for production")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Inject request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # Check API key if enabled
        if self.auth_enabled:
            provided_key = request.headers.get("X-API-Key", "")
            if provided_key != self.api_key:
                logger.warning(
                    f"Unauthorized request to {request.url.path} from {request.client.host}",
                    extra={"request_id": request_id},
                )
                raise HTTPException(status_code=401, detail="Invalid or missing API key")

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter per client IP.
    Allows `max_requests` per `window_seconds`.

    For production, replace with Redis-backed rate limiting.
    """

    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        # Clean old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip]
            if now - t < self.window_seconds
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s.",
            )

        self._requests[client_ip].append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(
            self.max_requests - len(self._requests[client_ip])
        )
        return response

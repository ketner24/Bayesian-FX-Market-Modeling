"""Tests for API middleware (auth and rate limiting)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fx_engine.middleware import APIKeyMiddleware, RateLimitMiddleware


def _make_app(api_key: str = "", rate_limit: int = 5) -> FastAPI:
    """Create a minimal FastAPI app with middleware for testing."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, max_requests=rate_limit, window_seconds=60)
    app.add_middleware(APIKeyMiddleware, api_key=api_key)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/protected")
    def protected():
        return {"data": "secret"}

    return app


class TestAPIKeyMiddleware:

    def test_no_auth_when_key_empty(self):
        client = TestClient(_make_app(api_key=""))
        resp = client.get("/protected")
        assert resp.status_code == 200

    def test_rejects_missing_key(self):
        client = TestClient(_make_app(api_key="my-secret"))
        resp = client.get("/protected")
        assert resp.status_code == 401

    def test_rejects_wrong_key(self):
        client = TestClient(_make_app(api_key="my-secret"))
        resp = client.get("/protected", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_accepts_correct_key(self):
        client = TestClient(_make_app(api_key="my-secret"))
        resp = client.get("/protected", headers={"X-API-Key": "my-secret"})
        assert resp.status_code == 200

    def test_health_always_public(self):
        client = TestClient(_make_app(api_key="my-secret"))
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_request_id_header(self):
        client = TestClient(_make_app())
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers


class TestRateLimitMiddleware:

    def test_allows_under_limit(self):
        client = TestClient(_make_app(rate_limit=10))
        for _ in range(5):
            resp = client.get("/protected")
            assert resp.status_code == 200

    def test_blocks_over_limit(self):
        client = TestClient(_make_app(rate_limit=3))
        for _ in range(3):
            client.get("/protected")
        resp = client.get("/protected")
        assert resp.status_code == 429

    def test_health_exempt_from_rate_limit(self):
        client = TestClient(_make_app(rate_limit=2))
        # Exhaust rate limit
        for _ in range(2):
            client.get("/protected")
        # Health should still work
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_rate_limit_header(self):
        client = TestClient(_make_app(rate_limit=10))
        resp = client.get("/protected")
        assert "X-RateLimit-Remaining" in resp.headers

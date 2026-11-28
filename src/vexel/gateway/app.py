"""The gateway's HTTP API (FastAPI). Four endpoints, nothing else:

- ``GET  /v1/health``: up, and which providers have keys (names only).
- ``POST /v1/auth/redeem`` ``{"invite": "VX-…", "label": "computer name"}``: a token for one
  person (one computer) of the invite's organisation. Rate limited per address.
- ``GET  /v1/quota`` (``Authorization: Bearer <token>``): requests, tokens and budget left.
- ``POST /v1/plan`` (bearer token) ``{"plan_id", "context", "image"?}``: one planning step
  (``vexel.gateway.service.plan``).

Requests over ``MAX_BODY`` bytes are refused before they are read. Run with
``vexel gateway serve`` behind a TLS proxy (see ``deploy/gateway``).
"""

# no `from __future__ import annotations`: FastAPI must see the real Request type in the handlers
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

MAX_BODY = 2_000_000


class _Limiter:
    """At most ``n`` events per ``window`` seconds per key (in memory)."""

    def __init__(self, n: int, window: float) -> None:
        self.n, self.window = n, window
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.n:
                return False
            q.append(now)
            return True


def create_app(store: Any = None) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from vexel import __version__
    from vexel.gateway import service
    from vexel.gateway.providers import DeepSeek, Gemini
    from vexel.gateway.store import Store

    store = store or Store()
    app = FastAPI(title="Vexel AI gateway", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    redeem_limit = _Limiter(10, 60.0)
    plan_limit = _Limiter(30, 60.0)

    @app.middleware("http")
    async def body_limit(request: Request, call_next: Any) -> Any:
        size = request.headers.get("content-length")
        if size is not None and size.isdigit() and int(size) > MAX_BODY:
            return JSONResponse({"error": f"request over {MAX_BODY // 1_000_000} MB"}, status_code=413)
        return await call_next(request)

    def _token(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        return auth[7:].strip() if auth.lower().startswith("bearer ") else ""

    def _client(request: Request) -> str:
        if os.environ.get("VEXEL_GATEWAY_TRUST_PROXY") == "1":  # behind our own TLS proxy only
            fwd = request.headers.get("x-forwarded-for", "")
            if fwd:
                return fwd.split(",")[0].strip()
        return request.client.host if request.client else "?"

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        mock = os.environ.get("VEXEL_GATEWAY_MOCK", "").strip() in ("1", "true")
        return {"ok": True, "version": __version__, "providers": ["mock"] if mock else [p.name for p in (DeepSeek(), Gemini()) if p.available()]}

    @app.post("/v1/auth/redeem")
    async def redeem(request: Request) -> Any:
        if not redeem_limit.allow(_client(request)):
            return JSONResponse({"error": "too many attempts, wait a minute"}, status_code=429)
        try:
            body = await request.json()
            token = store.redeem(str(body.get("invite", "")), str(body.get("label", "")))
        except PermissionError as e:
            return JSONResponse({"error": str(e)}, status_code=403)
        except (ValueError, AttributeError):
            return JSONResponse({"error": "send {\"invite\": ..., \"label\": ...}"}, status_code=400)
        user = store.user_for(token)
        return {"token": token, "org": user.org.name if user else ""}

    @app.get("/v1/quota")
    def quota(request: Request) -> Any:
        out = service.quota(store, _token(request))
        return JSONResponse(out.body, status_code=out.status)

    @app.post("/v1/plan")
    async def plan(request: Request) -> Any:
        token = _token(request)
        if not plan_limit.allow(token or _client(request)):
            return JSONResponse({"refused": "rate", "message": "too many requests, wait a minute"}, status_code=429)
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "body is not JSON"}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "body must be an object"}, status_code=400)
        import asyncio

        out = await asyncio.to_thread(service.plan, store, token, body)  # providers are slow: off the event loop
        return JSONResponse(out.body, status_code=out.status)

    return app


def serve(host: str = "127.0.0.1", port: int = 8780) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level="info", proxy_headers=os.environ.get("VEXEL_GATEWAY_TRUST_PROXY") == "1")


__all__ = ["MAX_BODY", "create_app", "serve"]

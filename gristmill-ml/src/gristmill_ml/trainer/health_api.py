"""FastAPI health and status API for gristmill-trainer (Section 4.6.5).

Listens on localhost:7432.  All endpoints return JSON.

Endpoints:
    GET  /health                 — liveness + last heartbeat seen
    GET  /status                 — full trainer state
    GET  /history                — past cycle summaries
    GET  /validation/latest      — last validation report
    POST /pause                  — suspend scheduling
    POST /resume                 — re-enable scheduling
    POST /rollback/{version}     — promote a historical checkpoint
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from gristmill_ml.trainer.service import GristMillTrainerService

logger = logging.getLogger(__name__)

API_HOST = "127.0.0.1"
API_PORT = 7432


def create_app(service: "GristMillTrainerService") -> FastAPI:
    """Build and return the FastAPI application bound to *service*."""
    app = FastAPI(
        title="gristmill-trainer",
        description="GristMill distillation trainer health & control API",
        version="2.0.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # ── GET /health ───────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> JSONResponse:
        info = service.health_info()
        return JSONResponse({"ok": True, **info})

    # ── GET /status ───────────────────────────────────────────────────────────

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse(service.status_snapshot())

    # ── GET /history ──────────────────────────────────────────────────────────

    @app.get("/history")
    async def history() -> JSONResponse:
        return JSONResponse(service.cycle_history())

    # ── GET /validation/latest ────────────────────────────────────────────────

    @app.get("/validation/latest")
    async def validation_latest() -> JSONResponse:
        result = service.latest_validation_result()
        if result is None:
            raise HTTPException(status_code=404, detail="No validation results yet")
        return JSONResponse(result)

    # ── POST /pause ───────────────────────────────────────────────────────────

    @app.post("/pause")
    async def pause() -> JSONResponse:
        service.pause("user_request")
        return JSONResponse({"paused": True})

    # ── POST /resume ──────────────────────────────────────────────────────────

    @app.post("/resume")
    async def resume() -> JSONResponse:
        service.resume()
        return JSONResponse({"paused": False})

    # ── POST /rollback/{version} ──────────────────────────────────────────────

    @app.post("/rollback/{version}")
    async def rollback(version: int) -> JSONResponse:
        ok = service.manual_rollback(version)
        if not ok:
            raise HTTPException(
                status_code=404,
                detail=f"No checkpoint at version {version}",
            )
        return JSONResponse({"rolled_back_to": version})

    return app


async def run_api(service: "GristMillTrainerService") -> None:
    """Start uvicorn in-process.  Call from asyncio event loop."""
    import uvicorn

    app = create_app(service)
    config = uvicorn.Config(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    logger.info("Trainer health API starting on http://%s:%d", API_HOST, API_PORT)
    await server.serve()

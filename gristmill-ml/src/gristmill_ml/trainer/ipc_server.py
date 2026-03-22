"""TrainerIpcServer — Unix domain socket server for trainer → Inference Stack IPC.

Emits newline-delimited JSON messages to all connected clients (Section 4.6.4).
The Inference Stack connects on startup and reconnects with exponential backoff
if the trainer is not yet running.

Message types emitted:
    checkpoint_promoted   — adapter hot-load trigger
    checkpoint_rolled_back — keep current adapter
    training_started      — UI status indicator
    training_progress     — emitted every 30s during active cycle
    trainer_paused        — UI status update
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SOCK = Path("/gristmill/run/trainer.sock")
_FALLBACK_SOCK = Path.home() / ".gristmill" / "run" / "trainer.sock"


class TrainerIpcServer:
    """asyncio Unix-socket server that broadcasts trainer events to subscribers.

    Usage::

        server = TrainerIpcServer()
        asyncio.create_task(server.serve())
        ...
        await server.emit("checkpoint_promoted", {"version": 3, "validation_score": 0.87})
    """

    def __init__(self, sock_path: Optional[Path] = None) -> None:
        if sock_path is None:
            parent = _DEFAULT_SOCK.parent
            sock_path = _DEFAULT_SOCK if (parent.exists() or _try_mkdir(parent)) else _FALLBACK_SOCK
        _try_mkdir(sock_path.parent)
        self.sock_path = sock_path
        self._clients: set[asyncio.StreamWriter] = set()
        self._server: Optional[asyncio.AbstractServer] = None

    # ── Server lifecycle ──────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Start listening.  Never returns (run as a task)."""
        # Remove stale socket file
        if self.sock_path.exists():
            self.sock_path.unlink()

        self._server = await asyncio.start_unix_server(
            self._handle_client, path=str(self.sock_path)
        )
        logger.info("TrainerIpcServer listening at %s", self.sock_path)
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        if self.sock_path.exists():
            self.sock_path.unlink(missing_ok=True)

    # ── Client handler ────────────────────────────────────────────────────────

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername", "<unknown>")
        logger.info("IPC client connected: %s (total=%d)", peer, len(self._clients) + 1)
        self._clients.add(writer)
        try:
            # Drain any bytes the client sends (it's a subscribe-only protocol)
            while not reader.at_eof():
                await reader.read(256)
        except (asyncio.CancelledError, ConnectionResetError):
            pass
        finally:
            self._clients.discard(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            logger.info("IPC client disconnected (remaining=%d)", len(self._clients))

    # ── Emit ──────────────────────────────────────────────────────────────────

    async def emit(self, message_type: str, payload: dict[str, Any]) -> None:
        """Broadcast a newline-delimited JSON message to all connected clients."""
        msg = json.dumps({"type": message_type, **payload}) + "\n"
        encoded = msg.encode()
        dead: list[asyncio.StreamWriter] = []
        for writer in list(self._clients):
            try:
                writer.write(encoded)
                await writer.drain()
            except Exception as exc:
                logger.debug("IPC client write failed: %s", exc)
                dead.append(writer)
        for w in dead:
            self._clients.discard(w)

    # ── Convenience emitters (typed) ──────────────────────────────────────────

    async def emit_checkpoint_promoted(
        self, version: int, validation_score: float, record_count: int
    ) -> None:
        await self.emit(
            "checkpoint_promoted",
            {
                "version": version,
                "validation_score": validation_score,
                "record_count": record_count,
            },
        )

    async def emit_checkpoint_rolled_back(self, version: int, reason: str) -> None:
        await self.emit(
            "checkpoint_rolled_back",
            {"version": version, "reason": reason},
        )

    async def emit_training_started(
        self, estimated_duration_minutes: int, record_count: int
    ) -> None:
        await self.emit(
            "training_started",
            {
                "estimated_duration_minutes": estimated_duration_minutes,
                "record_count": record_count,
            },
        )

    async def emit_training_progress(self, pct_complete: float, elapsed_minutes: float) -> None:
        await self.emit(
            "training_progress",
            {"pct_complete": round(pct_complete, 3), "elapsed_minutes": round(elapsed_minutes, 1)},
        )

    async def emit_trainer_paused(self, reason: str) -> None:
        await self.emit("trainer_paused", {"reason": reason})

    @property
    def client_count(self) -> int:
        return len(self._clients)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _try_mkdir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False

"""Entry point: python -m gristmill_ml.trainer  (or  gristmill-trainer CLI).

Starts three concurrent tasks in a single event loop:
    1. TrainerIpcServer  — Unix socket, emits events to Inference Stack
    2. GristMillTrainerService  — scheduler + cycle orchestration
    3. FastAPI health API  — localhost:7432
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

logger = logging.getLogger(__name__)


async def _main() -> None:
    from gristmill_ml.trainer.health_api import run_api
    from gristmill_ml.trainer.ipc_server import TrainerIpcServer
    from gristmill_ml.trainer.service import GristMillTrainerService

    # Wire components
    ipc = TrainerIpcServer()
    service = GristMillTrainerService(ipc_server=ipc)

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    logger.info("gristmill-trainer starting")

    ipc_task = asyncio.create_task(ipc.serve(), name="ipc-server")
    svc_task = asyncio.create_task(service.run(), name="trainer-service")
    api_task = asyncio.create_task(run_api(service), name="health-api")

    # Wait until a stop signal is received
    await stop_event.wait()

    logger.info("Stopping gristmill-trainer …")
    for task in (api_task, svc_task, ipc_task):
        task.cancel()
    await asyncio.gather(api_task, svc_task, ipc_task, return_exceptions=True)
    await ipc.stop()
    logger.info("gristmill-trainer stopped")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    asyncio.run(_main())


if __name__ == "__main__":
    main()

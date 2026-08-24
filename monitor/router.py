"""Monitor API: live state, a server-sent event stream, and start/stop."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from monitor.engine import ENGINE
from monitor.state import STATE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/monitor", tags=["monitor"])


def _auth():
    """Signed-in users only, when auth is available.

    Imported lazily so the monitor still runs in a dev app that has no auth
    wired up, rather than failing to mount.
    """
    try:
        from backend.auth import get_current_user

        return [Depends(get_current_user)]
    except Exception:                                   # noqa: BLE001
        return []


@router.get("/state")
async def state() -> dict:
    return STATE.snapshot()


@router.post("/start")
async def start(interval: float | None = None) -> dict:
    await ENGINE.start(interval)
    return {"running": True, "interval": ENGINE.interval,
            "watch_threshold": ENGINE.watch_threshold}


@router.post("/stop")
async def stop() -> dict:
    await ENGINE.stop()
    return {"running": False}


@router.post("/pause")
async def pause() -> dict:
    ENGINE.pause()
    return {"running": STATE.running, "paused": True}


@router.post("/resume")
async def resume() -> dict:
    ENGINE.resume()
    return {"running": STATE.running, "paused": False}


@router.post("/restart")
async def restart(interval: float | None = None) -> dict:
    await ENGINE.restart(interval)
    return {"running": True, "paused": False, "interval": ENGINE.interval}


@router.get("/runtime")
async def runtime() -> dict:
    """Monitor state plus the upstream detector's own runtime.

    One call answers "is the platform working" — whether the loop is running
    AND whether the model behind it is actually loaded.
    """
    import httpx

    from backend import config

    out = {
        "monitor": {
            "running": STATE.running,
            "paused": ENGINE.paused,
            "interval": ENGINE.interval,
            "watch_threshold": ENGINE.watch_threshold,
            "fusion": "meta_classifier" if ENGINE._fusion else "mean_fallback",
            **STATE.counters.as_dict(),
        },
        "detectors": {},
    }
    for name, key in (
        ("graph", "graph_api_base"),
        ("behavioural", "behavioral_api_base"),
        ("temporal", "temporal_api_base"),
    ):
        base = str(config.get("upstream", key)).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=4.0) as c:
                r = await c.get(f"{base}/api/graph/runtime" if name == "graph" else f"{base}/health")
            out["detectors"][name] = {"reachable": r.status_code < 500, **(r.json() if r.status_code == 200 else {})}
        except Exception as exc:                        # noqa: BLE001
            out["detectors"][name] = {"reachable": False, "error": type(exc).__name__}
    return out


@router.get("/stream")
async def stream() -> StreamingResponse:
    """Server-sent events for the live dashboard.

    The first frame is a full snapshot so a client that connects mid-run paints
    a correct screen immediately instead of waiting for the next event. A
    heartbeat every 15s keeps proxies from closing an idle connection.
    """
    queue = STATE.subscribe()

    async def gen():
        try:
            yield f"event: snapshot\ndata: {json.dumps(STATE.snapshot())}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield f"event: {event['kind']}\ndata: {json.dumps(event, default=str)}\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            STATE.unsubscribe(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # nginx would otherwise buffer the stream
        },
    )

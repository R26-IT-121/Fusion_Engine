"""Monitor API: live state, a server-sent event stream, and start/stop."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from monitor.engine import ENGINE
from monitor.state import STATE

logger = logging.getLogger(__name__)

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


# Reading the monitor requires a signed-in user. This dependency existed but
# was never passed to the router, so every GET here — live state, detector
# errors, queue depth, alert-delivery counts and the sending address — was
# answering anonymous callers. The write routes were guarded; the reads were
# not, which is the easier half to overlook.
router = APIRouter(prefix="/api/monitor", tags=["monitor"], dependencies=_auth())


def _admin():
    """Administrators only — the routes that change what the pipeline is doing.

    Reading the monitor is for everyone signed in; starting, pausing and
    stopping it is not. Until now these five routes carried the same guard as
    /state, so a read-only analyst could stop fraud screening for the whole
    institution with one request. Hiding the buttons in the console would not
    have fixed that: the endpoint was the hole, not the button.
    """
    try:
        from backend.auth import require_admin

        return [Depends(require_admin)]
    except Exception:                                   # noqa: BLE001
        return []


@router.get("/state")
async def state() -> dict:
    """Monitor state, plus where its transactions are coming from.

    `source` matters: a dashboard that shows throughput without saying whether
    those are ingested transactions or replayed samples invites the reader to
    assume the former.
    """
    from monitor import queue as ingest_queue

    snap = STATE.snapshot()
    snap["source"] = getattr(ENGINE, "_source", None)
    snap["queue"] = await ingest_queue.depth()
    return snap


@router.get("/briefing")
async def briefing(hours: int = 24) -> dict:
    """The daily digest, as data and as text."""
    from monitor import briefing as brief

    data = await brief.gather(hours=hours)
    return {**data, "text": brief.render_text(data)}


@router.post("/briefing/send")
async def send_briefing(hours: int = 24) -> dict:
    """Email the digest to the configured risk managers."""
    import asyncio

    from backend.email_service import _send_plain
    from backend.settings import get_alert_recipients
    from monitor import briefing as brief

    recipients = await get_alert_recipients()
    if not recipients:
        raise HTTPException(
            409,
            "No risk managers are configured. Add recipients under Settings.",
        )

    data = await brief.gather(hours=hours)
    body = brief.render_text(data)
    subject = (
        f"DeepSentinel briefing — {data.get('total_cases', 0)} case(s), "
        f"{data.get('open_for_review', 0)} awaiting review"
    )

    # _send_plain is synchronous and does network I/O, so it goes to a thread
    # rather than blocking the event loop the monitor runs on.
    sent = await asyncio.to_thread(_send_plain, subject, body, recipients)
    if not sent:
        raise HTTPException(
            409,
            "Email is not configured, or SMTP rejected the message. "
            "Check the SMTP settings and try the test email under Settings.",
        )
    return {"sent": True, "recipients": recipients, "subject": subject}


@router.post("/start", dependencies=_admin())
async def start(interval: float | None = None) -> dict:
    await ENGINE.start(interval)
    return {"running": True, "interval": ENGINE.interval,
            "watch_threshold": ENGINE.watch_threshold}


@router.post("/stop", dependencies=_admin())
async def stop() -> dict:
    await ENGINE.stop()
    return {"running": False}


@router.post("/pause", dependencies=_admin())
async def pause() -> dict:
    ENGINE.pause()
    return {"running": STATE.running, "paused": True}


@router.post("/resume", dependencies=_admin())
async def resume() -> dict:
    ENGINE.resume()
    return {"running": STATE.running, "paused": False}


@router.post("/restart", dependencies=_admin())
async def restart(interval: float | None = None) -> dict:
    await ENGINE.restart(interval)
    return {"running": True, "paused": False, "interval": ENGINE.interval}


@router.post("/clear", dependencies=_admin())
async def clear() -> dict:
    """Empty the alert list, the activity feed and the counters.

    Administrators only, for the same reason start and stop are: on a shared
    database this wipes what everyone's dashboard is looking at, and an
    analyst clearing the board mid-shift would drop the queue of things
    somebody still has to act on.

    Nothing in the database is touched. Cases these alerts raised remain in
    `fraud_cases` and stay readable under Cases — this clears the live view,
    not the record. That distinction is the whole reason it is safe to offer
    as a button.
    """
    removed = STATE.clear_live(actor="admin")

    # Shared state, so who cleared it is worth keeping. Audit is best-effort:
    # the clear has already happened and must not be reported as failed
    # because the audit write did not land.
    try:
        from backend.auth import audit

        await audit("monitor.clear", actor="admin", target="monitor live state",
                    detail=f"alerts={removed['alerts']}, events={removed['events']}")
    except Exception as exc:                             # noqa: BLE001
        logger.warning(f"Could not audit the monitor clear: {exc}")

    logger.info(f"Monitor live state cleared: {removed}")
    return {"cleared": True, "removed": removed, "counters": STATE.counters.as_dict()}


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
            body = r.json() if r.status_code == 200 else {}
            reachable = r.status_code < 500
            out["detectors"][name] = {
                "reachable": reachable,
                "ready": reachable and _ready(name, body),
                **body,
            }
        except Exception as exc:                        # noqa: BLE001
            out["detectors"][name] = {
                "reachable": False, "ready": False, "error": type(exc).__name__,
            }

    # The detectors are the loud failures — an operator sees a verdict go
    # missing. What follows are the quiet ones: a full pipeline that cannot
    # deliver its mail, a report generator whose quota ran out, an ingestion
    # queue backing up. Each of these fails without changing anything the
    # detectors report, which is exactly why they belong on an operator's page.
    out["services"] = await _services()
    out["delivery"] = await _delivery()
    out["queue"] = await _queue_depth()
    return out


async def _services() -> dict:
    """The dependencies that are not detectors, and can fail on their own."""
    from backend import main as backend_main

    svc = {
        "fusion": {
            "ok": backend_main.meta_classifier is not None,
            "detail": "meta-classifier loaded" if backend_main.meta_classifier
                      else "not loaded — fusion would average instead",
        },
        "retrieval": {
            "ok": backend_main.retriever is not None,
            "detail": "FATF typologies indexed" if backend_main.retriever
                      else "no vector store — reports cannot cite a typology",
        },
        "reporter": {
            "ok": backend_main.forensic_reporter is not None,
            "detail": "language model reachable" if backend_main.forensic_reporter
                      else "no backend — alerts go out without a narrative",
        },
    }
    try:
        from backend import config
        svc["database"] = {
            "ok": True,
            "detail": "postgres" if "postgres" in str(config.get("database", "url"))
                      else "sqlite (local file)",
        }
    except Exception:                                   # noqa: BLE001
        svc["database"] = {"ok": False, "detail": "connection settings unreadable"}
    return svc


async def _delivery() -> dict:
    """Whether alerts are reaching anyone.

    Raised and delivered are different numbers, and only the second one
    matters. The case table records what actually happened at send time, so
    a silent SMTP failure shows up here as a gap rather than as nothing.
    """
    out = {"configured": False, "recipients": 0, "raised": 0, "delivered": 0}
    try:
        from backend.email_service import _provider
        from backend.settings import list_risk_managers

        provider, cfg = _provider()
        out["configured"] = provider == "smtp"
        out["sending_as"] = cfg.get("username") if provider == "smtp" else None
        managers = await list_risk_managers()
        out["recipients"] = len([m for m in managers if getattr(m, "enabled", True)])
    except Exception:                                   # noqa: BLE001
        pass
    # fraud_cases, not analysis_records. The first is what the monitor writes
    # when it raises an alert; the second is what the analyzer writes when
    # somebody scores a transaction by hand, and nothing about that path sends
    # mail. Counting the analyzer's rows here reported four un-emailed manual
    # runs as four undelivered alerts, and hid the real figure entirely.
    try:
        from sqlalchemy import text

        from backend.db.session import get_session

        async with get_session() as db:
            row = (await db.execute(text(
                "SELECT COUNT(*), SUM(CASE WHEN alert_sent THEN 1 ELSE 0 END) "
                "FROM fraud_cases WHERE classification <> 'LOW'"
            ))).first()
        if row:
            out["raised"] = int(row[0] or 0)
            out["delivered"] = int(row[1] or 0)
    except Exception:                                   # noqa: BLE001
        # No fraud_cases table on this database — report nothing rather than
        # zero, which would read as "everything failed".
        out["raised"] = out["delivered"] = None
    return out


async def _queue_depth() -> dict:
    from monitor import queue as ingest_queue

    try:
        return await ingest_queue.depth()
    except Exception:                                   # noqa: BLE001
        return {"available": False}


def _ready(name: str, body: dict) -> bool:
    """Whether a detector can actually score, not merely whether it replied.

    Answering a health probe and being able to return a verdict are different
    things, and conflating them is how a dead detector comes to be counted as
    live: a service that has started but has no weights on disk still returns
    200. Each upstream says this differently, so the shapes are normalised here
    rather than being re-derived by every caller.
    """
    if name == "graph":
        return bool((body.get("model") or {}).get("loaded"))
    if name == "behavioural":
        # Reports which strata it managed to load; any missing one means part
        # of the traffic cannot be scored.
        return (body.get("status") == "ok"
                and bool(body.get("strata_loaded"))
                and not body.get("strata_missing"))
    if name == "temporal":
        # `ready` is authoritative where the build provides it. Older builds
        # only sent `status`, so fall back to that rather than reporting a
        # working detector as broken.
        if "ready" in body:
            return bool(body["ready"])
        return body.get("status") == "ok"
    return body.get("status") == "ok"


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

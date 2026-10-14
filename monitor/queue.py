"""Reading transactions from the ingestion queue.

The Query Runner writes arriving transactions into `transactions_live`. This is
the other half: the monitor claims rows from that table, screens them, and marks
them done.

Claiming, not selecting
-----------------------
Two workers that both `SELECT` the pending rows will both process them. On
PostgreSQL the claim uses `FOR UPDATE SKIP LOCKED`, so a worker that meets a row
another already holds steps over it instead of waiting — neither blocks and
neither duplicates. SQLite cannot express that and allows one writer at a time,
so it gets a plain UPDATE, which is correct for one worker and unsafe for
several. That is the real reason a multi-worker deployment needs PostgreSQL.

The table may not exist
-----------------------
The queue lives in whichever database `DATABASE_URL` points at. If the Query
Runner wrote to a different database, or nobody has created the tables yet, this
module reports that plainly instead of failing — the monitor then falls back to
its sample source and says so, rather than appearing to screen live traffic that
is not there.
"""

from __future__ import annotations

import json
import logging
import socket
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from backend.db.session import get_engine, get_session

logger = logging.getLogger(__name__)

WORKER_ID = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"

# Checked once and cached: a missing table is a deployment fact, not something
# that changes between polls, and probing it every second would be noise.
_table_present: bool | None = None


async def queue_available() -> bool:
    """Does `transactions_live` exist in the database we are connected to?"""
    global _table_present
    if _table_present is not None:
        return _table_present
    try:
        async with get_session() as db:
            await db.execute(text("SELECT 1 FROM transactions_live LIMIT 1"))
        _table_present = True
        logger.info("Ingestion queue found — monitor will screen real arrivals.")
    except Exception as exc:                                  # noqa: BLE001
        _table_present = False
        logger.info(
            "No transactions_live table in this database (%s). The monitor will "
            "use its sample source. Point DATABASE_URL at the same database the "
            "Query Runner writes to, and create the tables there.",
            type(exc).__name__,
        )
    return _table_present


def reset_cache() -> None:
    """Forget the table check — used after the database configuration changes."""
    global _table_present
    _table_present = None


def _as_dict(value) -> dict:
    """A JSON column read through raw SQL comes back as text on some drivers."""
    if isinstance(value, dict):
        return value
    if isinstance(value, (str, bytes)):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


async def claim(limit: int = 10) -> list[dict]:
    """Claim up to `limit` pending transactions for this worker."""
    if not await queue_available():
        return []

    engine = get_engine()
    is_pg = engine.dialect.name in ("postgresql", "postgres")
    now = datetime.now(timezone.utc)

    try:
        async with get_session() as db:
            if is_pg:
                rows = (await db.execute(
                    text("""
                        UPDATE transactions_live SET
                            status='claimed', claimed_by=:w, claimed_at=:now,
                            attempts = attempts + 1
                        WHERE id IN (
                            SELECT id FROM transactions_live
                            WHERE status='pending'
                            ORDER BY received_at
                            LIMIT :n
                            FOR UPDATE SKIP LOCKED
                        )
                        RETURNING id, transaction_id, payload
                    """),
                    {"w": WORKER_ID, "now": now, "n": limit},
                )).all()
            else:
                ids = [r[0] for r in (await db.execute(
                    text("SELECT id FROM transactions_live WHERE status='pending' "
                         "ORDER BY received_at LIMIT :n"),
                    {"n": limit},
                )).all()]
                if not ids:
                    return []
                in_list = ",".join(str(int(i)) for i in ids)
                await db.execute(
                    text(f"UPDATE transactions_live SET status='claimed', claimed_by=:w, "
                         f"claimed_at=:now, attempts=attempts+1 WHERE id IN ({in_list})"),
                    {"w": WORKER_ID, "now": now},
                )
                rows = (await db.execute(
                    text(f"SELECT id, transaction_id, payload FROM transactions_live "
                         f"WHERE id IN ({in_list})")
                )).all()

        return [
            {"row_id": r[0], "transaction_id": r[1], **_as_dict(r[2])}
            for r in rows
        ]
    except Exception as exc:                                  # noqa: BLE001
        logger.warning(f"Could not claim from the queue: {type(exc).__name__}: {exc}")
        return []


async def mark_done(row_id: int, escalated: bool = False, error: str | None = None) -> None:
    """Close a claimed row out. Never raises — bookkeeping must not stop screening."""
    try:
        async with get_session() as db:
            await db.execute(
                text("UPDATE transactions_live SET status=:s, screened_at=:now, "
                     "escalated=:esc, last_error=:err WHERE id=:id"),
                {
                    "s": "failed" if error else "screened",
                    "now": datetime.now(timezone.utc),
                    "esc": bool(escalated),
                    "err": (error or None),
                    "id": row_id,
                },
            )
    except Exception as exc:                                  # noqa: BLE001
        logger.warning(f"Could not mark row {row_id} screened: {exc}")


async def release_stale(older_than_seconds: int = 180) -> int:
    """Return rows from a worker that died back to pending.

    Without this a crash mid-batch leaves those transactions claimed forever —
    silently never screened, which is the worst failure this system can have.
    """
    if not await queue_available():
        return 0
    cutoff = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() - older_than_seconds, tz=timezone.utc
    )
    try:
        async with get_session() as db:
            result = await db.execute(
                text("UPDATE transactions_live SET status='pending', claimed_by=NULL "
                     "WHERE status='claimed' AND claimed_at < :cutoff"),
                {"cutoff": cutoff},
            )
            n = result.rowcount or 0
        if n:
            logger.info(f"Returned {n} stalled transaction(s) to the queue.")
        return n
    except Exception as exc:                                  # noqa: BLE001
        logger.warning(f"Stale release failed: {exc}")
        return 0


async def depth() -> dict:
    """How much work is waiting — surfaced in the monitor's state."""
    if not await queue_available():
        return {"available": False}
    try:
        async with get_session() as db:
            rows = (await db.execute(
                text("SELECT status, COUNT(*) FROM transactions_live GROUP BY status")
            )).all()
        counts = {r[0]: r[1] for r in rows}
        return {
            "available": True,
            "pending": counts.get("pending", 0),
            "claimed": counts.get("claimed", 0),
            "screened": counts.get("screened", 0),
            "failed": counts.get("failed", 0),
        }
    except Exception:                                         # noqa: BLE001
        return {"available": False}

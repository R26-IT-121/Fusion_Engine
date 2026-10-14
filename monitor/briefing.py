"""The daily briefing.

One email each morning: what was screened, what was caught, what is still
waiting for someone. Built entirely from recorded cases — if nothing was
detected, it says so rather than padding the message.

Why a digest and not just alerts
--------------------------------
Individual alerts answer "is this transaction suspicious". Nobody reading only
alerts can answer "is the system working, and is anything piling up". A quiet
night with zero alerts and a quiet night because the monitor was stopped look
identical from an inbox — this distinguishes them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from backend.db.session import get_session

logger = logging.getLogger(__name__)


async def gather(hours: int = 24) -> dict:
    """Everything the briefing reports on, for the last `hours`."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    out: dict = {
        "window_hours": hours,
        "since": since.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "available": False,
    }

    try:
        async with get_session() as db:
            by_class = {
                r[0]: r[1] for r in (await db.execute(
                    text("SELECT classification, COUNT(*) FROM fraud_cases "
                         "WHERE detected_at >= :s GROUP BY classification"),
                    {"s": since},
                )).all()
            }
            open_count = (await db.execute(
                text("SELECT COUNT(*) FROM fraud_cases WHERE review_status = 'open'")
            )).scalar_one()
            alerted = (await db.execute(
                text("SELECT COUNT(*) FROM fraud_cases "
                     "WHERE detected_at >= :s AND alert_sent = 1"),
                {"s": since},
            )).scalar_one()
            top = (await db.execute(
                text("SELECT case_ref, transaction_id, classification, fused_score, "
                     "typology_name, sink_account FROM fraud_cases "
                     "WHERE detected_at >= :s ORDER BY fused_score DESC LIMIT 5"),
                {"s": since},
            )).all()

            # Screened volume comes from the ingestion queue when it exists;
            # without it we can report what was caught but not out of how many,
            # and saying so is better than implying a denominator we do not have.
            screened = None
            try:
                screened = (await db.execute(
                    text("SELECT COUNT(*) FROM transactions_live "
                         "WHERE screened_at >= :s"), {"s": since},
                )).scalar_one()
            except Exception:                                 # noqa: BLE001
                pass

        out.update({
            "available": True,
            "screened": screened,
            "by_classification": by_class,
            "total_cases": sum(by_class.values()),
            "alerts_sent": alerted,
            "open_for_review": open_count,
            "top_cases": [
                {"case_ref": r[0], "transaction_id": r[1], "classification": r[2],
                 "fused_score": r[3], "typology": r[4], "sink_account": r[5]}
                for r in top
            ],
        })
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning(f"Briefing could not be assembled: {exc}")

    return out


def render_text(data: dict) -> str:
    """Plain text, so the digest is readable in any client."""
    if not data.get("available"):
        return ("DeepSentinel daily briefing\n\n"
                "No case history is available in this database, so there is "
                "nothing to report.")

    by = data["by_classification"]
    lines = [
        "DeepSentinel — daily briefing",
        f"Covering the last {data['window_hours']} hours.",
        "",
    ]

    if data["total_cases"] == 0:
        lines += [
            "No cases were detected in this period.",
            "",
            ("Note: this reads as a quiet period only if the monitor was running. "
             "Check the dashboard if that is unexpected."),
        ]
    else:
        if data.get("screened") is not None:
            lines.append(f"Screened          {data['screened']}")
        lines += [
            f"Cases detected    {data['total_cases']}",
            f"Alerts sent       {data['alerts_sent']}",
            "",
            "By severity",
        ]
        for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            if by.get(level):
                lines.append(f"  {level:<10} {by[level]}")

        if data["top_cases"]:
            lines += ["", "Highest scoring"]
            for c in data["top_cases"]:
                score = f"{c['fused_score']:.3f}" if c["fused_score"] is not None else "—"
                lines.append(
                    f"  {c['case_ref']}  {c['classification']:<8} {score}"
                    + (f"  {c['typology']}" if c["typology"] else "")
                )

    lines += [
        "",
        f"Awaiting review   {data['open_for_review']}",
        "",
        "— DeepSentinel. This is a summary; open the dashboard to act on a case.",
    ]
    return "\n".join(lines)

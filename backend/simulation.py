"""Replaying historical decisions at a different threshold.

The question an operator actually asks is not "what is our F1" but "if I move
the line, how many more alerts land on my desk tomorrow, and how many real cases
do I miss?" This answers that from what the system has already scored.

Why this is only meaningful because the scores are calibrated
------------------------------------------------------------
Sliding a threshold over raw model output tells you nothing transferable — the
number has no fixed meaning between models or over time. Isotonic calibration
brought expected calibration error from 0.80 to 0.024, so a threshold of 0.4 is
"40% likely to be fraud" rather than an arbitrary cut point. The slider is the
commercial expression of that work.

What this is not
----------------
A prediction. It replays decisions already made on transactions already seen,
so it says what *would have* happened, not what *will*. Alert volume shifts with
traffic; the response says how many records the estimate rests on so nobody
reads a confident number off twelve rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text

from backend.db.session import get_session

logger = logging.getLogger(__name__)

# Anything below this and the numbers are noise dressed as analysis.
MIN_SAMPLE = 20


@dataclass
class Point:
    threshold: float
    alerts: int
    precision: float | None
    recall: float | None
    f1: float | None
    true_positives: int
    false_positives: int
    false_negatives: int


async def _rows(days: int | None = None) -> list[tuple[float, bool | None]]:
    """(fused score, ground-truth label) for scored transactions.

    Reads `fraud_cases` when it exists, since that is where labels live, and
    falls back to `analysis_records` — which has scores but usually no labels,
    so volume can still be estimated even when accuracy cannot.
    """
    per = await _detector_rows(days)
    if per["fused"]:
        return per["fused"]

    async with get_session() as db:
        try:
            rows = (await db.execute(
                text("SELECT fraud_confidence_score, NULL FROM analysis_records "
                     "WHERE fraud_confidence_score IS NOT NULL")
            )).all()
            return [(float(r[0]), None) for r in rows]
        except Exception:                                     # noqa: BLE001
            return []


async def _detector_rows(days: int | None = None) -> dict[str, list]:
    """(score, label) per detector as well as fused.

    Every case stores what each detector said, so each one can be swept against
    the same history independently. That is the difference between "the
    platform's line is here" and "here is where my model's line should be" —
    and the four are genuinely different questions, because the detectors do
    not agree and their scores are not on a shared scale.

    A detector that never answered contributes no rows rather than a neutral
    score, so its curve is honestly empty instead of quietly flat.
    """
    out: dict[str, list] = {k: [] for k in ("graph", "behavioural", "temporal", "fused")}
    cols = {
        "graph": "graph_score",
        "behavioural": "behavioral_score",
        "temporal": "temporal_score",
        "fused": "fused_score",
    }
    try:
        async with get_session() as db:
            q = ("SELECT fused_score, graph_score, behavioral_score, temporal_score, "
                 "label_is_fraud FROM fraud_cases WHERE fused_score IS NOT NULL")
            if days:
                q += " AND detected_at >= :cutoff"
            rows = (await db.execute(
                text(q), {"cutoff": _cutoff(days)} if days else {},
            )).all()
    except Exception:                                         # noqa: BLE001
        return out

    for fused, g, b, t, label in rows:
        lab = None if label is None else bool(label)
        for key, value in (("fused", fused), ("graph", g),
                           ("behavioural", b), ("temporal", t)):
            if value is not None:
                out[key].append((float(value), lab))
    return out


def _cutoff(days: int):
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone.utc) - timedelta(days=days)


def _score_at(rows: list[tuple[float, bool | None]], threshold: float) -> Point:
    alerts = tp = fp = fn = 0
    for score, label in rows:
        flagged = score >= threshold
        if flagged:
            alerts += 1
        if label is None:
            continue                       # counts toward volume, not accuracy
        if flagged and label:
            tp += 1
        elif flagged and not label:
            fp += 1
        elif not flagged and label:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    return Point(
        threshold=round(threshold, 4),
        alerts=alerts,
        precision=None if precision is None else round(precision, 4),
        recall=None if recall is None else round(recall, 4),
        f1=None if f1 is None else round(f1, 4),
        true_positives=tp, false_positives=fp, false_negatives=fn,
    )


async def sweep(days: int | None = None, points: int = 41) -> dict:
    """Every threshold from 0 to 1, scored against history."""
    rows = await _rows(days)
    labelled = sum(1 for _, l in rows if l is not None)

    if not rows:
        return {
            "sample_size": 0, "labelled": 0, "curve": [], "best": None,
            "message": (
                "Nothing has been scored yet. Run the monitor, or analyse a few "
                "transactions, and the curve will fill in."
            ),
        }

    curve = [_score_at(rows, i / (points - 1)) for i in range(points)]

    # Only claim an optimum when there are labels and enough of them to mean
    # something. Otherwise the caller gets volume and no accuracy claim.
    best = None
    if labelled >= MIN_SAMPLE:
        # Exclude the degenerate end of the sweep. At threshold 0 everything
        # alerts, which scores perfect recall and therefore often wins on F1 —
        # but "alert on every transaction" is the absence of a decision, not an
        # operating point, and offering it as the recommended one would be
        # actively misleading to an operator.
        scored = [p for p in curve if p.f1 is not None and p.threshold > 0.0]
        if scored:
            best = max(scored, key=lambda p: p.f1)

    message = ""
    if labelled == 0:
        message = (
            "No ground-truth labels in this history, so alert volume can be "
            "estimated but accuracy cannot. Ingest a file with an isFraud column "
            "to see precision and recall."
        )
    elif labelled < MIN_SAMPLE:
        message = (
            f"Only {labelled} labelled record(s). Precision and recall are shown "
            f"but are not yet stable — {MIN_SAMPLE} is the minimum worth reading."
        )

    # Each detector swept on its own, so the page can offer a line per model
    # rather than one line for the platform.
    per = await _detector_rows(days)
    detectors = {}
    for key, drows in per.items():
        if key == "fused" or not drows:
            continue
        dcurve = [_score_at(drows, i / (points - 1)) for i in range(points)]
        dlabelled = sum(1 for _, l in drows if l is not None)
        dbest = None
        if dlabelled >= MIN_SAMPLE:
            scored_d = [p for p in dcurve if p.f1 is not None and p.threshold > 0.0]
            if scored_d:
                dbest = max(scored_d, key=lambda p: p.f1)
        detectors[key] = {
            "sample_size": len(drows),
            "labelled": dlabelled,
            "curve": [p.__dict__ for p in dcurve],
            "best": dbest.__dict__ if dbest else None,
        }

    return {
        "sample_size": len(rows),
        "labelled": labelled,
        "curve": [p.__dict__ for p in curve],
        "best": best.__dict__ if best else None,
        "detectors": detectors,
        "message": message,
    }


async def at(threshold: float, days: int | None = None) -> dict:
    """One threshold, for a slider that moves faster than a full sweep."""
    rows = await _rows(days)
    if not rows:
        return {"sample_size": 0, "message": "Nothing scored yet."}
    point = _score_at(rows, max(0.0, min(float(threshold), 1.0)))
    return {
        "sample_size": len(rows),
        "labelled": sum(1 for _, l in rows if l is not None),
        **point.__dict__,
    }

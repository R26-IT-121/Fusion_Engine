"""Recording what the models caught, and why.

`fraud_cases` is the table a stakeholder reads. For one alert it holds: what
each model scored and whether it answered at all, the matched typology, the
graph's structural evidence and the behavioural detector's attribution, how
long it took, who was told, and what a human decided afterwards.

The evidence columns are the difference between a case a reviewer can act on
and a row of numbers. A score says a transaction was unusual; the attribution
says which feature and which latent dimension made it unusual, which is the
question the reviewer actually has.

Two things are recorded that are easy to leave out and matter later:

  * **whether each model was available.** A confidence built from one detector
    is not the same claim as one built from three. Without this the number is
    uninterpretable six months on.
  * **the ground-truth label, when the source file carried one.** Kept so
    precision can be measured after the fact — never shown as a model output.

Writing a case must never break the alert that produced it, so every failure
here is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from backend.db.session import get_session

logger = logging.getLogger(__name__)

_table_present: bool | None = None


async def available() -> bool:
    global _table_present
    if _table_present is not None:
        return _table_present
    try:
        async with get_session() as db:
            await db.execute(text("SELECT 1 FROM fraud_cases LIMIT 1"))
        _table_present = True
    except Exception:                                         # noqa: BLE001
        _table_present = False
        logger.info(
            "No fraud_cases table in this database — cases will not be recorded. "
            "Create it with the Query Runner, pointed at the same database."
        )
    return _table_present


def reset_cache() -> None:
    global _table_present
    _table_present = None


async def _label_from_archive(transaction_id: str) -> bool | None:
    """The ground-truth label recorded at ingestion, if the file had one."""
    try:
        async with get_session() as db:
            row = (await db.execute(
                text("SELECT is_fraud FROM transactions_archive "
                     "WHERE transaction_id = :t ORDER BY id DESC LIMIT 1"),
                {"t": transaction_id},
            )).first()
        return None if row is None or row[0] is None else bool(row[0])
    except Exception:                                         # noqa: BLE001
        return None


def new_case_ref() -> str:
    """A short human-quotable reference, e.g. CASE-2026-08-26-A3F9."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"CASE-{today}-{uuid.uuid4().hex[:4].upper()}"


def _js(value) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return None


async def record(
    *,
    transaction_id: str,
    classification: str,
    fused_score: float,
    scores: dict,
    available_flags: dict,
    modalities_used: int,
    payload: dict | None = None,
    graph_evidence: dict | None = None,
    behavioral_evidence: dict | None = None,
    typology: dict | None = None,
    forensic_report: str | None = None,
    screening_ms: int | None = None,
    total_ms: int | None = None,
    alert_sent: bool = False,
    recipients: list[str] | None = None,
    label_is_fraud: bool | None = None,
    model_versions: dict | None = None,
) -> str | None:
    """Write one case. Returns its reference, or None if it could not be stored."""
    if not await available():
        return None

    # Ground truth is deliberately absent from the queue payload — a model must
    # never see it. When the source file carried a label it is in the archive,
    # so look it up there rather than plumbing it through the screening path.
    if label_is_fraud is None:
        label_is_fraud = await _label_from_archive(transaction_id)

    sg = graph_evidence or {}
    typ = typology or {}
    case_ref = new_case_ref()

    # Keep the transaction this verdict was reached on. Without it a case can
    # be read but never re-run, and "analyse the transaction behind this case"
    # is the first thing anyone wants to do with a case. Queue-sourced rows are
    # already in the archive; replayed samples were not persisted anywhere.
    await _archive_if_absent(transaction_id, payload)

    # Recorded explicitly rather than inferred later: with fewer than three
    # detectors the fusion applies an uncertainty penalty, and a reader needs to
    # know the score was deliberately conservative.
    penalty = modalities_used < 3

    try:
        async with get_session() as db:
            await db.execute(
                text("""
                    INSERT INTO fraud_cases (
                        case_ref, transaction_id, business_date, detected_at,
                        classification, fused_score,
                        graph_score, behavioral_score, temporal_score,
                        graph_available, behavioral_available, temporal_available,
                        modalities_used, uncertainty_penalty_applied,
                        typology_id, typology_name, typology_similarity,
                        graph_pattern, sink_account, implicated_accounts,
                        graph_evidence, behavioral_evidence, forensic_report,
                        screening_ms, total_ms,
                        alert_sent, alerted_at, recipients,
                        review_status, label_is_fraud, model_versions
                    ) VALUES (
                        :case_ref, :transaction_id, :business_date, :detected_at,
                        :classification, :fused_score,
                        :graph_score, :behavioral_score, :temporal_score,
                        :graph_available, :behavioral_available, :temporal_available,
                        :modalities_used, :penalty,
                        :typology_id, :typology_name, :typology_similarity,
                        :graph_pattern, :sink_account, :implicated_accounts,
                        :graph_evidence, :behavioral_evidence, :forensic_report,
                        :screening_ms, :total_ms,
                        :alert_sent, :alerted_at, :recipients,
                        'open', :label_is_fraud, :model_versions
                    )
                """),
                {
                    "case_ref": case_ref,
                    "transaction_id": transaction_id,
                    "business_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "detected_at": datetime.now(timezone.utc),
                    "classification": classification,
                    "fused_score": float(fused_score),
                    "graph_score": scores.get("graph"),
                    "behavioral_score": scores.get("behavioural"),
                    "temporal_score": scores.get("temporal"),
                    "graph_available": bool(available_flags.get("graph")),
                    "behavioral_available": bool(available_flags.get("behavioural")),
                    "temporal_available": bool(available_flags.get("temporal")),
                    "modalities_used": modalities_used,
                    "penalty": penalty,
                    "typology_id": typ.get("typology_id"),
                    "typology_name": typ.get("typology_name"),
                    "typology_similarity": typ.get("similarity_score"),
                    "graph_pattern": sg.get("pattern"),
                    "sink_account": sg.get("sink_account"),
                    "implicated_accounts": _js(
                        [n.get("account_id") for n in (sg.get("nodes") or [])][:50]
                    ),
                    "graph_evidence": _js(sg or None),
                    "behavioral_evidence": _js(behavioral_evidence or None),
                    "forensic_report": forensic_report,
                    "screening_ms": screening_ms,
                    "total_ms": total_ms,
                    "alert_sent": bool(alert_sent),
                    "alerted_at": datetime.now(timezone.utc) if alert_sent else None,
                    "recipients": _js(recipients),
                    "label_is_fraud": label_is_fraud,
                    "model_versions": _js(model_versions),
                },
            )
        logger.info(f"Recorded {case_ref} for {transaction_id} ({classification})")
        return case_ref
    except Exception as exc:                                  # noqa: BLE001
        # A case that cannot be stored must not take the alert down with it.
        logger.warning(f"Could not record case for {transaction_id}: {type(exc).__name__}: {exc}")
        return None


async def mark_alerted(case_ref: str, sent: bool) -> None:
    """Record whether an alert email actually went out for this case.

    Called after the send is attempted, so `alert_sent` means "someone was
    told" rather than "we meant to tell someone". The distinction matters: a
    case marked alerted that nobody received is worse than one marked not
    alerted, because it stops anyone looking.
    """
    from datetime import datetime, timezone

    try:
        async with get_session() as db:
            await db.execute(
                text("UPDATE fraud_cases SET alert_sent = :s, alerted_at = :at "
                     "WHERE case_ref = :r"),
                {"s": bool(sent),
                 "at": datetime.now(timezone.utc) if sent else None,
                 "r": case_ref},
            )
    except Exception as exc:                                  # noqa: BLE001
        logger.warning(f"Could not record alert outcome for {case_ref}: {exc}")


async def _archive_if_absent(transaction_id: str, payload: dict | None) -> None:
    """Store a replayed sample transaction so its case can be re-analysed.

    Rows that arrived through the ingestion queue are already in the archive
    and are left untouched — this only backfills the sample feed, and marks it
    as such in `source_file` so replayed traffic is never mistaken for
    evidence a customer uploaded.

    Best-effort by design: failing to archive a transaction must not prevent
    the case from being recorded.
    """
    if not payload:
        return
    try:
        async with get_session() as db:
            exists = (await db.execute(
                text("SELECT 1 FROM transactions_archive "
                     "WHERE transaction_id = :t LIMIT 1"),
                {"t": transaction_id},
            )).first()
            if exists:
                return
            await db.execute(
                text("""
                    INSERT INTO transactions_archive (
                        transaction_id, business_date, step, tx_type, amount,
                        name_orig, name_dest, old_balance_orig, new_balance_orig,
                        old_balance_dest, new_balance_dest,
                        raw, source_file, uploaded_by, uploaded_at
                    ) VALUES (
                        :t, :bd, :step, :type, :amount,
                        :orig, :dest, :ob_o, :nb_o, :ob_d, :nb_d,
                        :raw, 'monitor:sample-replay', 'monitor', :at
                    )
                """),
                {
                    "t": transaction_id,
                    "bd": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "step": _int(payload.get("step")),
                    "type": payload.get("type"),
                    "amount": _float(payload.get("amount")),
                    "orig": payload.get("nameOrig"),
                    "dest": payload.get("nameDest"),
                    "ob_o": _float(payload.get("oldbalanceOrg")),
                    "nb_o": _float(payload.get("newbalanceOrig")),
                    "ob_d": _float(payload.get("oldbalanceDest")),
                    "nb_d": _float(payload.get("newbalanceDest")),
                    "raw": _js(payload),
                    "at": datetime.now(timezone.utc),
                },
            )
    except Exception as exc:                                  # noqa: BLE001
        logger.debug(f"Could not archive {transaction_id} for replay: {exc}")


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None

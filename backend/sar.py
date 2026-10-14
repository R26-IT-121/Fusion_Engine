"""Suspicious Activity Report drafting.

A SAR is a legal statement made by a named officer to a regulator. This module
produces a *draft* of one and nothing more:

  - the generated text is stored immutably, so what the model wrote can always
    be compared with what a human approved;
  - approval records who accepted it and when;
  - nothing is transmitted anywhere. Filing remains a deliberate human act in
    the institution's own system of record.

Drafts are built from the persisted `analysis_records` row rather than by
re-running the pipeline. A filing must describe what the system concluded at
the time of the alert; re-scoring against a graph that has since moved on would
produce a document that does not match the alert it claims to describe.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select

from backend.db.models import AnalysisRecord, SarDraft, SarStatus, as_utc
from backend.db.session import get_session

logger = logging.getLogger(__name__)

# Carried on every rendering of a draft. The UI shows it, the print stylesheet
# keeps it, and it is prepended to any exported text — a document that leaves
# this system must never be mistakable for a filed report.
DRAFT_WATERMARK = (
    "DRAFT — machine-generated, not reviewed, not filed. "
    "Requires review and approval by a designated compliance officer."
)

APPROVED_WATERMARK = (
    "APPROVED by {reviewer} — reviewed and accepted. NOT FILED. "
    "Submission to a regulator remains a separate, deliberate action."
)


def _watermark(draft) -> str:
    """Every state carries a banner, and none of them says "filed".

    An approved report is still not a filed report. Dropping the banner on
    approval would let a reader assume the submission had happened, which is
    the one misreading this feature must never enable.
    """
    if draft.status == SarStatus.APPROVED.value:
        return APPROVED_WATERMARK.format(reviewer=draft.reviewed_by or "an officer")
    if draft.status == SarStatus.REJECTED.value:
        return "REJECTED — not accepted by the reviewer. Retained for the audit trail."
    return DRAFT_WATERMARK

# Fields copied into the prompt. Kept explicit so it is obvious at a glance
# what a draft is allowed to be built from.
_RECORD_FIELDS = (
    "transaction_id", "tx_type", "amount", "name_orig", "name_dest", "step",
    "fraud_confidence_score", "classification", "modalities_used",
    "graph_score", "behavioral_score", "temporal_score",
    "graph_available", "behavioral_available", "temporal_available",
    "typology_id", "typology_name", "similarity_score",
    "forensic_report", "analysed_by",
)


def _to_dict(record: AnalysisRecord) -> dict:
    out = {f: getattr(record, f, None) for f in _RECORD_FIELDS}
    created = as_utc(record.created_at)
    out["created_at"] = created.isoformat() if created else None
    return out


def _serialise(draft: SarDraft) -> dict:
    return {
        "id": draft.id,
        "analysis_id": draft.analysis_id,
        "status": draft.status,
        "generated_text": draft.generated_text,
        "edited_text": draft.edited_text,
        "text": draft.edited_text or draft.generated_text,
        "was_edited": bool(draft.edited_text and draft.edited_text != draft.generated_text),
        "model_version": draft.model_version,
        "generated_by": draft.generated_by,
        "generated_at": (as_utc(draft.generated_at).isoformat() if draft.generated_at else None),
        "reviewed_by": draft.reviewed_by,
        "reviewed_at": (as_utc(draft.reviewed_at).isoformat() if draft.reviewed_at else None),
        "review_note": draft.review_note,
        "watermark": _watermark(draft),
        "filed": False,   # this system never files; stated explicitly, always
    }


async def get_analysis(analysis_id: int) -> AnalysisRecord:
    async with get_session() as db:
        record = await db.get(AnalysisRecord, analysis_id)
        if record is None:
            raise HTTPException(404, f"No analysis record {analysis_id}")
        return record


async def latest_draft(analysis_id: int) -> dict | None:
    async with get_session() as db:
        rows = await db.execute(
            select(SarDraft)
            .where(SarDraft.analysis_id == analysis_id)
            .order_by(SarDraft.id.desc())
            .limit(1)
        )
        draft = rows.scalar_one_or_none()
        return _serialise(draft) if draft else None


async def generate(analysis_id: int, reporter, actor: str | None = None) -> dict:
    """Draft a SAR for one analysis. Returns the persisted draft."""
    from backend.rag.prompt_builder import build_sar_prompt

    record = await get_analysis(analysis_id)

    if not record.forensic_report:
        # The narrative is the substance of sections 3-5. Without it the draft
        # would be a table of numbers wearing a report's clothes.
        raise HTTPException(
            409,
            "This alert has no forensic narrative recorded, so a report cannot be "
            "drafted from it. Re-run the analysis with report generation enabled.",
        )

    if reporter is None:
        raise HTTPException(503, "No language model is configured for drafting.")

    package = build_sar_prompt(_to_dict(record))
    try:
        text = reporter.generate_report(package)
    except Exception as exc:                            # noqa: BLE001
        logger.exception("SAR drafting failed")
        raise HTTPException(502, f"Drafting failed: {type(exc).__name__}")

    if not text or not text.strip():
        raise HTTPException(502, "The model returned an empty draft.")

    async with get_session() as db:
        draft = SarDraft(
            analysis_id=analysis_id,
            generated_text=text.strip(),
            status=SarStatus.DRAFT.value,
            model_version=getattr(reporter, "model_version", None),
            generated_by=actor,
        )
        db.add(draft)
        await db.flush()
        result = _serialise(draft)
    return result


async def revise(draft_id: int, edited_text: str, actor: str | None = None) -> dict:
    """Record an officer's edits. The generated text is never overwritten."""
    if not edited_text or not edited_text.strip():
        raise HTTPException(422, "Edited text cannot be empty.")

    async with get_session() as db:
        draft = await db.get(SarDraft, draft_id)
        if draft is None:
            raise HTTPException(404, f"No draft {draft_id}")
        if draft.status == SarStatus.APPROVED.value:
            raise HTTPException(
                409,
                "This draft is already approved. Approved text is the record of what "
                "a named officer accepted and cannot be edited in place.",
            )
        draft.edited_text = edited_text.strip()
        draft.status = SarStatus.UNDER_REVIEW.value
        draft.reviewed_by = actor
        await db.flush()
        return _serialise(draft)


async def decide(
    draft_id: int, approve: bool, actor: str, note: str | None = None
) -> dict:
    """Approve or reject a draft.

    Approval means a named person accepted this text. It does not file anything
    and must never be presented as if it had.
    """
    if not actor:
        raise HTTPException(400, "An approval must be attributable to a named user.")

    async with get_session() as db:
        draft = await db.get(SarDraft, draft_id)
        if draft is None:
            raise HTTPException(404, f"No draft {draft_id}")
        draft.status = (SarStatus.APPROVED if approve else SarStatus.REJECTED).value
        draft.reviewed_by = actor
        draft.reviewed_at = datetime.now(timezone.utc)
        draft.review_note = note
        await db.flush()
        return _serialise(draft)

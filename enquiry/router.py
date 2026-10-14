"""Enterprise enquiry intake.

Public, unauthenticated, and deliberately narrow: an institution evaluating
DeepSentinel describes what they need and it reaches the team by email. It
creates no account and grants no access — provisioning stays a manual decision
made by an administrator after an actual conversation.

Every submission is persisted to the audit log as well as emailed, because an
SMTP failure should not silently lose an enquiry that someone took the trouble
to write.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

from backend import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/enquiry", tags=["enquiry"])

# A public endpoint that sends mail is a spam vector, so cap it per client.
_RATE_WINDOW = 3600.0
_RATE_MAX = 3
_seen: dict[str, deque] = defaultdict(deque)

ORG_TYPES = {
    "bank", "credit_union", "payment_processor", "fintech",
    "regulator", "consultancy", "other",
}


class Enquiry(BaseModel):
    organisation: str = Field(min_length=2, max_length=200)
    org_type: str = Field(default="bank")
    country: str = Field(default="", max_length=100)
    contact_name: str = Field(min_length=2, max_length=120)
    job_title: str = Field(default="", max_length=120)
    work_email: EmailStr
    phone: str = Field(default="", max_length=40)
    monthly_volume: str = Field(default="", max_length=60)
    interests: list[str] = Field(default_factory=list, max_length=10)
    timeline: str = Field(default="", max_length=60)
    message: str = Field(default="", max_length=4000)
    # Honeypot: a real person never sees this field, so anything in it is a bot.
    website: str = Field(default="", max_length=200)


def _rate_limited(ip: str) -> bool:
    now = time.time()
    hits = _seen[ip]
    while hits and now - hits[0] > _RATE_WINDOW:
        hits.popleft()
    if len(hits) >= _RATE_MAX:
        return True
    hits.append(now)
    return False


def _suggested_role(e: Enquiry) -> str:
    """A starting point for provisioning, not a decision.

    Whoever runs the call decides the real role; this just saves the admin
    re-reading the enquiry to guess where to begin.
    """
    title = (e.job_title or "").lower()
    if any(w in title for w in ("head", "chief", "director", "vp", "lead", "manager")):
        return "risk_manager"
    return "analyst"


def _format(e: Enquiry) -> str:
    interests = ", ".join(e.interests) or "not specified"
    return "\n".join([
        "New DeepSentinel enquiry",
        "=" * 40,
        f"Organisation : {e.organisation}",
        f"Type         : {e.org_type}",
        f"Country      : {e.country or '—'}",
        f"Volume       : {e.monthly_volume or '—'}",
        "",
        f"Contact      : {e.contact_name}",
        f"Title        : {e.job_title or '—'}",
        f"Email        : {e.work_email}",
        f"Phone        : {e.phone or '—'}",
        "",
        f"Interested in: {interests}",
        f"Timeline     : {e.timeline or '—'}",
        "",
        "Message:",
        e.message or "(none)",
        "",
        "-" * 40,
        "Next step: no account has been created by this enquiry.",
        f"After the call, provision access in Users → Create user.",
        f"Suggested starting role: {_suggested_role(e)}",
    ])


@router.post("", status_code=202)
async def submit(e: Enquiry, request: Request) -> dict:
    if e.website:
        # Silently accept so a bot gets no signal about what tripped it.
        logger.info("Enquiry honeypot triggered; discarding")
        return {"status": "received"}

    ip = request.client.host if request.client else "unknown"
    if _rate_limited(ip):
        raise HTTPException(429, "Too many enquiries from this address. Try again later.")

    body = _format(e)
    logger.info("Enquiry from %s (%s)", e.organisation, e.work_email)

    # Record it first: mail can fail, and losing the enquiry is the worse
    # outcome of the two.
    try:
        from backend.auth import audit

        await audit(
            "enquiry.received",
            actor=str(e.work_email),
            target=e.organisation,
            detail=body[:900],
            client_ip=ip,
        )
    except Exception as exc:                           # noqa: BLE001
        logger.warning(f"Could not audit enquiry: {exc}")

    delivered = False
    try:
        from backend.email_service import _provider, _send_plain  # type: ignore

        provider, _ = _provider()
        if provider:
            # The team's own inbox: the authenticated SMTP mailbox, which is
            # guaranteed to exist. sender_email is a display From and may be a
            # placeholder domain that bounces.
            from backend.email_service import _provider as _prov

            _, cfg = _prov()
            recipients = [cfg.get("username")] if cfg.get("username") else []
            delivered = _send_plain(
                subject=f"DeepSentinel enquiry — {e.organisation}",
                body=body,
                recipients=recipients,
                reply_to=str(e.work_email),
            )
    except Exception as exc:                           # noqa: BLE001
        logger.warning(f"Enquiry email not sent: {exc}")

    # 202 either way: the enquiry is recorded, and telling a prospective
    # customer "our mail server is down" helps nobody.
    return {"status": "received", "emailed": delivered}

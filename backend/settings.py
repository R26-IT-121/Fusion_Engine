"""
Risk manager recipients and alert thresholds.

Database-backed so the same configuration is seen from every machine — a risk
manager added on one device receives alerts regardless of where analysis runs.
"""

import logging
from typing import Optional

from fastapi import HTTPException
from sqlalchemy import func, select

from backend.db.models import AlertSettings, RiskManager
from backend.db.session import get_session

logger = logging.getLogger(__name__)


# ── Alert settings (single row) ──────────────────────────────────────────────


async def get_alert_settings() -> AlertSettings:
    async with get_session() as db:
        settings = await db.scalar(select(AlertSettings).where(AlertSettings.id == 1))
        if settings is None:
            settings = AlertSettings(id=1)
            db.add(settings)
            await db.flush()
            await db.refresh(settings)
        return settings


async def update_alert_settings(changes: dict, actor: Optional[str] = None) -> AlertSettings:
    """Apply a partial update. Unknown keys are rejected rather than ignored, so
    a typo surfaces instead of silently doing nothing."""
    allowed = {
        "fraud_threshold",
        "include_low_risk",
        "include_medium_risk",
        "include_high_risk",
        "include_critical_risk",
        "send_to_all",
        "backend_url",
    }
    unknown = set(changes) - allowed
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown setting(s): {', '.join(sorted(unknown))}. "
                   f"Valid: {', '.join(sorted(allowed))}",
        )

    if "fraud_threshold" in changes:
        try:
            threshold = float(changes["fraud_threshold"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="fraud_threshold must be a number")
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(
                status_code=422, detail="fraud_threshold must be between 0 and 1"
            )
        changes["fraud_threshold"] = threshold

    async with get_session() as db:
        settings = await db.scalar(select(AlertSettings).where(AlertSettings.id == 1))
        if settings is None:
            settings = AlertSettings(id=1)
            db.add(settings)

        for key, value in changes.items():
            setattr(settings, key, value)
        await db.flush()
        await db.refresh(settings)

    logger.info(f"Alert settings updated by {actor}: {sorted(changes)}")
    return settings


# ── Risk managers ────────────────────────────────────────────────────────────


async def list_risk_managers() -> list[RiskManager]:
    async with get_session() as db:
        result = await db.scalars(select(RiskManager).order_by(RiskManager.created_at))
        return list(result)


async def add_risk_manager(
    name: str, email: str, role: str = "Risk Manager"
) -> RiskManager:
    email = email.strip().lower()
    async with get_session() as db:
        existing = await db.scalar(
            select(RiskManager).where(func.lower(RiskManager.email) == email)
        )
        if existing is not None:
            raise HTTPException(
                status_code=409, detail=f"{email} is already receiving alerts"
            )

        manager = RiskManager(name=name.strip(), email=email, role=role)
        db.add(manager)
        await db.flush()
        await db.refresh(manager)

    logger.info(f"Risk manager added: {email}")
    return manager


async def remove_risk_manager(email: str) -> None:
    email = email.strip().lower()
    async with get_session() as db:
        manager = await db.scalar(
            select(RiskManager).where(func.lower(RiskManager.email) == email)
        )
        if manager is None:
            raise HTTPException(status_code=404, detail=f"{email} is not on the alert list")
        await db.delete(manager)

    logger.info(f"Risk manager removed: {email}")


async def set_risk_manager_enabled(email: str, enabled: bool) -> RiskManager:
    email = email.strip().lower()
    async with get_session() as db:
        manager = await db.scalar(
            select(RiskManager).where(func.lower(RiskManager.email) == email)
        )
        if manager is None:
            raise HTTPException(status_code=404, detail=f"{email} is not on the alert list")
        manager.enabled = enabled
        await db.flush()
        await db.refresh(manager)
        return manager


async def get_alert_recipients() -> list[str]:
    """Email addresses that should receive a fraud alert right now."""
    async with get_session() as db:
        result = await db.scalars(
            select(RiskManager.email).where(RiskManager.enabled.is_(True))
        )
        return list(result)


async def should_alert(classification: str) -> bool:
    """Whether this classification is configured to trigger an alert."""
    settings = await get_alert_settings()
    return {
        "CRITICAL": settings.include_critical_risk,
        "HIGH": settings.include_high_risk,
        "MEDIUM": settings.include_medium_risk,
        "LOW": settings.include_low_risk,
    }.get(classification, False)


async def get_backend_url() -> str:
    settings = await get_alert_settings()
    return settings.backend_url

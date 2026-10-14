"""Development launcher for the two assistants — no RAG stack required.

The full application (`backend.main`) loads the FATF vector store at startup,
which needs chromadb and sentence-transformers. Neither assistant uses that
store, so this launcher brings up only what they do need — auth, the database
and the two routers — letting you demo both without installing the heavy
dependencies.

**Development only.** The real deployment runs `backend.main:app`, where both
routers are already mounted.

Usage:
    pip install fastapi uvicorn pydantic sqlalchemy aiosqlite bcrypt pyjwt \
                email-validator python-multipart
    GRAPH_API_BASE=http://localhost:8000 python serve_bots_dev.py

Then sign in with the bootstrap admin (default admin / admin123, or whatever
ADMIN_BOOTSTRAP_PASSWORD is set to) and open /assistant in the web app.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("bots-dev")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.auth import ensure_bootstrap_admin
    from backend.db.session import init_db

    await init_db()
    await ensure_bootstrap_admin()
    logger.info("Database ready; bootstrap admin ensured.")

    # Turn the assistant on so it is visible out of the box in development.
    # In production this stays off until an admin enables it.
    from assistant import entitlement

    if not entitlement.load().enabled:
        entitlement.save({"enabled": True})
        logger.info("Operator assistant enabled for development.")
    yield


app = FastAPI(title="DeepSentinel Assistants (dev)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

from assistant import router as assistant_router  # noqa: E402
from chatbot import router as chatbot_router  # noqa: E402

app.include_router(chatbot_router)
app.include_router(assistant_router)

from enquiry import router as enquiry_router  # noqa: E402

app.include_router(enquiry_router)

from monitor import router as monitor_router  # noqa: E402

app.include_router(monitor_router)


# The real app defines auth inline in backend/main.py rather than as a router,
# so the two endpoints the web client needs to sign in are re-declared here
# using the same underlying functions — no duplicated auth logic.
from fastapi import Depends, Request  # noqa: E402

from backend.auth import LoginRequest, UserOut, get_current_user  # noqa: E402
from backend.db.models import User  # noqa: E402


@app.post("/auth/login", tags=["auth"])
async def login(req: LoginRequest, request: Request) -> dict:
    from backend.auth import authenticate_user, create_access_token

    client_ip = request.client.host if request.client else None
    user = await authenticate_user(req.username, req.password, client_ip=client_ip)
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "expires_in": 8 * 3600,
        "user": {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


@app.get("/auth/me", tags=["auth"])
async def me(user: User = Depends(get_current_user)):
    return UserOut.from_model(user)


@app.post("/auth/logout", tags=["auth"])
async def logout(_: User = Depends(get_current_user)) -> dict:
    return {"status": "logged_out"}


# ── Settings ────────────────────────────────────────────────────────────────
# The real app defines these inline in backend/main.py. Re-declared here over
# the same backend.settings functions so the Settings page works in the dev
# app — without them the page 404s and the recipient form silently fails.

from pydantic import BaseModel, EmailStr  # noqa: E402


class RiskManagerRequest(BaseModel):
    name: str
    email: EmailStr
    role: str = "Risk Manager"


@app.get("/settings", tags=["settings"])
async def get_settings(_: User = Depends(get_current_user)) -> dict:
    from backend.settings import get_alert_settings, list_risk_managers

    managers = await list_risk_managers()
    alerts = await get_alert_settings()
    return {
        "risk_managers": [
            {"name": m.name, "email": m.email, "role": m.role, "enabled": m.enabled}
            for m in managers
        ],
        "alert_settings": {
            "fraud_threshold": alerts.fraud_threshold,
            "include_low_risk": alerts.include_low_risk,
            "include_medium_risk": alerts.include_medium_risk,
            "include_high_risk": alerts.include_high_risk,
            "include_critical_risk": alerts.include_critical_risk,
            "send_to_all": alerts.send_to_all,
        },
        "backend_url": alerts.backend_url,
    }


@app.post("/settings/risk-manager", status_code=201, tags=["settings"])
async def add_risk_manager_endpoint(
    req: RiskManagerRequest, user: User = Depends(get_current_user)
) -> dict:
    from backend.settings import add_risk_manager

    manager = await add_risk_manager(name=req.name, email=req.email, role=req.role)
    return {"status": "added", "email": manager.email}


@app.delete("/settings/risk-manager/{email}", tags=["settings"])
async def remove_risk_manager_endpoint(
    email: str, user: User = Depends(get_current_user)
) -> dict:
    from backend.settings import remove_risk_manager

    await remove_risk_manager(email)
    return {"status": "removed", "email": email}


@app.post("/settings/alert-settings", tags=["settings"])
async def update_alert_settings_endpoint(
    settings: dict, user: User = Depends(get_current_user)
) -> dict:
    from backend.settings import update_alert_settings

    updated = await update_alert_settings(settings, actor=user.username)
    return {"status": "updated", "fraud_threshold": updated.fraud_threshold}


@app.post("/email/send-test", tags=["email"])
async def send_test_email(
    req: RiskManagerRequest, _: User = Depends(get_current_user)
) -> dict:
    """Send a sample alert so an operator can prove delivery works."""
    from datetime import datetime

    from backend.email_service import FraudAlert, SendOutcome, send_fraud_alert
    from backend.settings import get_backend_url
    from fastapi import HTTPException

    alert = FraudAlert(
        transaction_id="TEST_TX_001",
        fraud_confidence=0.87,
        classification="HIGH",
        timestamp=datetime.now().isoformat(),
        graph_score=0.85,
        behavioral_score=0.88,
        temporal_score=0.90,
        graph_signal="Graph pattern: HUB_AND_SPOKE. Convergence count: 3 distinct senders.",
        behavioral_signal="High reconstruction error in spending pattern. DSAA score: 0.88",
        temporal_signal="Burstiness coefficient 0.92 — machine-paced activity.",
        forensic_report=(
            "Test alert from DeepSentinel. If you are reading this, email "
            "delivery is configured correctly."
        ),
        typology_name="Mule Network - Hub and Spoke",
        typology_id="TY_001_MULE",
    )

    result = await send_fraud_alert(alert, [req.email], backend_url=await get_backend_url())

    # 409, not 500: nothing is broken, the server simply is not set up to send.
    if result.outcome is SendOutcome.NOT_CONFIGURED:
        raise HTTPException(status_code=409, detail=result.detail)
    if result.outcome is SendOutcome.FAILED:
        raise HTTPException(status_code=502, detail=result.detail)

    return {
        "status": "sent",
        "recipient": req.email,
        "provider": result.provider,
        "note": "Check the spam folder if it does not arrive within a minute.",
    }


@app.get("/email-template/preview", tags=["email"])
async def email_template_preview(classification: str = "HIGH"):
    """Render the alert template so the page can show what recipients see."""
    from datetime import datetime

    from backend.email_service import FraudAlert, build_email_html
    from fastapi.responses import HTMLResponse

    alert = FraudAlert(
        transaction_id="PREVIEW_TX",
        fraud_confidence={"CRITICAL": 0.96, "HIGH": 0.87, "MEDIUM": 0.62, "LOW": 0.21}
        .get(classification.upper(), 0.87),
        classification=classification.upper(),
        timestamp=datetime.now().isoformat(),
        graph_score=0.85, behavioral_score=0.88, temporal_score=0.90,
        graph_signal="Graph pattern: HUB_AND_SPOKE. Convergence count: 3 distinct senders.",
        behavioral_signal="High reconstruction error in spending pattern.",
        temporal_signal="Burstiness coefficient 0.92 — machine-paced activity.",
        forensic_report="Preview of the alert an analyst receives.",
        typology_name="Mule Network - Hub and Spoke",
        typology_id="TY_001_MULE",
    )
    return HTMLResponse(build_email_html(alert, "http://localhost:8090"))


@app.get("/email/status", tags=["email"])
async def email_status(_: User = Depends(get_current_user)) -> dict:
    """Which mail provider is configured, so the page can say so honestly."""
    from backend.email_service import _provider

    provider, _cfg = _provider()
    return {"configured": bool(provider), "provider": provider}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "assistants-dev"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8090")))

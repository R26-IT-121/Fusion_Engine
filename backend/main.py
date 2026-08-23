"""
DeepSentinel — Fusion Engine & Generative Explainability API
FastAPI orchestration layer. Handles:
  - Async parallel calls to upstream graph/behavioral/temporal model APIs
  - Graceful degradation if upstream models time out
  - Meta-classifier fusion
  - RAG retrieval from FATF knowledge base
  - LLM forensic report generation
  - Mock score fallback for demo/testing
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

from backend import config
from backend.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    LoginRequest,
    PasswordChange,
    UserCreate,
    UserOut,
    get_current_user,
    require_admin,
    require_manager,
)
from backend.db.models import User

# .env is still read so environment variables set there override config.ini,
# which keeps existing docker-compose and platform deploys working unchanged.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("deepsentinel")

# --- Config: environment > config.ini > default (see backend/config.py) ---
CHROMA_DB_PATH = config.get("paths", "chroma_db")
FATF_DATA_PATH = config.get("paths", "fatf_data")
MODEL_SAVE_PATH = config.get("paths", "meta_classifier")

# Upstream base URLs — adapters append the correct path per model
BEHAVIORAL_API_BASE = config.get("upstream", "behavioral_api_base")  # M1 VAE
GRAPH_API_BASE = config.get("upstream", "graph_api_base")            # M2 GraphSAGE
TEMPORAL_API_BASE = config.get("upstream", "temporal_api_base")      # M3 TCN

UPSTREAM_TIMEOUT = config.get("upstream", "timeout_ms") / 1000.0

# --- Lazy-initialized singletons ---
knowledge_base = None
retriever = None
meta_classifier = None
forensic_reporter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global knowledge_base, retriever, meta_classifier, forensic_reporter

    from backend.rag.knowledge_base import FATFKnowledgeBase
    from backend.rag.retriever import FATFRetriever
    from backend.fusion_engine import MetaClassifier
    from backend.llm.forensic_reporter import ForensicReporter, create_llm_backend

    logger.info("=== DeepSentinel Fusion Engine — Starting Up ===")

    # Fail fast on a misconfigured deploy rather than at the first request.
    strict = config.is_production()
    problems = config.validate(strict=strict)
    if problems:
        for p in problems:
            logger.error(f"Configuration error: {p}")
        raise RuntimeError(
            f"{len(problems)} configuration error(s) — refusing to start in "
            f"production. See errors above."
        )
    logger.info(config.describe())

    logger.info("Connecting to database...")
    from backend.auth import ensure_bootstrap_admin
    from backend.db.session import init_db

    await init_db()
    await ensure_bootstrap_admin()

    logger.info("Initializing FATF Knowledge Base...")
    knowledge_base = FATFKnowledgeBase(
        chroma_db_path=CHROMA_DB_PATH,
        fatf_data_path=FATF_DATA_PATH,
    )
    knowledge_base.initialize()

    retriever = FATFRetriever(
        collection=knowledge_base.get_collection(),
        embedder=knowledge_base.get_embedder(),
        top_k=1,
    )

    logger.info("Initializing Meta Classifier...")
    meta_classifier = MetaClassifier(model_save_path=MODEL_SAVE_PATH)
    meta_classifier.initialize()

    logger.info("Initializing LLM backend...")
    try:
        llm_backend = create_llm_backend()
        forensic_reporter = ForensicReporter(backend=llm_backend)
        logger.info("LLM backend ready.")
    except ValueError as e:
        logger.warning(f"LLM backend not configured: {e}. Reports will be unavailable.")
        forensic_reporter = None

    logger.info("=== DeepSentinel ready. ===")
    yield

    from backend.db.session import close_db

    await close_db()
    logger.info("DeepSentinel shutting down.")


app = FastAPI(
    title="DeepSentinel — Fusion Engine & Generative Explainability",
    description=(
        "Weighted ensemble meta-classifier + RAG-grounded LLM forensic reporting "
        "for the DeepSentinel multi-modal fraud detection platform. Member 4 — IT22192882."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS. The wildcard is fine for local development, but a deployment that
# accepts bearer tokens should name its frontend origin explicitly — otherwise
# any page on the internet can call this API with a victim's token.
_cors_raw = str(config.get("auth", "cors_origins")).strip()
_cors_origins = (
    ["*"] if _cors_raw == "*" else [o.strip() for o in _cors_raw.split(",") if o.strip()]
)

if _cors_origins == ["*"] and config.is_production():
    logger.warning(
        "CORS is set to '*' in production. Set CORS_ORIGINS to your frontend "
        "origin so other sites cannot call this API with a user's token."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class TransactionData(BaseModel):
    """Full PaySim-style transaction — forwarded to upstream model APIs."""
    step: int = Field(ge=1, le=744, description="PaySim simulation hour (1–744)")
    type: str = Field(description="TRANSFER | CASH_OUT | CASH_IN | PAYMENT | DEBIT")
    amount: float = Field(ge=0)
    nameOrig: str
    nameDest: str
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    isFlaggedFraud: int = Field(default=0, ge=0, le=1)


class AnalyzeRequest(BaseModel):
    transaction_id: Optional[str] = Field(
        default=None,
        description="Transaction identifier. Auto-generated UUID if omitted.",
    )
    transaction: Optional[TransactionData] = Field(
        default=None,
        description=(
            "Full PaySim transaction data. When provided, forwarded to each upstream "
            "model API so they can run their own analysis. Takes priority over direct scores."
        ),
    )
    graph_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="GraphSAGE fraud probability (0–1). Used only when transaction is omitted.",
    )
    behavioral_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="VAE behavioral anomaly score (0–1). Used only when transaction is omitted.",
    )
    temporal_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="TCN temporal anomaly score (0–1). Used only when transaction is omitted.",
    )
    use_mock: bool = Field(
        default=False,
        description="Force mock score generator (ignores all scores and transaction data).",
    )
    mock_scenario: Optional[str] = Field(
        default=None,
        description=(
            "Mock scenario: smurfing | layering | mule_network | "
            "account_takeover | velocity_fraud | legitimate. Defaults to random."
        ),
    )
    include_baseline: bool = Field(
        default=False,
        description=(
            "Also generate an ungrounded baseline report (no FATF context) "
            "for ablation / novelty demonstration."
        ),
    )


class RetrievalInfo(BaseModel):
    typology_id: str
    typology_name: str
    stage: str
    risk_level: str
    similarity_score: float


class AnalyzeResponse(BaseModel):
    transaction_id: str
    fraud_confidence_score: float
    classification: str
    graph_score: float
    behavioral_score: float
    temporal_score: float
    graph_available: bool
    behavioral_available: bool
    temporal_available: bool
    modalities_used: int
    retrieval: RetrievalInfo
    forensic_report: Optional[str]
    baseline_report: Optional[str]
    mock_scenario: Optional[str]
    # Rich upstream signals (populated when transaction data provided)
    behavioral_signal: Optional[str] = None
    graph_signal: Optional[str] = None
    temporal_signal: Optional[str] = None


# ── Upstream callers ──────────────────────────────────────────────────────────

async def _fetch_from_upstream_apis(
    transaction: TransactionData,
    transaction_id: str,
) -> tuple:
    """
    Call all three upstream model APIs in parallel using their correct schemas.
    Returns (behavioral_resp, graph_resp, temporal_resp) — all UpstreamResponse.
    """
    from backend.adapters.upstream import (
        call_behavioral_api,
        call_graph_api,
        call_temporal_api,
    )

    tx_dict = transaction.model_dump()
    tx_dict["transaction_id"] = transaction_id  # GraphSAGE expects top-level transaction_id

    async with httpx.AsyncClient() as client:
        b_task = call_behavioral_api(client, BEHAVIORAL_API_BASE, tx_dict, UPSTREAM_TIMEOUT)
        g_task = call_graph_api(client, GRAPH_API_BASE, tx_dict, UPSTREAM_TIMEOUT)
        t_task = call_temporal_api(client, TEMPORAL_API_BASE, tx_dict, UPSTREAM_TIMEOUT)

        behavioral, graph, temporal = await asyncio.gather(b_task, g_task, t_task)

    return behavioral, graph, temporal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _classify(confidence: float) -> str:
    if confidence >= 0.80:
        return "CRITICAL"
    if confidence >= 0.65:
        return "HIGH"
    if confidence >= 0.50:
        return "MEDIUM"
    return "LOW"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "knowledge_base": knowledge_base is not None,
        "meta_classifier": meta_classifier is not None,
        "llm_reporter": forensic_reporter is not None,
        "upstream_bases": {
            "behavioral": BEHAVIORAL_API_BASE,
            "graph": GRAPH_API_BASE,
            "temporal": TEMPORAL_API_BASE,
        },
    }


@app.get("/typologies")
async def list_typologies():
    """Return all FATF typologies stored in the knowledge base."""
    collection = knowledge_base.get_collection()
    results = collection.get(include=["metadatas"])
    return {
        "count": len(results["ids"]),
        "typologies": [
            {"id": tid, **meta}
            for tid, meta in zip(results["ids"], results["metadatas"])
        ],
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Full pipeline: obtain scores → fuse → retrieve FATF typology → generate forensic report.

    Score acquisition priority:
      1. use_mock=True  → mock generator
      2. transaction    → call all three upstream APIs with full transaction data
      3. direct scores  → use provided graph_score / behavioral_score / temporal_score
      4. fallback       → mock generator (all scores missing)
    """
    # Drains the shared pipeline generator and returns its terminal result, so
    # this endpoint and /analyze/stream cannot diverge.
    from backend.pipeline import PipelineResult, Stage, Status, StageEvent, run_pipeline

    async for event in run_pipeline(
        request,
        meta_classifier=meta_classifier,
        retriever=retriever,
        forensic_reporter=forensic_reporter,
        fetch_upstream=_fetch_from_upstream_apis,
        classify=_classify,
    ):
        if isinstance(event, PipelineResult):
            return AnalyzeResponse(**event.payload)

        if isinstance(event, StageEvent) and event.status == Status.ERROR:
            # A bad scenario name is the caller's mistake; anything else is ours.
            status = 400 if event.stage == Stage.MODELS else 500
            if event.stage != Stage.REPORT:
                raise HTTPException(status_code=status, detail=event.message)

    raise HTTPException(status_code=500, detail="Pipeline produced no result.")


@app.post("/analyze/stream", tags=["analysis"])
async def analyze_stream(request: AnalyzeRequest):
    """
    Run the same analysis, emitting each stage as it completes.

    Server-sent events. Every stage carries its measured duration, so the client
    renders what actually happened rather than an animation timed to look busy.

        event: stage     one per stage transition (running → done/error/skipped)
        event: complete  the assembled result, identical to POST /analyze
        event: error     the pipeline could not continue
    """
    from fastapi.responses import StreamingResponse

    from backend.pipeline import PipelineResult, Status, StageEvent, run_pipeline

    async def emit() -> AsyncIterator[str]:
        def sse(event: str, payload: dict) -> str:
            # json.dumps, not str(): a stray newline inside a value would
            # otherwise terminate the event early and corrupt the stream.
            return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"

        try:
            async for item in run_pipeline(
                request,
                meta_classifier=meta_classifier,
                retriever=retriever,
                forensic_reporter=forensic_reporter,
                fetch_upstream=_fetch_from_upstream_apis,
                classify=_classify,
            ):
                if isinstance(item, PipelineResult):
                    yield sse("complete", item.payload)
                elif isinstance(item, StageEvent):
                    yield sse("stage", item.to_dict())
        except Exception as e:
            logger.exception("Streaming pipeline failed")
            yield sse("error", {"message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        emit(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Without this, nginx buffers the whole response and every event
            # arrives at once — which would defeat the point.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/retrain")
async def retrain_classifier():
    """Force retrain the meta-classifier (use after upstream models are recalibrated)."""
    meta_classifier.retrain()
    return {"status": "retrained"}


@app.post("/rebuild-kb")
async def rebuild_knowledge_base():
    """Force rebuild the FATF ChromaDB knowledge base (use after updating typologies)."""
    knowledge_base.rebuild()
    global retriever
    from backend.rag.retriever import FATFRetriever
    retriever = FATFRetriever(
        collection=knowledge_base.get_collection(),
        embedder=knowledge_base.get_embedder(),
        top_k=1,
    )
    return {"status": "knowledge base rebuilt"}


# ──────────────────────────────────────────────────────────────────────────────
# SETTINGS & EMAIL MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────


class RiskManagerRequest(BaseModel):
    name: str
    email: EmailStr
    role: str = "Risk Manager"


@app.get("/settings", tags=["settings"])
async def get_settings(user: User = Depends(require_manager)):
    """Read risk managers and alert thresholds. Admin or risk manager."""
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
    req: RiskManagerRequest, user: User = Depends(require_manager)
):
    """Add a fraud alert recipient. Admin or risk manager."""
    from backend.auth import audit
    from backend.settings import add_risk_manager

    manager = await add_risk_manager(name=req.name, email=req.email, role=req.role)
    await audit("risk_manager.add", actor=user.username, target=manager.email)
    return {"status": "added", "email": manager.email}


@app.delete("/settings/risk-manager/{email}", tags=["settings"])
async def remove_risk_manager_endpoint(
    email: str, user: User = Depends(require_manager)
):
    """Remove a fraud alert recipient. Admin or risk manager."""
    from backend.auth import audit
    from backend.settings import remove_risk_manager

    await remove_risk_manager(email)
    await audit("risk_manager.remove", actor=user.username, target=email)
    return {"status": "removed", "email": email}


@app.post("/settings/alert-settings", tags=["settings"])
async def update_alert_settings_endpoint(
    settings: dict, user: User = Depends(require_manager)
):
    """Update alert thresholds. Admin or risk manager."""
    from backend.auth import audit
    from backend.settings import update_alert_settings

    updated = await update_alert_settings(settings, actor=user.username)
    await audit(
        "settings.update", actor=user.username, detail=f"keys={sorted(settings)}"
    )
    return {
        "status": "updated",
        "alert_settings": {
            "fraud_threshold": updated.fraud_threshold,
            "include_low_risk": updated.include_low_risk,
            "include_medium_risk": updated.include_medium_risk,
            "include_high_risk": updated.include_high_risk,
            "include_critical_risk": updated.include_critical_risk,
            "send_to_all": updated.send_to_all,
        },
    }


@app.post("/settings/backend-url", tags=["settings"])
async def update_backend_url(payload: dict, user: User = Depends(require_admin)):
    """Set the dashboard URL used in alert email links. Admin only."""
    from backend.settings import update_alert_settings

    updated = await update_alert_settings(
        {"backend_url": payload.get("url", "http://localhost:8000")},
        actor=user.username,
    )
    return {"status": "updated", "backend_url": updated.backend_url}


@app.get("/email-template/preview")
async def preview_email_template(classification: str = "HIGH"):
    """Preview email template (returns HTML)."""
    from backend.email_service import FraudAlert, build_email_html
    from datetime import datetime
    from fastapi.responses import HTMLResponse

    test_alert = FraudAlert(
        transaction_id="PREVIEW_TX_001",
        fraud_confidence=0.75 if classification == "MEDIUM" else 0.87,
        classification=classification,
        timestamp=datetime.now().isoformat(),
        graph_score=0.85,
        behavioral_score=0.88,
        temporal_score=0.90,
        graph_signal="Graph pattern: HUB_AND_SPOKE. Convergence count: 3 distinct senders. Fresh sender ratio: 66.7%.",
        behavioral_signal="Anomaly fingerprint - Signal 1: High reconstruction error in transaction velocity. Signal 2: KL divergence indicates unusual feature distribution.",
        temporal_signal="Step burstiness coefficient: 0.92 (significantly elevated). Triggering predecessor detected in 12 transactions.",
        forensic_report="This transaction exhibits multiple correlated fraud signals across all modalities. The network analysis reveals a hub-and-spoke pattern typical of money mule operations. Behavioral analysis detects anomalous reconstruction errors suggesting coordinated activity. Temporal analysis shows elevated burstiness indicating rapid, automated transfers. FATF classification: MULE_NETWORK. Recommended action: FLAG_FOR_REVIEW.",
        typology_name="Mule Network - Hub and Spoke",
        typology_id="TY_001_MULE",
    )

    from backend.settings import get_backend_url

    html = build_email_html(test_alert, await get_backend_url())
    return HTMLResponse(content=html)


@app.post("/email/send-test", tags=["email"])
async def send_test_email(
    req: RiskManagerRequest, user: User = Depends(require_manager)
):
    """Send a test fraud alert to verify email delivery. Admin or risk manager."""
    from backend.email_service import send_fraud_alert, FraudAlert
    from datetime import datetime

    test_alert = FraudAlert(
        transaction_id="TEST_TX_001",
        fraud_confidence=0.87,
        classification="HIGH",
        timestamp=datetime.now().isoformat(),
        graph_score=0.85,
        behavioral_score=0.88,
        temporal_score=0.90,
        graph_signal="Graph pattern: HUB_AND_SPOKE. Convergence count: 3 distinct senders.",
        behavioral_signal="High reconstruction error detected in spending patterns. DSAA score: 0.88",
        temporal_signal="Step burstiness coefficient: 0.92 (high velocity activity). Triggering predecessor detected.",
        forensic_report="This transaction exhibits multiple fraud signals: Hub-and-spoke network pattern in sender graph, anomalous behavioral reconstruction error, and high temporal burstiness. Combined risk confidence 87%. Recommended action: BLOCK_TRANSACTION.",
        typology_name="Mule Network - Hub and Spoke",
        typology_id="TY_001_MULE",
    )

    from backend.settings import get_backend_url

    success = await send_fraud_alert(
        test_alert, [req.email], backend_url=await get_backend_url()
    )
    if not success:
        raise HTTPException(
            status_code=502,
            detail="Email provider rejected the send. Check the SendGrid API key "
                   "and that the sender address is verified.",
        )
    return {
        "status": "sent",
        "recipient": req.email,
        "note": "Check the spam folder if it does not arrive.",
    }


# ──────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION & USER MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/auth/login", tags=["auth"])
async def login(req: LoginRequest, request: Request):
    """Exchange credentials for a JWT."""
    from backend.auth import authenticate_user, create_access_token

    client_ip = request.client.host if request.client else None
    user = await authenticate_user(req.username, req.password, client_ip=client_ip)
    token = create_access_token(user)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


@app.get("/auth/me", tags=["auth"])
async def get_me(user: User = Depends(get_current_user)):
    """Current user."""
    return UserOut.from_model(user)


@app.post("/auth/logout", tags=["auth"])
async def logout(user: User = Depends(get_current_user)):
    """Record a logout. The client discards the token; JWTs are stateless."""
    from backend.auth import audit

    await audit("auth.logout", actor=user.username)
    return {"status": "logged_out"}


@app.post("/auth/change-password", tags=["auth"])
async def change_own_password(
    req: PasswordChange, user: User = Depends(get_current_user)
):
    """Change your own password. Invalidates existing sessions."""
    from backend.auth import change_password

    await change_password(user.username, req.current_password, req.new_password)
    return {"status": "changed", "note": "Sign in again with the new password."}


# ── User administration (admin only) ─────────────────────────────────────────


@app.get("/users", tags=["users"])
async def list_users_endpoint(user: User = Depends(require_admin)):
    """List all users. Admin only."""
    from backend.auth import list_users

    return [UserOut.from_model(u) for u in await list_users()]


@app.post("/users", status_code=201, tags=["users"])
async def create_user_endpoint(req: UserCreate, user: User = Depends(require_admin)):
    """Create a user. Admin only."""
    from backend.auth import create_user

    created = await create_user(req, created_by=user.username)
    return UserOut.from_model(created)


@app.patch("/users/{username}/enabled", tags=["users"])
async def set_user_enabled_endpoint(
    username: str, payload: dict, user: User = Depends(require_admin)
):
    """Enable or disable a user. Admin only."""
    from backend.auth import set_user_enabled

    enabled = bool(payload.get("enabled", True))
    if username == user.username and not enabled:
        raise HTTPException(status_code=409, detail="You cannot disable your own account")

    await set_user_enabled(username, enabled, actor=user.username)
    return {"status": "updated", "username": username, "enabled": enabled}


@app.delete("/users/{username}", tags=["users"])
async def delete_user_endpoint(username: str, user: User = Depends(require_admin)):
    """Delete a user. Admin only. The last admin cannot be removed."""
    from backend.auth import delete_user

    if username == user.username:
        raise HTTPException(status_code=409, detail="You cannot delete your own account")

    await delete_user(username, actor=user.username)
    return {"status": "deleted", "username": username}


@app.get("/audit-log", tags=["users"])
async def get_audit_log(limit: int = 100, user: User = Depends(require_admin)):
    """Recent security events, newest first. Admin only."""
    from sqlalchemy import select

    from backend.db.models import AuditLog, as_utc
    from backend.db.session import get_session

    limit = max(1, min(limit, 500))
    async with get_session() as db:
        rows = await db.scalars(
            select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)
        )
        # as_utc: SQLite returns naive datetimes, and a naive ISO string is read
        # as local time by the browser, shifting every audit entry.
        return [
            {
                "timestamp": as_utc(r.timestamp),
                "actor": r.actor,
                "action": r.action,
                "target": r.target,
                "outcome": r.outcome,
                "client_ip": r.client_ip,
                "detail": r.detail,
            }
            for r in rows
        ]

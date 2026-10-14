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
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

from backend import config
from backend.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    LoginRequest,
    PasswordChange,
    ProfileUpdate,
    UserCreate,
    UserOut,
    get_current_user,
    require_admin,
    require_any_user,
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
    from backend.auth import ensure_bootstrap_users
    from backend.db.session import init_db

    await init_db()
    await ensure_bootstrap_users()

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
        "Multi-modal fraud detection. Three detectors score every transaction "
        "in parallel — the payment graph around it, how it fits its transaction "
        "type, and the run it arrived in — and a meta-classifier fuses them into "
        "one verdict with a cited forensic narrative.\n\n"
        "`/public/capabilities` needs no credentials and reports which detectors "
        "are answering. Everything else requires a bearer token from `/auth/login`."
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
    # PUT is used by the settings writes — thresholds, package tier, graph
    # settings, report style. Leaving it out let the preflight fail, and the
    # browser reports a blocked preflight as no response at all, so the
    # console said the backend was down while it was answering fine.
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Project assistant ─────────────────────────────────────────────────────────
# Grounded Q&A over the project's own documentation (see chatbot/README.md).
# Optional: a failure to import must not take the API down, so it is guarded.
try:
    from chatbot import router as chatbot_router

    app.include_router(chatbot_router)
    logger.info("Project assistant mounted at /api/chat")
except Exception as exc:  # noqa: BLE001
    logger.warning(f"Project assistant unavailable: {exc}")

# Operator assistant — tool-using agent over the live platform. Gated: disabled
# by default, admin-enabled, entitled roles only (see assistant/entitlement.py).
try:
    from assistant import router as assistant_router

    app.include_router(assistant_router)
    logger.info("Operator assistant mounted at /api/assistant")
except Exception as exc:  # noqa: BLE001
    logger.warning(f"Operator assistant unavailable: {exc}")

# Commercial enquiries. Public and unauthenticated — it creates nothing and
# grants nothing, it only routes a message to the team.
try:
    from enquiry import router as enquiry_router

    app.include_router(enquiry_router)
    logger.info("Enquiry intake mounted at /api/enquiry")
except Exception as exc:  # noqa: BLE001
    logger.warning(f"Enquiry intake unavailable: {exc}")

# Always-on transaction monitoring: the graph model screens everything and
# escalates only what looks structurally suspicious.
try:
    from monitor import router as monitor_router

    app.include_router(monitor_router)
    logger.info("Monitor mounted at /api/monitor")
except Exception as exc:  # noqa: BLE001
    logger.warning(f"Monitor unavailable: {exc}")


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
    # Signed per-detector contribution to the fused log-odds, summing
    # to z minus the intercept. Empty for an older saved model that
    # cannot be decomposed.
    contributions: dict[str, float] = {}
    retrieval: RetrievalInfo
    forensic_report: Optional[str]
    baseline_report: Optional[str]
    mock_scenario: Optional[str]
    # Rich upstream signals (populated when transaction data provided)
    behavioral_signal: Optional[str] = None
    graph_signal: Optional[str] = None
    # Novelty 3's forensic subgraph: which accounts are implicated, the sink,
    # the pattern, and per-edge attention weights. The evidence behind the score.
    graph_evidence: Optional[dict] = None
    # The behavioural detector's equivalent, and a different shape by nature:
    # a decomposition rather than a structure. Which stratum model answered, how
    # the three score terms combined, the per-feature and per-latent-dimension
    # attribution shares, and the discovered typology the fingerprint matched.
    behavioral_evidence: Optional[dict] = None
    # Primary key of the persisted record. The UI needs it to draft a SAR
    # against this exact result rather than re-deriving one from a re-run.
    analysis_id: Optional[int] = None
    temporal_signal: Optional[str] = None
    # The sequential detector's evidence: the current transaction's F1-F10
    # feature values plus the fraud_attention-identified triggering
    # predecessor (its own feature vector and attention weight) — the
    # temporal analogue of graph_evidence's subgraph and behavioral_evidence's
    # decomposition.
    temporal_evidence: Optional[dict] = None


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
    """Band a fused score, on the line the rest of the system uses.

    These cutoffs used to be written here as literals, which meant an operator
    could move the threshold on the settings page and every /analyze verdict
    would carry on being banded at the old one.
    """
    from backend import thresholds

    b = thresholds.current() or thresholds.DEFAULT_BANDS
    if confidence >= float(b["critical"]):
        return "CRITICAL"
    if confidence >= float(b["high"]):
        return "HIGH"
    if confidence >= float(b["medium"]):
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
            from backend.settings import record_analysis

            analysis_id = await record_analysis(
                event.payload,
                transaction=request.transaction.model_dump() if request.transaction else None,
            )
            return AnalyzeResponse(**event.payload, analysis_id=analysis_id)

        if isinstance(event, StageEvent) and event.status == Status.ERROR:
            # A bad scenario name is the caller's mistake; anything else is ours.
            status = 400 if event.stage == Stage.MODELS else 500
            if event.stage != Stage.REPORT:
                raise HTTPException(status_code=status, detail=event.message)

    raise HTTPException(status_code=500, detail="Pipeline produced no result.")


@app.get("/analyze/sample-transaction", tags=["analysis"])
async def sample_transaction():
    """One real transaction drawn from the graph service, ready to analyse.

    The analyzer used to offer only hand-written scenarios with simulated
    scores, which meant the page demonstrated the plumbing rather than the
    system. These are genuine PaySim records between genuine accounts — the
    same source the live monitor screens — so a run here exercises the real
    model on real input.
    """
    import httpx

    base = str(config.get("upstream", "graph_api_base")).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{base}/api/graph/sample-transactions", params={"n": 1})
            r.raise_for_status()
            txns = r.json().get("transactions") or []
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Graph service unreachable at {base}: {type(exc).__name__}",
        )

    if not txns:
        raise HTTPException(status_code=503, detail="Graph service returned no transactions.")

    txn = txns[0]
    # Ground truth is for measuring the system, never for showing as a model
    # output — strip it before it can reach the page.
    is_fraud = txn.pop("_is_fraud", None)
    return {"transaction": txn, "ground_truth_is_fraud": is_fraud}


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
                    from backend.settings import record_analysis

                    analysis_id = await record_analysis(
                        item.payload,
                        transaction=(
                            request.transaction.model_dump() if request.transaction else None
                        ),
                    )
                    yield sse("complete", {**item.payload, "analysis_id": analysis_id})
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


@app.post("/analyze/batch", tags=["analysis"])
async def analyze_batch(
    file: UploadFile = File(...),
    alert_threshold: float | None = Form(None),
    narrate_top: int = Form(3),
    user: User = Depends(get_current_user),
):
    """
    Score a whole file of transactions, streaming progress as it goes.

    Accepts CSV or Excel in the PaySim schema. An isFraud column, if present, is
    treated as ground truth and reported back as precision and recall — it never
    reaches the models.

    Narration is generated only for the `narrate_top` highest-scoring rows.

    `alert_threshold` defaults to the line the live monitor is alerting on right
    now — the operator's setting if one exists, otherwise the measured medium
    band. It used to default to a hardcoded 0.6, twenty times the medium band,
    so the same file scored here and screened live produced different verdicts
    from the same three models and the same fused score. A system that
    disagrees with itself about what counts as an alert cannot defend either
    number.

    It stays settable, because trying a different line against a labelled file
    is most of what a batch tool is for. The default is simply no longer a
    figure nobody chose.
    Producing one per transaction would take seconds each and turn a 300-row
    file into an hour-long job.

        event: meta      row count and whether labels were found
        event: progress  one per transaction, with its score
        event: summary   totals, and detection metrics when labels were present
        event: error     the file could not be processed
    """
    import time

    from fastapi.responses import StreamingResponse

    from backend.adapters.upstream import UpstreamResponse
    from backend.batch import (
        BatchError,
        BatchSummary,
        UpstreamCircuit,
        parse_file,
        update_summary,
    )

    raw = await file.read()

    async def emit() -> AsyncIterator[str]:
        def sse(event: str, payload: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"

        try:
            rows = parse_file(file.filename or "upload.csv", raw)
        except BatchError as e:
            yield sse("error", {"message": str(e)})
            return
        except Exception as e:
            logger.exception("Batch parse failed")
            yield sse("error", {"message": f"The file could not be read: {type(e).__name__}"})
            return

        labelled = sum(1 for r in rows if r.is_fraud_label is not None)

        # Resolved here rather than in the signature: a Form default has to be
        # a constant, and this one has to follow whatever the operator set.
        from backend import thresholds as _thr

        live_bands = _thr.current() or _thr.DEFAULT_BANDS
        # A distinct name on purpose. Assigning to `alert_threshold` here made
        # it a local of this nested generator, which shadowed the enclosing
        # parameter and raised UnboundLocalError on the read above it.
        if alert_threshold is None:
            threshold = float(live_bands["medium"])
            threshold_source = "live"
        else:
            threshold = float(alert_threshold)
            threshold_source = "custom"

        yield sse(
            "meta",
            {
                "filename": file.filename,
                "rows": len(rows),
                "labelled": labelled,
                "has_labels": labelled > 0,
                "alert_threshold": threshold,
                "threshold_source": threshold_source,
                "live_bands": live_bands,
                # So the per-detector rows are measured at each model's own
                # operating point rather than a copy kept in the browser.
                "own_thresholds": await detector_own_thresholds(),
            },
        )

        summary = BatchSummary(total=len(rows))
        summary.has_labels = labelled > 0
        scored: list[dict] = []
        started = time.perf_counter()

        # Stops re-dialling a model that has proved unreachable; see UpstreamCircuit.
        circuit = UpstreamCircuit()
        unavailable = UpstreamResponse(score=0.5, available=False)
        announced: set[str] = set()

        for row in rows:
            try:
                tx = TransactionData(**row.transaction)
            except Exception as e:
                summary.skipped += 1
                yield sse(
                    "progress",
                    {
                        "index": row.index,
                        "skipped": True,
                        "message": f"Row {row.index} rejected: {e.__class__.__name__}",
                    },
                )
                continue

            # Scores and fusion only — no narration per row.
            if circuit.skipped and len(circuit.skipped) == 3:
                # Every model is down; no point dialling at all.
                b = g = t = unavailable
            else:
                b, g, t = await _fetch_from_upstream_apis(tx, f"BATCH_{row.index}")

            for name, resp in (("behavioral", b), ("graph", g), ("temporal", t)):
                circuit.record(name, resp.available)

            # Tell the client the first time a model is written off, so a long
            # run does not look healthy when two thirds of it is imputed.
            for name in circuit.skipped:
                if name not in announced:
                    announced.add(name)
                    yield sse(
                        "upstream",
                        {
                            "modality": name,
                            "message": (
                                f"The {name} model API did not respond and has been "
                                f"skipped for the rest of this file. Its score is "
                                f"imputed and confidence penalised."
                            ),
                        },
                    )

            if circuit.is_open("behavioral"):
                b = unavailable
            if circuit.is_open("graph"):
                g = unavailable
            if circuit.is_open("temporal"):
                t = unavailable

            fusion = await asyncio.to_thread(
                meta_classifier.fuse,
                graph_score=g.score if g.available else None,
                behavioral_score=b.score if b.available else None,
                temporal_score=t.score if t.available else None,
            )

            classification = _classify(fusion.confidence_score)

            # With no model reachable every score is imputed to the same neutral
            # value, which fuses to a figure above any sane threshold — so the
            # system would alert on every row, including the legitimate ones.
            # An alert carrying no evidence is worse than no alert: it trains
            # the reviewer to ignore them. Rows scored with zero modalities are
            # reported as unscored and excluded from the metrics.
            scored_at_all = fusion.modalities_used > 0
            alerted = scored_at_all and fusion.confidence_score >= threshold

            if scored_at_all:
                update_summary(summary, classification, alerted, row.is_fraud_label)
            else:
                summary.analysed += 1
                summary.unscored += 1

            record = {
                "index": row.index,
                "nameOrig": tx.nameOrig,
                "nameDest": tx.nameDest,
                "type": tx.type,
                "amount": tx.amount,
                "score": fusion.confidence_score,
                "classification": classification,
                "alerted": alerted,
                "unscored": not scored_at_all,
                "label": row.is_fraud_label,
                "typology_label": row.typology_label,
                # Only report a detector's score when that detector answered.
                # Fusion imputes 0.5 for a missing modality — correct for the
                # arithmetic, since it is the neutral prior the uncertainty
                # penalty is then applied to — but emitting it here put a
                # number that no model produced in the same column as ones
                # that did. A detector that did not run reports null.
                "graph_score": fusion.graph_score if fusion.graph_available else None,
                "behavioral_score": (fusion.behavioral_score
                                     if fusion.behavioral_available else None),
                "temporal_score": (fusion.temporal_score
                                   if fusion.temporal_available else None),
                "modalities_used": fusion.modalities_used,
                # The per-detector terms, so a caller can show how each row was
                # decided rather than only what it was decided to be. Already
                # computed on the way to the score; carrying them costs nothing.
                "contributions": fusion.contributions,
                "driver": fusion.driver,
            }
            scored.append(record)
            yield sse("progress", record)

        elapsed_ms = int((time.perf_counter() - started) * 1000)

        # Narrate only the highest-risk rows.
        narratives = []
        top = sorted(scored, key=lambda r: r["score"], reverse=True)[: max(0, narrate_top)]
        for record in top:
            if not record["alerted"] or forensic_reporter is None:
                continue
            try:
                retrievals = await asyncio.to_thread(
                    retriever.retrieve,
                    graph_score=record["graph_score"],
                    behavioral_score=record["behavioral_score"],
                    temporal_score=record["temporal_score"],
                    confidence_score=record["score"],
                )
                if not retrievals:
                    continue
                from backend.rag.prompt_builder import (
                    UpstreamContext,
                    build_chain_of_evidence_prompt,
                )

                report = await asyncio.to_thread(
                    forensic_reporter.generate_report,
                    build_chain_of_evidence_prompt(
                        transaction_id=f"ROW_{record['index']}",
                        graph_score=record["graph_score"],
                        behavioral_score=record["behavioral_score"],
                        temporal_score=record["temporal_score"],
                        confidence_score=record["score"],
                        graph_available=record["graph_score"] is not None,
                        behavioral_available=record["behavioral_score"] is not None,
                        temporal_available=record["temporal_score"] is not None,
                        retrieval=retrievals[0],
                        upstream_context=UpstreamContext(),
                    ),
                )
                narratives.append(
                    {
                        "index": record["index"],
                        "score": record["score"],
                        "typology": retrievals[0].typology_name,
                        "report": report,
                    }
                )
                yield sse("narrative", narratives[-1])
            except Exception as e:
                logger.error(f"Batch narration failed for row {record['index']}: {e}")

        yield sse(
            "summary",
            {
                "total": summary.total,
                "analysed": summary.analysed,
                "skipped": summary.skipped,
                "unscored": summary.unscored,
                "alerts": summary.alerts,
                "by_classification": summary.by_classification,
                "has_labels": summary.has_labels,
                "metrics": summary.metrics(),
                "elapsed_ms": elapsed_ms,
                "narratives": len(narratives),
                "skipped_upstreams": circuit.skipped,
            },
        )

        from backend.auth import audit

        await audit(
            "analysis.batch",
            actor=user.username,
            target=file.filename,
            detail=f"rows={summary.analysed} alerts={summary.alerts}",
        )

    return StreamingResponse(
        emit(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
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
    """Render the alert email exactly as the monitor sends it.

    Uses monitor.alert_email — the template that actually ships. It previously
    rendered a second, older template from email_service, so the preview showed
    a design no recipient ever received. Inline images are swapped for data
    URIs, since a browser cannot resolve cid: references.
    """
    import base64

    from fastapi.responses import HTMLResponse

    from backend import thresholds
    from monitor import alert_email, assets

    sev = (classification or "HIGH").upper()
    alert, sg, scores = alert_email.sample(sev)
    html = alert_email.build(
        alert, sg, scores,
        bands=thresholds.current() or thresholds.DEFAULT_BANDS,
        has_image=False,
        case_ref="CASE-2026-0184",
        console_url=str(config.get("upstream", "console_url") or "").rstrip("/"),
        report_attached=True,
    )
    for cid, data in assets.inline_for(sev).items():
        mime = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
        html = html.replace(
            f"cid:{cid}", f"data:{mime};base64,{base64.b64encode(data).decode()}")
    return HTMLResponse(content=html)


@app.post("/email/send-test", tags=["email"])
async def send_test_email(
    req: RiskManagerRequest, user: User = Depends(require_manager)
):
    """Send a test alert to verify delivery. Admin or risk manager.

    Sends the same template the monitor sends, banners and all, so a
    successful test proves the thing that will actually arrive — not a
    different email that happens to share a subject line.
    """
    import asyncio

    from backend import thresholds
    from backend.email_service import SendOutcome, _send_rich
    from monitor import alert_email, assets

    sev = "HIGH"
    alert, sg, scores = alert_email.sample(sev)
    html = alert_email.build(
        alert, sg, scores,
        bands=thresholds.current() or thresholds.DEFAULT_BANDS,
        has_image=False, case_ref="CASE-2026-0184",
        console_url=str(config.get("upstream", "console_url") or "").rstrip("/"),
        report_attached=False,
    )
    text = alert_email.build_text(alert, sg, scores, report_attached=False)

    sent = await asyncio.to_thread(
        _send_rich, f"[TEST] [{sev}] DeepSentinel alert {alert['transaction_id']}",
        text, html, [req.email], assets.inline_for(sev) or None, None)

    if not sent:
        raise HTTPException(
            status_code=409,
            detail="Email is not configured, or SMTP rejected the message. "
                   "Check the SMTP settings under Settings.")

    return {
        "sent": True,
        "recipient": req.email,
        "template": "monitor alert (the one real alerts use)",
        "images": sorted(assets.inline_for(sev)),
    }


@app.get("/graph/neighbourhood", tags=["graph"])
async def graph_neighbourhood(
    account: str, hops: int = 1, max_edges: int = 150,
    scope: str = "component",
    user: User = Depends(require_any_user),
):
    """The payment graph immediately around one account.

    Proxied rather than called directly from the browser: the detector services
    are not exposed to the internet and carry no auth of their own, so the
    console reaches them through here and inherits the platform's session.

    Bounded at the far end — the served graph is 3.27M accounts and nothing is
    ever going to hand a browser all of it. The caller asks for one account,
    draws what comes back, and walks outward from there.
    """
    import httpx

    from backend import graph_explorer

    cfg = graph_explorer.current()
    if not cfg["enabled"]:
        raise HTTPException(
            403, "The graph explorer is switched off. An administrator can "
                 "turn it back on under System.")
    # Clamped here, not in the browser. A limit the client enforces is a
    # suggestion; the detector is what actually pays for a large request.
    hops = max(1, min(int(hops), cfg["max_hops"]))
    max_edges = max(10, min(int(max_edges), cfg["max_edges"]))

    base = str(config.get("upstream", "graph_api_base")).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{base}/api/graph/neighbourhood",
                params={"account": account, "hops": hops,
                        "max_edges": max_edges, "scope": scope},
            )
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(
            502, f"The network detector did not answer: {type(exc).__name__}"
        ) from exc

    if r.status_code == 404:
        raise HTTPException(404, f"No account {account!r} in the graph snapshot.")
    if r.status_code != 200:
        raise HTTPException(502, f"The network detector returned {r.status_code}.")
    return r.json()


@app.post("/graph/demo/score-account", tags=["graph"])
async def graph_demo_score_account(
    body: dict,
    user: User = Depends(require_any_user),
):
    """Score an account the relational model has never seen.

    Demo surface, and deliberately separate from /analyze. The platform's
    normal path answers about transactions between accounts the snapshot
    already contains; this one exists to show the thing that path cannot show —
    that an account which did not exist at training time still gets a real
    embedding, aggregated from whoever it is attached to.

    Only the relational detector runs. No fusion, no other modality, no
    alerting, nothing written to a case. What is on screen is attributable to
    one model.
    """
    import httpx

    base = str(config.get("upstream", "graph_api_base")).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{base}/api/graph/demo/score-account", json=body)
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(
            502, f"The network detector did not answer: {type(exc).__name__}"
        ) from exc
    if r.status_code >= 400:
        # Pass the detector's own explanation through. These are the messages
        # that say which counterparties it could not find, and replacing them
        # with a generic 502 is what makes a demo impossible to debug on stage.
        try:
            detail = r.json()
        except Exception:                               # noqa: BLE001
            detail = {"message": r.text[:200]}
        raise HTTPException(r.status_code if r.status_code < 500 else 502,
                            detail.get("message", "The detector refused."))
    return r.json()


@app.post("/graph/demo/score-csv", tags=["graph"])
async def graph_demo_score_csv(
    file: UploadFile = File(...),
    user: User = Depends(require_any_user),
):
    """Run a CSV through the relational model and nothing else."""
    import httpx

    base = str(config.get("upstream", "graph_api_base")).rstrip("/")
    raw = await file.read()
    if len(raw) > 4_000_000:
        raise HTTPException(413, "That file is larger than the 4 MB demo limit.")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            r = await client.post(
                f"{base}/api/graph/demo/score-csv",
                files={"file": (file.filename or "demo.csv", raw, "text/csv")},
            )
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(
            502, f"The network detector did not answer: {type(exc).__name__}"
        ) from exc
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:                               # noqa: BLE001
            detail = {"message": r.text[:200]}
        raise HTTPException(r.status_code if r.status_code < 500 else 502,
                            detail.get("message", "The detector refused."))
    return r.json()


@app.get("/graph/settings", tags=["graph"])
async def get_graph_settings(user: User = Depends(require_any_user)):
    """Whether the explorer is on, and how far it may reach. Readable by anyone
    signed in, so the page can explain itself rather than just failing."""
    from backend import graph_explorer

    return graph_explorer.current()


class GraphSettings(BaseModel):
    enabled: bool | None = None
    max_hops: int | None = None
    max_edges: int | None = None


@app.put("/graph/settings", tags=["graph"])
async def set_graph_settings(body: GraphSettings,
                             user: User = Depends(require_admin)):
    """Administrators only. This is a load control rather than a preference:
    it decides how hard everyone else can make the network detector work."""
    from backend import graph_explorer
    from backend.auth import audit

    cfg = graph_explorer.update(
        enabled=body.enabled, max_hops=body.max_hops,
        max_edges=body.max_edges, actor=user.username)
    await audit("graph.settings", actor=user.username, target="graph explorer",
                detail=str(cfg))
    return cfg


@app.get("/analyses/{analysis_id}/report.pdf", tags=["report"])
async def analysis_report_pdf(
    analysis_id: int,
    style: str | None = None,
    user: User = Depends(require_any_user),
):
    """The forensic narrative for one analysis, as a filed document.

    The console could already show this narrative as text and the monitor could
    already attach it to an alert email, but there was no way to get the
    document itself out of a screen you were looking at — which is the thing an
    investigator actually keeps. Same writer, same styles, same bytes the alert
    attaches, so what is downloaded is what gets filed.

    Built from the stored record rather than from whatever the browser happens
    to be holding: the PDF is evidence, and it should say what was persisted.
    """
    from fastapi.responses import Response

    from backend import packages, report_styles, sar
    from monitor.engine import _report_pdf

    # Same gate as the endpoint that returns this narrative as text. Without it
    # the licence check was bypassable by asking for the PDF instead of the
    # JSON — the paid feature handed over in a different content type.
    packages.require("forensic_report")

    if style is not None and style not in report_styles.STYLES:
        raise HTTPException(404, f"No report style named {style!r}.")

    record = await sar.get_analysis(analysis_id)
    if not record.forensic_report:
        raise HTTPException(
            409,
            "This analysis has no forensic narrative recorded, so there is "
            "nothing to render. Re-run it with report generation enabled.",
        )

    # _report_pdf speaks the monitor's alert shape; a stored analysis carries
    # the same facts under different names.
    scores = {
        "graph": record.graph_score,
        "behavioural": record.behavioral_score,
        "temporal": record.temporal_score,
    }
    answered = {k: v for k, v in scores.items() if v is not None}
    alert = {
        "transaction_id": record.transaction_id,
        "severity": record.classification,
        "fused_score": record.fraud_confidence_score,
        "graph_score": record.graph_score,
        "pattern": None,
        "sink_account": record.name_dest,
        "amount": record.amount,
        "from": record.name_orig,
        "to": record.name_dest,
        "modalities_used": record.modalities_used,
        "fusion_method": "meta_classifier",
        # The loudest detector that actually answered. Not a contribution —
        # those are not persisted on the record — so it is named for what it
        # is rather than dressed up as an attribution.
        "driver": max(answered, key=answered.get) if answered else None,
        "scores": scores,
        "at": 0,
    }

    pdf = _report_pdf(alert, record.forensic_report,
                      style=style or report_styles.selected())
    stamp = (record.transaction_id or str(analysis_id))[:18]
    return Response(
        content=pdf, media_type="application/pdf",
        headers={"Content-Disposition":
                 f'attachment; filename="deepsentinel-report-{stamp}.pdf"'})


@app.get("/packages/catalogue", tags=["packages"])
async def packages_catalogue():
    """The plans as a buyer sees them.

    Unauthenticated on purpose — it is the public pricing page's data, and a
    price list nobody can read before signing up is not a price list.

    Built from the same tables the gate enforces, so the website cannot end up
    advertising a feature the software does not actually unlock.
    """
    from backend import packages

    return packages.catalogue()


@app.post("/detectors/temporal/warm", tags=["detectors"])
async def warm_temporal_window(user: User = Depends(require_any_user)):
    """Fill the sequence detector's window so it can answer at all.

    The detector holds the last 32 transactions and refuses to score until it
    has them. That is correct for a live stream, which supplies 32 in a few
    seconds, and impossible on a page that sends one transaction per click —
    the window creeps up by one and never arrives.

    So the window is filled here, from genuine PaySim rows served by the graph
    service rather than invented ones. That matters: the detector reports which
    earlier transaction it attended to, and priming with fabricated rows would
    have it name a transaction that never happened.
    """
    import httpx

    graph = str(config.get("upstream", "graph_api_base")).rstrip("/")
    temporal = str(config.get("upstream", "temporal_api_base")).rstrip("/")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            before = (await client.get(f"{temporal}/api/v1/runtime")).json()
        except Exception as exc:                        # noqa: BLE001
            raise HTTPException(
                502, f"The sequence detector did not answer: {type(exc).__name__}"
            ) from exc

        need = max(0, int(before.get("window_size", 32))
                   - int(before.get("buffer_filled", 0)))
        if need == 0:
            return {"warmed": 0, "buffer_filled": before.get("buffer_filled"),
                    "warming_up": False,
                    "note": "The window was already full."}

        try:
            rows = (await client.get(
                f"{graph}/api/graph/sample-transactions",
                params={"n": need, "fraud_ratio": 0.1})).json()["transactions"]
        except Exception as exc:                        # noqa: BLE001
            raise HTTPException(
                502, "Could not fetch transactions to warm the window: "
                     f"{type(exc).__name__}") from exc

        sent = 0
        for row in rows:
            payload = {**row,
                       "composite_id": f"{row.get('nameOrig')}_{row.get('step')}"}
            try:
                # 503 is the expected answer while filling — the call still
                # advances the window, which is the whole point.
                await client.post(f"{temporal}/api/v1/classify", json=payload)
                sent += 1
            except Exception:                           # noqa: BLE001
                break

        after = (await client.get(f"{temporal}/api/v1/runtime")).json()

    return {
        "warmed": sent,
        "buffer_filled": after.get("buffer_filled"),
        "window_size": after.get("window_size"),
        "warming_up": after.get("warming_up"),
        "note": ("Filled with real transactions drawn from the served graph, "
                 "so the predecessor the detector names is a genuine one."),
    }


@app.get("/graph/model", tags=["graph"])
async def graph_model(user: User = Depends(require_any_user)):
    """What the network detector is serving, and on what.

    Straight from the detector rather than restated here: the size of the
    graph, the protocol it was evaluated under, the calibration, and the
    operating point. The protocol matters most — a score means nothing without
    knowing whether the window it was measured on was held out.
    """
    import httpx

    base = str(config.get("upstream", "graph_api_base")).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            health = (await client.get(f"{base}/health")).json()
            runtime = (await client.get(f"{base}/api/graph/runtime")).json()
            # Older builds of the detector do not serve this; the page copes.
            r = await client.get(f"{base}/api/graph/performance")
            performance = r.json() if r.status_code == 200 else None
    except Exception as exc:                            # noqa: BLE001
        raise HTTPException(
            502, f"The network detector did not answer: {type(exc).__name__}"
        ) from exc
    return {**health, "runtime": runtime, "performance": performance}


@app.get("/fusion/model", tags=["analysis"])
async def fusion_model(user: User = Depends(require_any_user)):
    """The meta-classifier's own shape, plus what it has decided so far.

    The weights are published deliberately. This component's claim is that the
    fusion is linear and its terms can be read off a verdict; a page that
    shows the weights is that claim kept rather than asserted.
    """
    from backend import thresholds
    from backend.settings import analysis_statistics

    described = (meta_classifier.describe()
                 if meta_classifier is not None else {"method": "unavailable"})

    counts = {}
    try:
        counts = await analysis_statistics()
    except Exception as exc:                            # noqa: BLE001
        logger.debug(f"No analysis statistics for the fusion page: {exc}")

    bands = thresholds.current() or thresholds.DEFAULT_BANDS
    from backend import fusion_eval

    return {
        **described,
        "bands": bands,
        "decided": counts,
        "performance": fusion_eval.evaluate(meta_classifier, bands),
    }


@app.get("/fusion/stages", tags=["analysis"])
async def fusion_stages(user: User = Depends(require_any_user)):
    """Each step between three scores and a filed document, checked.

    The fusion tab treats this as one component, and it is five: weigh the
    detectors, match a known typology, write the narrative, render the
    document, deliver it. Each can fail on its own and only the first is
    visible in a verdict — a report generator that has run out of quota and an
    SMTP password that expired both leave the numbers looking perfect.

    The PDF stage is genuinely exercised rather than inspected: the writer is
    asked for a document and the bytes are counted. A renderer that imports
    cleanly and produces nothing would pass any lighter check.
    """
    from backend import report_styles, thresholds
    from monitor.router import _delivery as delivery_status

    stages: list[dict] = []

    # 1 ── weigh the detectors
    described = (meta_classifier.describe()
                 if meta_classifier is not None else {"method": "unavailable"})
    stages.append({
        "key": "meta_classifier",
        "name": "Meta-classifier",
        "does": "Weighs the three detectors into one confidence",
        "ok": described.get("method") == "meta_classifier",
        "detail": ("linear model loaded" if described.get("method") == "meta_classifier"
                   else "falling back to the mean of available scores"),
        "figures": ([{"label": k, "value": f"{v:+.3f}"}
                     for k, v in (described.get("weights") or {}).items()]
                    or [{"label": "weights", "value": "—"}]),
    })

    # 2 ── match a known pattern
    try:
        n_typologies = len(knowledge_base.get_collection().get(include=[])["ids"])
    except Exception as exc:                            # noqa: BLE001
        logger.debug(f"Cannot count typologies: {exc}")
        n_typologies = 0
    stages.append({
        "key": "retrieval",
        "name": "FATF typology",
        "does": "Finds the closest known laundering pattern",
        "ok": n_typologies > 0,
        "detail": (f"{n_typologies} typologies indexed" if n_typologies
                   else "no knowledge base loaded"),
        "figures": [{"label": "indexed", "value": str(n_typologies)},
                    {"label": "source", "value": "FATF"}],
    })

    # 3 ── write it up
    stages.append({
        "key": "reporter",
        "name": "Report writer",
        "does": "Writes the narrative, citing only what was retrieved",
        "ok": forensic_reporter is not None,
        "detail": ("language model configured" if forensic_reporter is not None
                   else "not configured — verdicts still stand, narratives do not"),
        "figures": [{"label": "grounding", "value": "retrieval only"}],
    })

    # 4 ── render the document, for real
    pdf_ok, pdf_detail, pdf_bytes = False, "not attempted", 0
    try:
        from monitor.alert_email import sample
        from monitor.engine import _report_pdf

        alert, _sg, scores = sample("HIGH")
        alert["scores"] = scores
        blob = _report_pdf(alert, "SECTION 1 - EXECUTIVE SUMMARY Self-check.",
                           style=report_styles.selected())
        pdf_bytes = len(blob)
        pdf_ok = blob[:5] == b"%PDF-" and pdf_bytes > 800
        pdf_detail = (f"rendered {pdf_bytes:,} bytes" if pdf_ok
                      else "writer returned something that is not a PDF")
    except Exception as exc:                            # noqa: BLE001
        pdf_detail = f"{type(exc).__name__}: {exc}"[:120]
    stages.append({
        "key": "pdf",
        "name": "PDF writer",
        "does": "Renders the report as the document that gets filed",
        "ok": pdf_ok,
        "detail": pdf_detail,
        "figures": [{"label": "style", "value": report_styles.selected()},
                    {"label": "test render", "value": f"{pdf_bytes:,} B" if pdf_bytes else "—"}],
    })

    # 5 ── deliver it
    delivery = {}
    try:
        delivery = await delivery_status()
    except Exception as exc:                            # noqa: BLE001
        logger.debug(f"No delivery status for the fusion page: {exc}")
    raised = delivery.get("raised") or 0
    delivered = delivery.get("delivered") or 0
    stages.append({
        "key": "email",
        "name": "Email delivery",
        "does": "Sends the alert to the nominated risk managers",
        # Configured but never exercised is not a failure; configured and
        # dropping everything is. They are told apart here rather than both
        # being shown as a warning.
        "ok": bool(delivery.get("configured")) and (raised == 0 or delivered > 0),
        "detail": (f"{delivered} of {raised} alerts delivered" if raised
                   else "configured, nothing raised yet"),
        "figures": [
            {"label": "recipients", "value": str(delivery.get("recipients") or 0)},
            {"label": "sending as", "value": delivery.get("sending_as") or "—"},
        ],
    })

    return {"stages": stages, "bands": thresholds.current() or {}}


@app.get("/report-styles", tags=["report"])
async def list_report_styles(user: User = Depends(require_any_user)):
    """The available looks for the forensic report PDF, and which is in force.

    Readable by anyone signed in: an analyst who receives these should be able
    to see what the options are even if they are not the one who picks.
    """
    from backend import report_styles

    return {"styles": report_styles.listing(),
            "selected": report_styles.selected()}


@app.get("/report-styles/{name}/preview", tags=["report"])
async def preview_report_style(name: str, user: User = Depends(require_any_user)):
    """The report rendered in one style, as a real PDF.

    A picture of a layout is not the layout. This returns the same bytes the
    alert would attach, so what is previewed is what gets filed — the same
    reason the email preview renders the shipping template rather than a copy
    of it.
    """
    from fastapi.responses import Response

    from backend import report_styles
    from monitor.alert_email import sample
    from monitor.engine import _report_pdf

    if name not in report_styles.STYLES:
        raise HTTPException(404, f"No report style named {name!r}.")

    alert, sg, scores = sample("CRITICAL")
    alert["scores"] = scores
    narrative = (
        "SECTION 1 - EXECUTIVE SUMMARY The transaction moved "
        f"{alert['amount']:,.2f} from {alert['from']} to {alert['to']} and was "
        f"assigned a fused fraud confidence of {alert['fused_score']:.4f} across "
        "three detectors.\n\n"
        "SECTION 2 - MULTI-MODAL EVIDENCE ANALYSIS The behavioural model returned "
        f"{scores['behavioural']:.4f}, the largest single contribution. The network "
        f"model returned {scores['graph']:.4f} and identified a hub-and-spoke "
        f"structure converging on {alert['sink_account']}. The timing model "
        f"returned {scores['temporal']:.4f}.\n\n"
        "SECTION 3 - TYPOLOGY GROUNDING The retrieved typology is Mule Network - "
        "Hub and Spoke, matched against the indexed FATF descriptions.\n\n"
        "SECTION 4 - FORENSIC CONFIDENCE ASSESSMENT All three detectors answered, "
        "so no modality was imputed and no uncertainty shrink was applied.\n\n"
        "SECTION 5 - INVESTIGATIVE RECOMMENDATION Not available in the source record."
    )
    pdf = _report_pdf(alert, narrative, style=name)
    return Response(
        content=pdf, media_type="application/pdf",
        headers={"Content-Disposition":
                 f'inline; filename="report-preview-{name}.pdf"'})


class ReportStyleChoice(BaseModel):
    style: str


@app.put("/report-styles/selected", tags=["report"])
async def choose_report_style(body: ReportStyleChoice,
                              user: User = Depends(require_manager)):
    """Set the style every future report is rendered in.

    Administrators and risk managers. It is shared and cosmetic — it changes
    how the report looks, never what it says — so it does not need the same
    guard as the pipeline controls, but it is still everyone's document and the
    change is audited.
    """
    from backend import report_styles
    from backend.auth import audit

    try:
        out = report_styles.choose(body.style, actor=user.username)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    await audit("report.style", actor=user.username, target="forensic report",
                detail=f"style={body.style}")
    return out


@app.get("/email/status", tags=["email"])
async def email_status(user: User = Depends(require_manager)):
    """Report whether outgoing email is configured, and how."""
    from backend.email_service import _provider

    provider, settings = _provider()
    if provider == "smtp":
        return {
            "configured": True,
            "provider": "smtp",
            "sending_as": settings["username"],
            "host": f"{settings['host']}:{settings['port']}",
        }
    if provider == "sendgrid":
        return {
            "configured": True,
            "provider": "sendgrid",
            "sending_as": config.get("email", "sender_email"),
            "note": "The sender address must be verified in SendGrid or sends return 403.",
        }
    return {
        "configured": False,
        "provider": None,
        "detail": (
            "No provider configured — alerts will not be delivered. Set SMTP "
            "credentials or a SendGrid API key in config.ini."
        ),
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


@app.patch("/auth/me", tags=["auth"])
async def update_me(req: ProfileUpdate, user: User = Depends(get_current_user)):
    """Change your own display name or email. Audited."""
    from backend.auth import update_profile

    return UserOut.from_model(await update_profile(user.username, req))


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


# ── Analysis history ──────────────────────────────────────────────────────────
# Recovered from the fusion_engine branch, which also deleted the chatbot and
# assistant integration — so these were ported rather than the branch merged.


@app.get("/analyses", tags=["analysis"])
async def list_analyses(
    limit: int = 50,
    classification: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Recent analyses, newest first. Optionally filtered by classification."""
    from backend.db.models import as_utc
    from backend.settings import list_recent_analyses

    records = await list_recent_analyses(limit=limit, classification=classification)
    return [
        {
            # The row's own id. Without it the history list is a dead end:
            # every per-analysis route — /analyses/{id}/sar, /explain — is
            # keyed on it, so the UI could list a record and then had no way
            # to open anything about it.
            "id": r.id,
            "transaction_id": r.transaction_id,
            "created_at": as_utc(r.created_at),
            "fraud_confidence_score": r.fraud_confidence_score,
            "classification": r.classification,
            "modalities_used": r.modalities_used,
            "graph_score": r.graph_score,
            "behavioral_score": r.behavioral_score,
            "temporal_score": r.temporal_score,
            "typology_name": r.typology_name,
            "typology_id": r.typology_id,
            "similarity_score": r.similarity_score,
            "type": r.tx_type,
            "amount": r.amount,
            "nameOrig": r.name_orig,
            "nameDest": r.name_dest,
            "alert_sent": r.alert_sent,
            "mock_scenario": r.mock_scenario,
        }
        for r in records
    ]


@app.get("/analyses/statistics", tags=["analysis"])
async def get_analysis_statistics(user: User = Depends(get_current_user)):
    """Aggregate counts across everything analysed so far."""
    from backend.settings import analysis_statistics

    return await analysis_statistics()


# ── Packages ─────────────────────────────────────────────────────────────────
# Which commercial package this deployment is licensed for. Detection, fusion,
# alerting and monitoring are never gated — see backend/packages.py.


@app.get("/packages", tags=["packages"])
async def get_package(user: User = Depends(get_current_user)):
    """The licensed package and which features it unlocks."""
    from backend import packages

    return packages.status()


@app.put("/packages", tags=["packages"])
async def set_package_endpoint(
    body: dict, user: User = Depends(require_admin)
):
    """Change the licensed package. Admin only, and audited."""
    from backend import packages
    from backend.auth import audit

    name = body.get("package")
    if not name:
        raise HTTPException(422, "Body must contain a 'package' field.")

    previous = packages.current().value
    pkg = packages.set_package(str(name), actor=user.username)
    await audit(
        "package.change",
        actor=user.username,
        target=pkg.value,
        detail=f"{previous} -> {pkg.value}",
    )
    return packages.status()


# ── Plain-English explanation ────────────────────────────────────────────────


@app.post("/analyses/{analysis_id}/explain", tags=["analysis"])
async def explain_plainly(analysis_id: int, user: User = Depends(get_current_user)):
    """Restate this alert's forensic report for a non-specialist."""
    from backend import packages
    from backend.rag.prompt_builder import build_plain_english_prompt
    from backend.sar import get_analysis

    packages.require("forensic_report")
    record = await get_analysis(analysis_id)
    if not record.forensic_report:
        raise HTTPException(
            409, "This alert has no forensic report to restate."
        )
    if forensic_reporter is None:
        raise HTTPException(503, "No language model is configured.")

    package = build_plain_english_prompt(
        record.forensic_report, record.classification or "UNKNOWN"
    )
    try:
        text_out = forensic_reporter.generate_report(package)
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(502, f"Explanation failed: {type(exc).__name__}")

    return {
        "analysis_id": analysis_id,
        "classification": record.classification,
        "plain_english": (text_out or "").strip(),
        # Named so the UI can say this is a restatement, not a second opinion.
        "derived_from": "forensic_report",
    }


# ── Cases: review, timeline, sharing ─────────────────────────────────────────


# The fraud_cases table is defined by the Query Runner's schema, and the
# platform reads it over SQL rather than redeclaring an ORM model here. Two
# declarations of one table drift, and a silently-renamed column is exactly the
# failure that made every fused score read as zero.

CASE_COLUMNS = (
    "case_ref, transaction_id, detected_at, classification, fused_score, "
    "graph_score, behavioral_score, temporal_score, "
    "graph_available, behavioral_available, temporal_available, "
    "modalities_used, uncertainty_penalty_applied, "
    "typology_name, typology_id, graph_pattern, sink_account, "
    "graph_evidence, forensic_report, screening_ms, total_ms, "
    "alert_sent, alerted_at, recipients, "
    "review_status, reviewed_by, reviewed_at, review_note, label_is_fraud"
)
_CASE_FIELDS = [c.strip() for c in CASE_COLUMNS.split(",")]

# The queue lists up to 200 cases at a time, so the two remaining evidence
# blobs are read only when a single case is opened. Both are written at
# detection time by monitor/cases.py; without them the case desk can say what
# the behavioural and temporal detectors scored but not what they saw, which
# is the question a reviewer actually has.
CASE_DETAIL_COLUMNS = CASE_COLUMNS + ", behavioral_evidence, temporal_evidence"
_CASE_DETAIL_FIELDS = [c.strip() for c in CASE_DETAIL_COLUMNS.split(",")]


def _case_row(row, fields: Optional[list] = None) -> dict:
    """One case row as the UI consumes it.

    `fields` names the projection the row was selected with, since the detail
    view reads two columns the queue does not.
    """
    import json as _json

    def load(v):
        if isinstance(v, (dict, list)) or v is None:
            return v
        try:
            return _json.loads(v)
        except (ValueError, TypeError):
            return None

    d = dict(zip(fields or _CASE_FIELDS, row))
    for k in ("detected_at", "alerted_at", "reviewed_at"):
        d[k] = str(d[k]) if d[k] else None
    for k in ("graph_evidence", "behavioral_evidence", "temporal_evidence",
              "recipients"):
        if k in d:
            d[k] = load(d[k])
    for k in ("graph_available", "behavioral_available", "temporal_available",
              "uncertainty_penalty_applied", "alert_sent"):
        d[k] = None if d[k] is None else bool(d[k])
    d["label_is_fraud"] = None if d["label_is_fraud"] is None else bool(d["label_is_fraud"])
    return d


@app.get("/cases", tags=["cases"])
async def list_cases(
    limit: int = 50,
    review_status: Optional[str] = None,
    classification: Optional[str] = None,
    user: User = Depends(get_current_user),
):
    """Recorded cases, newest first."""
    from sqlalchemy import text as sql

    from backend.db.session import get_session

    where, params = ["1=1"], {"lim": min(limit, 200)}
    if review_status:
        where.append("review_status = :rs")
        params["rs"] = review_status
    if classification:
        where.append("classification = :cl")
        params["cl"] = classification.upper()

    try:
        async with get_session() as db:
            rows = (await db.execute(
                sql(f"SELECT {CASE_COLUMNS} FROM fraud_cases "
                    f"WHERE {' AND '.join(where)} "
                    f"ORDER BY detected_at DESC LIMIT :lim"),
                params,
            )).all()
        return {"cases": [_case_row(r) for r in rows]}
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(
            503,
            f"The fraud_cases table is not available in this database "
            f"({type(exc).__name__}). Create it with the Query Runner, pointed "
            f"at the same database as this service.",
        )


@app.get("/cases/{case_ref}", tags=["cases"])
async def get_case(case_ref: str, user: User = Depends(get_current_user)):
    """One case in full, with a timeline derived from its own timestamps."""
    from sqlalchemy import text as sql

    from backend.db.session import get_session

    async with get_session() as db:
        row = (await db.execute(
            sql(f"SELECT {CASE_DETAIL_COLUMNS} FROM fraud_cases WHERE case_ref = :r"),
            {"r": case_ref},
        )).first()
    if row is None:
        raise HTTPException(404, f"No case {case_ref}")

    case = _case_row(row, _CASE_DETAIL_FIELDS)

    # Derived rather than stored: two sources for one chronology eventually
    # disagree, and then neither can be trusted.
    timeline = [{"stage": "Detected", "at": case["detected_at"],
                 "detail": f"Screened and classified {case['classification']}"}]
    if case["screening_ms"]:
        timeline.append({"stage": "Graph screening", "at": case["detected_at"],
                         "detail": f"{case['screening_ms']} ms"})
    used = case["modalities_used"] or 0
    timeline.append({
        "stage": "Fusion", "at": case["detected_at"],
        "detail": (f"{used} of 3 detectors contributed"
                   + (" — uncertainty penalty applied"
                      if case["uncertainty_penalty_applied"] else "")),
    })
    if case["typology_name"]:
        timeline.append({"stage": "Typology matched", "at": case["detected_at"],
                         "detail": case["typology_name"]})
    if case["alert_sent"]:
        n = len(case["recipients"] or [])
        timeline.append({"stage": "Alert sent", "at": case["alerted_at"],
                         "detail": f"{n} recipient(s)" if n else "Notified"})
    if case["reviewed_at"]:
        timeline.append({
            "stage": "Reviewed", "at": case["reviewed_at"],
            "detail": f"{case['review_status']} by {case['reviewed_by'] or 'unknown'}",
        })

    return {**case, "timeline": timeline}


@app.patch("/cases/{case_ref}/review", tags=["cases"])
async def review_case(
    case_ref: str, body: dict, user: User = Depends(get_current_user)
):
    """Record an analyst's verdict.

    `confirmed_fraud` and `false_positive` are what a retraining set is built
    from, so the decision is attributed and audited.
    """
    from datetime import datetime, timezone

    from sqlalchemy import text as sql

    from backend.auth import audit
    from backend.db.session import get_session

    status = str(body.get("review_status") or "").lower()
    valid = {"open", "investigating", "confirmed_fraud", "false_positive", "closed"}
    if status not in valid:
        raise HTTPException(422, f"review_status must be one of: {', '.join(sorted(valid))}")

    async with get_session() as db:
        result = await db.execute(
            sql("UPDATE fraud_cases SET review_status = :s, reviewed_by = :by, "
                "reviewed_at = :at, review_note = COALESCE(:note, review_note) "
                "WHERE case_ref = :r"),
            {"s": status, "by": user.username, "at": datetime.now(timezone.utc),
             "note": (str(body["note"])[:2000] if body.get("note") else None),
             "r": case_ref},
        )
        if (result.rowcount or 0) == 0:
            raise HTTPException(404, f"No case {case_ref}")
        row = (await db.execute(
            sql(f"SELECT {CASE_COLUMNS} FROM fraud_cases WHERE case_ref = :r"),
            {"r": case_ref},
        )).first()

    await audit(f"case.{status}", actor=user.username, target=case_ref,
                detail=body.get("note"))
    return _case_row(row)


# ── The operating point ──────────────────────────────────────────────────────


class NetworkCapability(BaseModel):
    accounts: Optional[int] = Field(None, description="Accounts in the payment graph")
    transfers: Optional[int] = Field(None, description="Directed transfers between them")
    hops: Optional[int] = Field(None, description="Neighbourhood depth read per transaction")
    live: bool = Field(..., description="Whether the model is loaded and can score")


class BehaviouralCapability(BaseModel):
    strata: Optional[int] = Field(None, description="Models held, one per transaction type")
    latency_ms: Optional[float] = Field(None, description="Mean scoring time in milliseconds")
    live: bool = Field(..., description="Whether the service is answering")


class TemporalCapability(BaseModel):
    window: Optional[int] = Field(None, description="Preceding transactions read with each one")
    live: bool = Field(..., description="Whether the model is loaded and can score")


class FusionCapability(BaseModel):
    signals: int = Field(..., description="Detector scores combined into the verdict")
    typologies: Optional[int] = Field(None, description="Laundering methods indexed for retrieval")
    live: bool = Field(..., description="Whether fusion and retrieval are both available")


class Capabilities(BaseModel):
    """What each detector is and whether it is currently answering."""

    network: NetworkCapability
    behavioural: BehaviouralCapability
    temporal: TemporalCapability
    fusion: FusionCapability

    model_config = {
        "json_schema_extra": {
            "example": {
                "network": {"accounts": 3277509, "transfers": 2770409, "hops": 2, "live": True},
                "behavioural": {"strata": 4, "latency_ms": 3.41, "live": True},
                "temporal": {"window": 32, "live": False},
                "fusion": {"signals": 3, "typologies": 10, "live": True},
            },
        },
    }


def _typology_count() -> int | None:
    """How many laundering methods are indexed, straight from the store."""
    try:
        return knowledge_base.get_collection().count()
    except Exception:                                   # noqa: BLE001
        return None


_CAPS_CACHE: dict = {"at": 0.0, "body": None}
_CAPS_TTL = 60.0


async def detector_own_thresholds() -> dict:
    """Each detector's own decision threshold, asked of the detector.

    A per-model row measured at a number typed into the console is a number
    about the console, not about the model. These are read from the services
    themselves so the table cannot drift when someone retunes upstream.

    The behavioural service publishes none, so 0.5 is used and reported as a
    midpoint rather than a tuned value — the difference matters when reading
    that row.
    """
    async def probe(key: str) -> dict:
        base = str(config.get("upstream", key)).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{base}/health")
            return r.json() if r.status_code == 200 else {}
        except Exception:                               # noqa: BLE001
            return {}

    graph, behav, temp = await asyncio.gather(
        probe("graph_api_base"),
        probe("behavioral_api_base"),
        probe("temporal_api_base"),
    )

    def num(*candidates):
        for c in candidates:
            if isinstance(c, (int, float)):
                return float(c)
        return None

    return {
        "graph": {
            "value": num(graph.get("tuned_threshold"),
                         (graph.get("risk_bands") or {}).get("high"), 0.1830),
            "source": "tuned on the validation window"
                      if graph.get("tuned_threshold") is not None
                      else "last known value — the detector did not answer",
        },
        "behavioural": {
            "value": num(behav.get("threshold"), 0.5),
            "source": "tuned" if behav.get("threshold") is not None
                      else "midpoint — this model publishes no threshold",
        },
        "temporal": {
            "value": num(temp.get("threshold"), 0.4545),
            "source": "tuned" if temp.get("threshold") is not None
                      else "last known value — the detector did not answer",
        },
    }


class SimulationReset(BaseModel):
    """Confirmation for clearing simulation data."""

    confirm: str = Field(
        ...,
        description="Must be exactly 'reset simulation'. A destructive call "
                    "should not be one stray click or a bare curl away.",
    )
    dry_run: bool = Field(
        True,
        description="Report what would be removed without removing it. Defaults "
                    "to true so the harmless call is the easy one.",
    )


# Everything a simulation produces, and nothing else. The exclusions matter
# more than the list: users, risk-manager recipients, alerting settings and the
# audit log are configuration and history, not test output. Wiping those would
# lock the team out of a shared database and erase the record of who did it.
SIMULATION_TABLES = (
    "transactions_live",      # the ingestion queue the Query Runner writes
    "transactions_archive",   # payloads kept for case reconstruction
    "fraud_cases",            # what the monitor raised
    "analysis_records",       # what the analyzer scored
    "sar_drafts",             # filings drafted from those analyses
)


@app.post(
    "/simulation/reset",
    tags=["simulation"],
    summary="Clear simulation data from the shared database",
)
async def reset_simulation(body: SimulationReset, user: User = Depends(require_admin)):
    """Remove the transactions and cases a test run produced.

    For testing only. Several people share one database, so a run leaves
    traffic and cases behind that the next person then has to read around —
    this is how you hand the database back in the state you found it.

    Deliberately narrow. It empties the five tables a simulation writes and
    touches nothing else: accounts, alert recipients, thresholds and the audit
    trail all survive, because losing those costs the team far more than a
    dirty queue does.

    Defaults to a dry run. Pass `dry_run: false` to actually delete, and the
    deletion is itself written to the audit log — a reset that leaves no trace
    of who reset it is how a shared environment becomes unaccountable.
    """
    from sqlalchemy import text as _text

    from backend.auth import audit
    from backend.db.session import get_session

    if body.confirm != "reset simulation":
        raise HTTPException(
            422,
            "Set confirm to exactly 'reset simulation' to proceed. "
            "This clears shared test data for everyone.",
        )

    counts: dict[str, int] = {}
    async with get_session() as db:
        for table in SIMULATION_TABLES:
            try:
                counts[table] = int(
                    (await db.execute(_text(f'SELECT COUNT(*) FROM "{table}"'))).scalar() or 0)
            except Exception:                           # noqa: BLE001
                counts[table] = -1                      # table absent on this database

    if body.dry_run:
        return {
            "dry_run": True,
            "would_remove": counts,
            "total": sum(v for v in counts.values() if v > 0),
            "preserved": ["users", "risk_managers", "alert_settings", "audit_log"],
            "note": "Nothing was deleted. Send dry_run false to proceed.",
        }

    removed: dict[str, int] = {}
    async with get_session() as db:
        for table in SIMULATION_TABLES:
            if counts.get(table, -1) < 0:
                continue
            try:
                await db.execute(_text(f'DELETE FROM "{table}"'))
                removed[table] = counts[table]
            except Exception as exc:                    # noqa: BLE001
                logger.warning(f"Could not clear {table}: {exc}")
                removed[table] = -1

    # The monitor's alerts, activity feed and counters live in this process,
    # not in any of those tables. Clearing the rows without clearing these left
    # the dashboard listing alerts whose cases no longer existed — and, because
    # the same file had been replayed a few times, listing them repeatedly. A
    # reset that leaves the screen showing the old run has not reset anything
    # the user can actually see.
    live: dict[str, int] = {}
    try:
        from monitor.state import STATE

        live = STATE.clear_live(actor=user.username)
    except Exception as exc:                            # noqa: BLE001
        logger.warning(f"Could not clear the monitor's live state: {exc}")

    total = sum(v for v in removed.values() if v > 0)
    await audit("simulation.reset", actor=user.username, target="shared database",
                detail=f"cleared {total} row(s): "
                       + ", ".join(f"{k}={v}" for k, v in removed.items())
                       + f"; live alerts={live.get('alerts', 0)}")
    logger.info(f"Simulation data cleared by {user.username}: {removed}, live={live}")

    return {
        "dry_run": False,
        "removed": removed,
        "cleared_live": live,
        "total": total,
        "preserved": ["users", "risk_managers", "alert_settings", "audit_log"],
    }


@app.get(
    "/public/capabilities",
    tags=["public"],
    response_model=Capabilities,
    summary="Detector status — which models are live",
    responses={200: {"description": "Live status and capability of each detector."}},
)
async def public_capabilities():
    """Which detectors are answering, and what each one is.

    No credentials required — this is the endpoint to open in Swagger or
    Postman to show the state of the models.

    Deliberately not /api/monitor/runtime with the auth taken off. That
    endpoint answers "is the system healthy" and carries things an anonymous
    caller has no business with — the alert sending address, how many alerts
    went undelivered, queue depth, and upstream stack traces. This is a
    curated subset: sizes, capabilities and whether each detector is
    answering. Nothing here reveals internal addresses, failures or volumes.

    Cached for a minute, because it is reachable without a session and each
    miss probes three internal services.
    """
    import time as _t

    import httpx

    from backend import config

    now = _t.monotonic()
    if _CAPS_CACHE["body"] is not None and now - _CAPS_CACHE["at"] < _CAPS_TTL:
        return _CAPS_CACHE["body"]

    async def probe(key: str, path: str = "/health") -> dict:
        base = str(config.get("upstream", key)).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{base}{path}")
            return r.json() if r.status_code == 200 else {}
        except Exception:                               # noqa: BLE001
            return {}

    graph = await probe("graph_api_base", "/api/graph/runtime")
    behav = await probe("behavioral_api_base")
    temp = await probe("temporal_api_base")

    body = {
        "network": {
            "accounts": (graph.get("precomputed") or {}).get("accounts"),
            "transfers": (graph.get("precomputed") or {}).get("transactions"),
            "hops": (graph.get("model") or {}).get("k_hop"),
            "live": bool((graph.get("model") or {}).get("loaded")),
        },
        "behavioural": {
            "strata": len(behav.get("strata_loaded") or []) or None,
            "latency_ms": behav.get("mean_latency_ms"),
            "live": behav.get("status") == "ok",
        },
        "temporal": {
            "window": temp.get("window_size"),
            "live": bool(temp.get("status") == "ok" and not temp.get("load_error")),
        },
        "fusion": {
            "signals": 3,
            "typologies": _typology_count(),
            "live": meta_classifier is not None and retriever is not None,
        },
    }
    _CAPS_CACHE.update(at=now, body=body)
    return body


@app.get("/settings/thresholds", tags=["settings"])
async def get_thresholds(user: User = Depends(get_current_user)):
    """The fused operating point, and where it came from."""
    from backend import thresholds

    chosen = thresholds.current()
    return {
        "bands": chosen or thresholds.DEFAULT_BANDS,
        "source": "operator" if chosen else "model",
        "editable": ["critical", "high", "medium"],
        "note": ("Set here, the monitor alerts on this line. Cleared, it uses the "
                 "relational model's own calibrated bands."),
    }


@app.put("/settings/thresholds", tags=["settings"])
async def set_thresholds(body: dict, user: User = Depends(require_admin)):
    """Move the line the monitor actually alerts on. Admin only, and audited."""
    from backend import thresholds
    from backend.auth import audit

    previous = thresholds.current() or thresholds.DEFAULT_BANDS
    bands = thresholds.set_bands(body.get("bands") or body, actor=user.username)
    await audit("thresholds.set", actor=user.username, target="fused",
                detail=f"{previous} -> {bands}")
    return {"bands": bands, "source": "operator"}


@app.delete("/settings/thresholds", tags=["settings"])
async def clear_thresholds(user: User = Depends(require_admin)):
    """Hand the operating point back to the model's calibration."""
    from backend import thresholds
    from backend.auth import audit

    thresholds.clear(actor=user.username)
    await audit("thresholds.cleared", actor=user.username, target="fused")
    return {"bands": thresholds.DEFAULT_BANDS, "source": "model"}


# ── One detector, on its own ─────────────────────────────────────────────────
# The platform's whole argument is that three models see different things and
# fusion reconciles them — which means the fused number is the only thing most
# of the interface shows. That is right for an operator and wrong for anyone who
# has to defend a single component: there was no way to run one detector alone
# and look at what it, specifically, produced.


DETECTORS = {
    "graph": {
        "label": "Edge-Enhanced GraphSAGE",
        "owner": "relational",
        "base": "graph_api_base",
        "reads": "The payment network around this transaction.",
    },
    "behavioural": {
        "label": "Stratified VAE with Dual-Signal Anomaly Attribution",
        "owner": "behavioural",
        "base": "behavioral_api_base",
        "reads": "Whether this fits normal behaviour for its transaction type.",
    },
    "temporal": {
        "label": "Transaction-Sequence TCN with fraud_attention",
        "owner": "temporal",
        "base": "temporal_api_base",
        "reads": "The transactions immediately preceding this one.",
    },
}


class DetectorInfo(BaseModel):
    """One detector: what it is, where it lives, and whether it can score."""

    name: str = Field(..., description="Identifier to pass to POST /detectors/{name}")
    label: str = Field(..., description="Model name")
    reads: str = Field(..., description="What this detector looks at")
    live: bool = Field(..., description="Reachable and able to score right now")
    status: str = Field(..., description="serving | warming_up | unreachable | error")
    detail: Optional[str] = Field(None, description="Why it cannot score, when it cannot")
    model_version: Optional[str] = None
    docs_url: Optional[str] = Field(
        None, description="This detector's own OpenAPI docs, served by its own process")

    model_config = {"protected_namespaces": ()}


@app.get(
    "/detectors",
    tags=["detectors"],
    response_model=list[DetectorInfo],
    summary="List the detectors and whether each one can score",
)
async def list_detectors(user: User = Depends(get_current_user)):
    """Every detector, its status, and a link to its own API docs.

    Each model runs as its own service with its own OpenAPI page; this is the
    index across all three, so one request answers "what is deployed and what
    is answering" without visiting three ports.
    """
    import httpx

    out: list[dict] = []
    for name, meta in DETECTORS.items():
        base = str(config.get("upstream", meta["base"])).rstrip("/")
        probe = "/api/graph/runtime" if name == "graph" else "/health"
        info = {
            "name": name,
            "label": meta["label"],
            "reads": meta["reads"],
            "docs_url": f"{base}/docs",
            "live": False,
            "status": "unreachable",
            "detail": "The service did not respond.",
            "model_version": None,
        }
        try:
            async with httpx.AsyncClient(timeout=4.0) as c:
                r = await c.get(f"{base}{probe}")
            body = r.json() if r.status_code == 200 else {}
            info["model_version"] = body.get("model_version") or (
                body.get("model") or {}).get("stage")

            if body.get("load_error"):
                info.update(status="error", detail=body["load_error"])
            elif body.get("warming_up"):
                filled, size = body.get("buffer_filled", 0), body.get("window_size", 0)
                info.update(
                    status="warming_up",
                    detail=f"Needs a full window before it can score — {filled} of {size}.")
            elif name == "graph" and not (body.get("model") or {}).get("loaded"):
                info.update(status="error", detail="Reachable, but no model is loaded.")
            elif r.status_code == 200:
                info.update(live=True, status="serving", detail=None)
            else:
                info.update(status="error", detail=f"Health probe returned {r.status_code}.")
        except Exception as exc:                        # noqa: BLE001
            info["detail"] = f"{type(exc).__name__} contacting the service."
        out.append(info)
    return out


@app.post("/detectors/{name}", tags=["detectors"])
async def score_one_detector(
    name: str, body: dict, user: User = Depends(get_current_user)
):
    """Run a single detector and return exactly what it said.

    No fusion, no retrieval, no report — one model, its raw response, and how
    long it took. Nothing is imputed: a detector that does not answer is
    reported as unavailable rather than given a neutral score.
    """
    import time as _time

    from backend.adapters.upstream import (
        behavioural_evidence, call_behavioral_api, call_graph_api, call_temporal_api,
    )

    if name not in DETECTORS:
        raise HTTPException(
            404, f"Unknown detector '{name}'. One of: {', '.join(DETECTORS)}.")

    txn = body.get("transaction") or body
    if not txn.get("amount"):
        raise HTTPException(422, "Body must contain a transaction with an amount.")

    meta = DETECTORS[name]
    base = str(config.get("upstream", meta["base"])).rstrip("/")
    caller = {"graph": call_graph_api, "behavioural": call_behavioral_api,
              "temporal": call_temporal_api}[name]

    import httpx

    t0 = _time.perf_counter()
    try:
        # The adapters take the shared client and a timeout — this endpoint is
        # a one-shot call, so it opens its own.
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await caller(client, base, txn, 30.0)
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(
            503, f"{meta['label']} could not be reached at {base} "
                 f"({type(exc).__name__}).")
    elapsed = int((_time.perf_counter() - t0) * 1000)

    evidence = None
    if name == "graph":
        evidence = (res.extra or {}).get("suspicious_subgraph")
    elif name == "behavioural":
        # The adapter has already built the full decomposition and parked it
        # under `extra["evidence"]`. Running the flattener over `extra` instead
        # re-reads `extra["anomaly_fingerprint"]`, which is the two-string
        # headline summary the adapter keeps for backward compatibility, not
        # the fingerprint — so the per-feature and per-dimension shares, the
        # typology and the engineered feature values were all being dropped on
        # the way to the caller. The fall-back keeps the old behaviour for any
        # response that predates that key.
        evidence = ((res.extra or {}).get("evidence")
                    or behavioural_evidence(res.extra or {}))
    elif name == "temporal":
        evidence = (res.extra or {}).get("temporal_evidence")

    return {
        "detector": name,
        "label": meta["label"],
        "reads": meta["reads"],
        "endpoint": base,
        "available": res.available,
        # The score only when the detector actually answered. Reporting a
        # number for a model that did not run is the bug this whole endpoint
        # exists to make impossible to hide.
        "score": res.score if res.available else None,
        "summary": res.fraud_signal_summary,
        "typology_hint": res.typology_hint,
        "evidence": evidence,
        "raw": res.extra or {},
        "latency_ms": elapsed,
    }


# ── Picking a specific transaction to analyse ────────────────────────────────
# Analysing a randomly pulled transaction proves the pipeline runs. Analysing
# the transaction behind a case you are looking at proves something an
# investigator cares about, and is reproducible — the same input gives the same
# run, which a random pull never does.


def _payload_from_row(raw, cols: dict) -> dict:
    """The analyzable transaction, preferring the stored raw record.

    `raw` is exactly what was ingested; the typed columns are our parse of it.
    Prefer the former so the model sees what arrived, and fall back to
    rebuilding from the columns when a row predates raw capture.
    """
    import json as _json

    if raw:
        d = raw if isinstance(raw, dict) else _json.loads(raw)
        if d:
            # Cast the numerics: the archive stores the CSV's strings verbatim,
            # and the model APIs expect numbers.
            for k in ("amount", "oldbalanceOrg", "newbalanceOrig",
                      "oldbalanceDest", "newbalanceDest"):
                if k in d and d[k] is not None:
                    try:
                        d[k] = float(d[k])
                    except (TypeError, ValueError):
                        pass
            if d.get("step") is not None:
                try:
                    d["step"] = int(float(d["step"]))
                except (TypeError, ValueError):
                    pass
            # Ground truth must never reach a model.
            d.pop("isFraud", None)
            d.pop("is_fraud", None)
            return d

    return {
        "transaction_id": cols.get("transaction_id"),
        "step": cols.get("step"),
        "type": cols.get("tx_type"),
        "amount": cols.get("amount"),
        "nameOrig": cols.get("name_orig"),
        "nameDest": cols.get("name_dest"),
        "oldbalanceOrg": cols.get("old_balance_orig"),
        "newbalanceOrig": cols.get("new_balance_orig"),
        "oldbalanceDest": cols.get("old_balance_dest"),
        "newbalanceDest": cols.get("new_balance_dest"),
    }


@app.get("/transactions", tags=["analysis"])
async def search_transactions(
    q: Optional[str] = None,
    limit: int = 40,
    user: User = Depends(get_current_user),
):
    """Ingested transactions, searchable by id or either account.

    Joined to `fraud_cases` so a row says whether the platform has already
    judged it — picking one then means "re-run the case in front of me", which
    is the reason to choose a transaction rather than accept a random one.
    """
    from sqlalchemy import text as sql

    from backend.db.session import get_session

    where, params = ["1=1"], {"lim": max(1, min(limit, 200))}
    if q:
        where.append("(a.transaction_id LIKE :q OR a.name_orig LIKE :q "
                     "OR a.name_dest LIKE :q)")
        params["q"] = f"%{q.strip()}%"

    try:
        async with get_session() as db:
            rows = (await db.execute(
                sql(f"""
                    SELECT a.transaction_id, a.step, a.tx_type, a.amount,
                           a.name_orig, a.name_dest, a.is_fraud,
                           c.case_ref, c.classification, c.fused_score
                      FROM transactions_archive a
                      LEFT JOIN fraud_cases c
                        ON c.transaction_id = a.transaction_id
                     WHERE {' AND '.join(where)}
                     ORDER BY a.uploaded_at DESC, a.id DESC
                     LIMIT :lim
                """),
                params,
            )).all()
    except Exception as exc:                                  # noqa: BLE001
        raise HTTPException(
            503,
            f"No ingested transactions are available in this database "
            f"({type(exc).__name__}). Upload a file with the Query Runner first.",
        )

    return {
        "transactions": [
            {
                "transaction_id": r[0], "step": r[1], "type": r[2], "amount": r[3],
                "nameOrig": r[4], "nameDest": r[5],
                "label_is_fraud": None if r[6] is None else bool(r[6]),
                "case_ref": r[7], "classification": r[8], "fused_score": r[9],
            }
            for r in rows
        ]
    }


@app.get("/transactions/{transaction_id}", tags=["analysis"])
async def get_transaction(
    transaction_id: str, user: User = Depends(get_current_user)
):
    """One transaction, in the shape the analyzer sends to the models.

    Looks in the archive first — that is the record of what was ingested — then
    the live queue, so a transaction still awaiting screening can be analysed
    too.
    """
    from sqlalchemy import text as sql

    from backend.db.session import get_session

    async with get_session() as db:
        row = None
        try:
            row = (await db.execute(
                sql("SELECT raw, transaction_id, step, tx_type, amount, name_orig, "
                    "name_dest, old_balance_orig, new_balance_orig, "
                    "old_balance_dest, new_balance_dest "
                    "FROM transactions_archive WHERE transaction_id = :t LIMIT 1"),
                {"t": transaction_id},
            )).first()
        except Exception:                                     # noqa: BLE001
            pass

        if row is None:
            try:
                row = (await db.execute(
                    sql("SELECT payload, transaction_id, step, tx_type, amount, "
                        "name_orig, name_dest, old_balance_orig, new_balance_orig, "
                        "old_balance_dest, new_balance_dest "
                        "FROM transactions_live WHERE transaction_id = :t LIMIT 1"),
                    {"t": transaction_id},
                )).first()
            except Exception:                                 # noqa: BLE001
                pass

    if row is None:
        raise HTTPException(
            404,
            f"No stored transaction {transaction_id}. Cases replayed from the "
            f"graph service's sample feed are not persisted, so their original "
            f"payload cannot be recovered — ingest through the Query Runner to "
            f"be able to re-analyse.",
        )

    keys = ("transaction_id", "step", "tx_type", "amount", "name_orig",
            "name_dest", "old_balance_orig", "new_balance_orig",
            "old_balance_dest", "new_balance_dest")
    return {"transaction": _payload_from_row(row[0], dict(zip(keys, row[1:])))}


# ── Threshold simulation ─────────────────────────────────────────────────────
# Replays decisions already made at a different threshold. Historical, not
# predictive — see backend/simulation.py.


@app.get("/analyses/simulate", tags=["analysis"])
async def simulate_threshold(
    threshold: float | None = None,
    days: int | None = None,
    user: User = Depends(get_current_user),
):
    """Alert volume and, where labels exist, accuracy at a given threshold.

    Without `threshold`, returns the full curve so a slider can move without a
    round trip per pixel.
    """
    from backend import packages, simulation

    packages.require("threshold_sim")
    if threshold is None:
        return await simulation.sweep(days=days)
    return await simulation.at(threshold, days=days)


# ── Suspicious Activity Report drafting ──────────────────────────────────────
# The system drafts; a named officer reviews, edits and decides. Nothing here
# files anything with any authority.


@app.get("/analyses/{analysis_id}/sar", tags=["sar"])
async def get_sar_draft(analysis_id: int, user: User = Depends(get_current_user)):
    """The latest draft for this alert, or 404 if none has been generated."""
    from backend import packages, sar

    packages.require("sar_draft")
    draft = await sar.latest_draft(analysis_id)
    if draft is None:
        raise HTTPException(404, "No draft has been generated for this alert yet.")
    return draft


@app.post("/analyses/{analysis_id}/sar", tags=["sar"])
async def create_sar_draft(analysis_id: int, user: User = Depends(get_current_user)):
    """Draft a SAR from a stored alert. Audited, because it is a compliance artefact."""
    from backend import packages, sar
    from backend.auth import audit

    packages.require("sar_draft")
    draft = await sar.generate(analysis_id, forensic_reporter, actor=user.username)
    await audit(
        "sar.generate",
        actor=user.username,
        target=f"analysis:{analysis_id}",
        detail=f"draft {draft['id']} generated",
    )
    return draft


@app.patch("/analyses/sar/{draft_id}", tags=["sar"])
async def revise_sar_draft(
    draft_id: int, body: dict, user: User = Depends(get_current_user)
):
    """Record an officer's edits. The generated text is preserved separately."""
    from backend import packages, sar
    from backend.auth import audit

    packages.require("sar_draft")
    text = body.get("text")
    if text is None:
        raise HTTPException(422, "Body must contain a 'text' field.")
    draft = await sar.revise(draft_id, str(text), actor=user.username)
    await audit("sar.revise", actor=user.username, target=f"sar:{draft_id}")
    return draft


@app.post("/analyses/sar/{draft_id}/decision", tags=["sar"])
async def decide_sar_draft(
    draft_id: int, body: dict, user: User = Depends(get_current_user)
):
    """Approve or reject a draft.

    Approval attributes the text to this user. It does not file the report —
    filing is a separate, deliberate act in the institution's own system.
    """
    from backend import packages, sar
    from backend.auth import audit

    packages.require("sar_draft")
    if "approve" not in body:
        raise HTTPException(422, "Body must contain an 'approve' boolean.")

    approve = bool(body["approve"])
    draft = await sar.decide(
        draft_id, approve, actor=user.username, note=body.get("note")
    )
    await audit(
        "sar.approve" if approve else "sar.reject",
        actor=user.username,
        target=f"sar:{draft_id}",
        detail=body.get("note"),
    )
    return draft


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

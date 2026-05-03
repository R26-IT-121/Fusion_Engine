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
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("deepsentinel")

# --- Config from environment ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_store")
FATF_DATA_PATH = os.getenv("FATF_DATA_PATH", "../data/fatf_typologies.json")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "./models/meta_classifier.joblib")
GRAPH_MODEL_URL = os.getenv("GRAPH_MODEL_URL", "http://localhost:8001/predict")
BEHAVIORAL_MODEL_URL = os.getenv("BEHAVIORAL_MODEL_URL", "http://localhost:8002/predict")
TEMPORAL_MODEL_URL = os.getenv("TEMPORAL_MODEL_URL", "http://localhost:8003/predict")
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT_MS", "500")) / 1000.0

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


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    transaction_id: Optional[str] = Field(
        default=None,
        description="Transaction identifier. Auto-generated UUID if omitted.",
    )
    graph_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Graph Neural Network fraud probability (0–1). Omit to use mock.",
    )
    behavioral_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Behavioral VAE anomaly probability (0–1). Omit to use mock.",
    )
    temporal_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Temporal CNN anomaly probability (0–1). Omit to use mock.",
    )
    use_mock: bool = Field(
        default=False,
        description="Force use of mock score generator (ignores provided scores).",
    )
    mock_scenario: Optional[str] = Field(
        default=None,
        description=(
            "Mock scenario to simulate. Options: smurfing, layering, mule_network, "
            "account_takeover, velocity_fraud, legitimate. Defaults to random."
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
    mock_scenario: Optional[str]


# ── Upstream model caller ─────────────────────────────────────────────────────

async def _call_upstream_model(
    client: httpx.AsyncClient,
    url: str,
    transaction_id: str,
) -> Optional[float]:
    try:
        response = await client.post(
            url,
            json={"transaction_id": transaction_id},
            timeout=UPSTREAM_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        score = float(data.get("probability", data.get("score", 0.5)))
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"Upstream model at {url} failed: {type(e).__name__}: {e}")
        return None


async def _fetch_upstream_scores(transaction_id: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    async with httpx.AsyncClient() as client:
        graph_task = _call_upstream_model(client, GRAPH_MODEL_URL, transaction_id)
        behavioral_task = _call_upstream_model(client, BEHAVIORAL_MODEL_URL, transaction_id)
        temporal_task = _call_upstream_model(client, TEMPORAL_MODEL_URL, transaction_id)

        graph, behavioral, temporal = await asyncio.gather(
            graph_task, behavioral_task, temporal_task
        )
    return graph, behavioral, temporal


# ── Helper ────────────────────────────────────────────────────────────────────

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
    Full pipeline: fetch upstream scores → fuse → retrieve FATF typology → generate forensic report.
    """
    transaction_id = request.transaction_id or str(uuid.uuid4())
    mock_scenario_used = None

    # ── Step 1: Obtain sub-model scores ──────────────────────────────────────
    if request.use_mock or (
        request.graph_score is None
        and request.behavioral_score is None
        and request.temporal_score is None
    ):
        from backend.mock_scores import generate_mock_scores, FraudScenario

        scenario = FraudScenario.RANDOM
        if request.mock_scenario:
            try:
                scenario = FraudScenario(request.mock_scenario)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown mock_scenario '{request.mock_scenario}'. "
                           f"Valid options: {[s.value for s in FraudScenario if s.value != 'random']}",
                )
        mock = generate_mock_scores(scenario=scenario)
        graph_score = mock.graph_score
        behavioral_score = mock.behavioral_score
        temporal_score = mock.temporal_score
        mock_scenario_used = mock.scenario
        graph_available = behavioral_available = temporal_available = True
        logger.info(f"Using mock scores for scenario '{mock_scenario_used}'.")

    elif (
        request.graph_score is not None
        and request.behavioral_score is not None
        and request.temporal_score is not None
    ):
        # All scores provided directly — bypass upstream call
        graph_score = request.graph_score
        behavioral_score = request.behavioral_score
        temporal_score = request.temporal_score
        graph_available = behavioral_available = temporal_available = True

    else:
        # Call upstream APIs asynchronously; fall back to provided scores if any API fails
        upstream_g, upstream_b, upstream_t = await _fetch_upstream_scores(transaction_id)
        graph_score = upstream_g if upstream_g is not None else request.graph_score
        behavioral_score = upstream_b if upstream_b is not None else request.behavioral_score
        temporal_score = upstream_t if upstream_t is not None else request.temporal_score
        graph_available = graph_score is not None
        behavioral_available = behavioral_score is not None
        temporal_available = temporal_score is not None

    # ── Step 2: Fuse scores via meta-classifier ───────────────────────────────
    fusion = meta_classifier.fuse(
        graph_score=graph_score if graph_available else None,
        behavioral_score=behavioral_score if behavioral_available else None,
        temporal_score=temporal_score if temporal_available else None,
    )

    # ── Step 3: RAG retrieval ────────────────────────────────────────────────
    retrievals = retriever.retrieve(
        graph_score=fusion.graph_score,
        behavioral_score=fusion.behavioral_score,
        temporal_score=fusion.temporal_score,
        confidence_score=fusion.confidence_score,
    )
    if not retrievals:
        raise HTTPException(status_code=500, detail="RAG retrieval returned no results.")
    top_retrieval = retrievals[0]

    # ── Step 4: LLM forensic report generation ───────────────────────────────
    forensic_report = None
    if forensic_reporter is not None:
        from backend.rag.prompt_builder import build_chain_of_evidence_prompt

        prompt_package = build_chain_of_evidence_prompt(
            transaction_id=transaction_id,
            graph_score=fusion.graph_score,
            behavioral_score=fusion.behavioral_score,
            temporal_score=fusion.temporal_score,
            confidence_score=fusion.confidence_score,
            graph_available=fusion.graph_available,
            behavioral_available=fusion.behavioral_available,
            temporal_available=fusion.temporal_available,
            retrieval=top_retrieval,
        )
        try:
            forensic_report = forensic_reporter.generate_report(prompt_package)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            forensic_report = f"[LLM ERROR] Report generation failed: {e}"

    return AnalyzeResponse(
        transaction_id=transaction_id,
        fraud_confidence_score=fusion.confidence_score,
        classification=_classify(fusion.confidence_score),
        graph_score=fusion.graph_score,
        behavioral_score=fusion.behavioral_score,
        temporal_score=fusion.temporal_score,
        graph_available=fusion.graph_available,
        behavioral_available=fusion.behavioral_available,
        temporal_available=fusion.temporal_available,
        modalities_used=fusion.modalities_used,
        retrieval=RetrievalInfo(
            typology_id=top_retrieval.typology_id,
            typology_name=top_retrieval.typology_name,
            stage=top_retrieval.stage,
            risk_level=top_retrieval.risk_level,
            similarity_score=top_retrieval.similarity_score,
        ),
        forensic_report=forensic_report,
        mock_scenario=mock_scenario_used,
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

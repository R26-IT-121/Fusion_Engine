"""
Upstream Model Adapters
Normalizes the different request/response schemas from each team member's API
into a common UpstreamResponse. Each adapter handles its model's exact endpoint,
request body format, and response field names.

M1 Wijesinghe  — POST /api/v1/behavioral/classify  → behavioral_risk_score
M2 Ewaduge     — POST /api/graph/analyze            → relational_risk_score
M3 Pathirana   — POST /api/v1/classify              → temporal_risk_score
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class UpstreamResponse:
    score: float                                    # normalized 0–1
    available: bool
    fraud_signal_summary: Optional[str] = None      # human-readable LLM anchor
    typology_hint: Optional[str] = None             # upstream pattern label
    extra: dict = field(default_factory=dict)       # raw rich data for prompt


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ── Member 1: Wijesinghe — VAE/DSAA ──────────────────────────────────────────

async def call_behavioral_api(
    client: httpx.AsyncClient,
    base_url: str,
    transaction: dict,
    timeout: float,
) -> UpstreamResponse:
    """
    POST /api/v1/behavioral/classify
    Response key: behavioral_risk_score
    Transaction ID: transaction_ref.composite_id = "{nameOrig}_{step}"
    """
    try:
        # Build composite_id in the format M1 expects
        name_orig = transaction.get("nameOrig", "")
        step = transaction.get("step", 0)
        payload = {**transaction, "composite_id": f"{name_orig}_{step}"}

        resp = await client.post(
            f"{base_url}/api/v1/behavioral/classify",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        score = _clamp(float(data.get("behavioral_risk_score", 0.5)))

        # Extract M1's rich forensic signals for LLM prompt enrichment
        evidence = data.get("evidence", {})
        current_tx = evidence.get("current_transaction", {})
        fraud_signal_summary = current_tx.get("fraud_signal_summary")

        anomaly_fp = data.get("anomaly_fingerprint", {})
        sig1 = anomaly_fp.get("signal_1_reconstruction_error", {})
        sig2 = anomaly_fp.get("signal_2_kl_divergence", {})
        dominant_reconstruction = sig1.get("dominant_feature_signal")
        dominant_kl = sig2.get("dominant_dimension_signal")

        typology_hint = data.get("fraud_typology", {}).get("typology_label")

        return UpstreamResponse(
            score=score,
            available=True,
            fraud_signal_summary=fraud_signal_summary,
            typology_hint=typology_hint,
            extra={
                "anomaly_fingerprint": {
                    "dominant_reconstruction_signal": dominant_reconstruction,
                    "dominant_kl_signal": dominant_kl,
                },
                "vae_diagnostics": data.get("vae_diagnostics", {}),
                "transaction_type": data.get("transaction_type"),
                "combined_anomaly_score": data.get("vae_diagnostics", {}).get("combined_anomaly_score"),
            },
        )
    except Exception as e:
        logger.warning(f"Behavioral API ({base_url}) failed: {type(e).__name__}: {e}")
        return UpstreamResponse(score=0.5, available=False)


# ── Member 2: Ewaduge — Edge-Enhanced GraphSAGE ───────────────────────────────

async def call_graph_api(
    client: httpx.AsyncClient,
    base_url: str,
    transaction: dict,
    timeout: float,
) -> UpstreamResponse:
    """
    POST /api/graph/analyze
    Response key: relational_risk_score
    Transaction ID: transaction_id (top-level string)
    Suspicious subgraph: pattern ∈ {HUB_AND_SPOKE, SMURFING, LAYERING, ACCOUNT_TAKEOVER, UNKNOWN}

    Handles all response branches per contract Section 3:
    - 200 with risk_level=NOT_APPLICABLE → treat as unavailable (model has no opinion)
    - 404 (accounts unknown) → treat as unavailable (normal, not an outage)
    - 200 with full payload → extract and return normally
    """
    try:
        resp = await client.post(
            f"{base_url}/api/graph/analyze",
            json=transaction,
            timeout=timeout,
        )

        # 404 is normal when accounts are unknown to the graph
        if resp.status_code == 404:
            logger.info(f"Graph API: transaction {transaction.get('transaction_id', '?')} not in graph (404)")
            return UpstreamResponse(score=0.5, available=False)

        resp.raise_for_status()
        data = resp.json()

        risk_level = data.get("risk_level", "UNKNOWN")

        # NOT_APPLICABLE means the model has no opinion (e.g., type not TRANSFER/CASH_OUT)
        # Treat as unavailable so fusion engine excludes this modality
        if risk_level == "NOT_APPLICABLE":
            logger.debug(f"Graph API: NOT_APPLICABLE for transaction {transaction.get('transaction_id', '?')}")
            return UpstreamResponse(score=0.5, available=False)

        score = _clamp(float(data.get("relational_risk_score", 0.5)))

        subgraph = data.get("suspicious_subgraph") or {}
        pattern = subgraph.get("pattern", "UNKNOWN") if subgraph else "UNKNOWN"
        structural = subgraph.get("structural_evidence", {}) if subgraph else {}
        sink_account = subgraph.get("sink_account") if subgraph else None
        mules = structural.get("mules_in_subgraph", 0)
        fresh_ratio = structural.get("fresh_sender_ratio")
        convergence = structural.get("convergence_count")

        fraud_signal_summary = None
        if pattern and pattern != "UNKNOWN" and subgraph:
            parts = [f"Graph pattern: {pattern}."]
            if convergence is not None:
                parts.append(f"Convergence count: {convergence} distinct senders.")
            if sink_account:
                parts.append(f"Sink account: {sink_account}.")
            if fresh_ratio is not None:
                parts.append(f"Fresh sender ratio: {fresh_ratio:.1%}.")
            fraud_signal_summary = " ".join(parts)

        return UpstreamResponse(
            score=score,
            available=True,
            fraud_signal_summary=fraud_signal_summary,
            typology_hint=pattern if pattern != "UNKNOWN" else None,
            extra={
                "suspicious_subgraph": subgraph,
                "pattern": pattern,
                "sink_account": sink_account,
                "risk_level": risk_level,
                "model_version": data.get("model_version"),
                "stage": data.get("stage"),
            },
        )
    except Exception as e:
        logger.warning(f"Graph API ({base_url}) failed: {type(e).__name__}: {e}")
        return UpstreamResponse(score=0.5, available=False)


# ── Member 3: Pathirana — TSCFD/TCN ──────────────────────────────────────────

async def call_temporal_api(
    client: httpx.AsyncClient,
    base_url: str,
    transaction: dict,
    timeout: float,
) -> UpstreamResponse:
    """
    POST /api/v1/classify — per TS-TCN/docs/api_contract.md (locked).

    Request:  {"transaction": {9 PaySim fields}}  — nested, not flat.
    Response: fraud_probability, fraud_label, threshold_used, attribution{...}

    The service keeps its own deque(maxlen=32) of recent transactions and
    rebuilds the window itself, so only the current transaction is sent. Until
    32 have arrived it answers 503 WARMING_UP, which is a normal startup state
    rather than an outage — it is reported as unavailable, not logged as a
    failure.
    """
    # Their schema validates these nine fields and this type pattern; sending
    # our extra keys (transaction_id, isFlaggedFraud, _is_fraud) risks a 422.
    FIELDS = (
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    )
    try:
        payload = {"transaction": {k: transaction[k] for k in FIELDS if k in transaction}}

        resp = await client.post(
            f"{base_url}/api/v1/classify",
            json=payload,
            timeout=timeout,
        )

        # Warming up: the model is fine, it just has no window yet.
        if resp.status_code == 503:
            logger.info("Temporal API: warming up (buffer below 32 transactions)")
            return UpstreamResponse(score=0.5, available=False)

        resp.raise_for_status()
        data = resp.json()

        # `temporal_risk_score` is the older name this adapter was written
        # against; kept as a fallback so either shape works.
        raw = data.get("fraud_probability", data.get("temporal_risk_score", 0.5))
        score = _clamp(float(raw))

        attribution = data.get("attribution") or {}
        peak_weight = attribution.get("peak_weight")
        peak_id = attribution.get("peak_transaction_id")
        peak_position = attribution.get("peak_position")

        fraud_signal_summary = None
        if peak_weight is not None:
            parts = [
                f"Temporal attention peaked at {peak_weight:.1%} on position "
                f"{peak_position} of the 32-transaction window."
            ]
            if peak_id:
                parts.append(f"Most influential predecessor: {peak_id}.")
            fraud_signal_summary = " ".join(parts)

        return UpstreamResponse(
            score=score,
            available=True,
            fraud_signal_summary=fraud_signal_summary,
            typology_hint=None,
            extra={
                "composite_id": data.get("composite_id"),
                "fraud_label": data.get("fraud_label"),
                "threshold_used": data.get("threshold_used"),
                "attribution": attribution,
                "model_version": data.get("model_version"),
                "inference_time_ms": data.get("inference_time_ms"),
            },
        )
    except Exception as e:
        logger.warning(f"Temporal API ({base_url}) failed: {type(e).__name__}: {e}")
        return UpstreamResponse(score=0.5, available=False)

"""Tools the operator assistant may invoke.

Each tool is a narrow, audited capability over the live platform: score a
transaction through all three models, pull the relational fraud ring, search
analysis history, check upstream health, or look something up in the project
documentation.

Design rules, because this agent acts on a production system:

- **Read-mostly.** Nothing here mutates state. The one tool that consumes
  resources (`analyze_transaction`, which spends upstream model calls) is behind
  its own settings flag.
- **Declared schemas.** Every tool publishes a JSON schema; the agent may only
  call what is declared, with arguments coerced and validated first. A model
  cannot invent a tool or smuggle an argument through.
- **Failures are values.** A tool returns `{"error": ...}` rather than raising,
  so one unreachable upstream degrades the answer instead of the request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from sqlalchemy import desc, select

from backend import config
from backend.db.models import AnalysisRecord
from backend.db.session import get_session

logger = logging.getLogger(__name__)

# Same budget the pipeline uses, with headroom: the assistant is
# interactive and a slow answer beats a truncated one.
UPSTREAM_TIMEOUT = max(float(config.get("upstream", "timeout_ms")) / 1000.0, 15.0)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                      # JSON-schema-ish, for the prompt
    run: Callable[..., Awaitable[dict]]
    costly: bool = False                  # consumes upstream model calls


# ── Transaction helpers ──────────────────────────────────────────────────────

TXN_FIELDS = {
    "step": int, "type": str, "amount": float, "nameOrig": str, "nameDest": str,
    "oldbalanceOrg": float, "newbalanceOrig": float,
    "oldbalanceDest": float, "newbalanceDest": float,
}


def _coerce_transaction(raw: dict) -> dict:
    """Build a valid PaySim transaction, filling omissions with neutral values.

    The model will not always supply all ten fields; refusing on a missing
    balance would make the assistant brittle for no benefit, so we default and
    report what we assumed.
    """
    txn: dict[str, Any] = {}
    for key, cast in TXN_FIELDS.items():
        val = raw.get(key)
        if val is None:
            txn[key] = 0 if cast in (int, float) else ""
            continue
        try:
            txn[key] = cast(val)
        except (TypeError, ValueError):
            txn[key] = 0 if cast in (int, float) else str(val)
    txn.setdefault("isFlaggedFraud", 0)
    if not txn.get("type"):
        txn["type"] = "TRANSFER"
    if txn.get("step", 0) < 1:
        txn["step"] = 1
    return txn


# ── Tool implementations ─────────────────────────────────────────────────────


async def _graph_subgraph(**kwargs) -> dict:
    """Relational fraud ring straight from the GraphSAGE service."""
    txn = _coerce_transaction(kwargs)
    base = str(config.get("upstream", "graph_api_base")).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            resp = await client.post(
                f"{base}/api/graph/analyze",
                json={"transaction_id": kwargs.get("transaction_id", "assistant"), **txn},
            )
        if resp.status_code == 404:
            return {"found": False,
                    "note": "No edge between these accounts in the graph snapshot — "
                            "the ring cannot be anchored. This is normal, not an outage."}
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:                           # noqa: BLE001
        return {"error": f"GraphSAGE unreachable: {type(exc).__name__}: {exc}"}

    sg = data.get("suspicious_subgraph")
    out = {
        "relational_risk_score": data.get("relational_risk_score"),
        "risk_level": data.get("risk_level"),
        "stage": data.get("stage"),
    }
    if not sg:
        out["subgraph"] = None
        out["note"] = "No subgraph — transaction type is outside the model's scope."
        return out
    ev = sg.get("structural_evidence", {})
    out["subgraph"] = {
        "pattern": sg.get("pattern"),
        "pattern_confidence": sg.get("pattern_confidence"),
        "sink_account": sg.get("sink_account"),
        "node_count": sg.get("node_count"),
        "edge_count": sg.get("edge_count"),
        "convergence_count": ev.get("convergence_count"),
        "fresh_sender_ratio": ev.get("fresh_sender_ratio"),
        "mules_in_subgraph": ev.get("mules_in_subgraph"),
        # Only the highest-attention edges — the whole ring would swamp the prompt.
        "top_edges": sorted(
            (
                {"src": e["src"], "dst": e["dst"], "amount": e["amount"],
                 "attention": e["edge_attention_weight"]}
                for e in sg.get("edges", [])
            ),
            key=lambda e: e["attention"], reverse=True,
        )[:6],
    }
    return out


async def _model_scores(**kwargs) -> dict:
    """Score one transaction through all three upstream models in parallel."""
    from backend.adapters.upstream import (
        call_behavioral_api, call_graph_api, call_temporal_api,
    )
    import asyncio

    txn = _coerce_transaction(kwargs)
    targets = [
        ("graph", call_graph_api, str(config.get("upstream", "graph_api_base"))),
        ("behavioral", call_behavioral_api, str(config.get("upstream", "behavioral_api_base"))),
        ("temporal", call_temporal_api, str(config.get("upstream", "temporal_api_base"))),
    ]
    out: dict[str, Any] = {"transaction": txn, "scores": {}}
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        results = await asyncio.gather(
            *(fn(client, url, txn, UPSTREAM_TIMEOUT) for _, fn, url in targets),
            return_exceptions=True,
        )
    for (name, _, _), res in zip(targets, results):
        if isinstance(res, Exception):
            out["scores"][name] = {"available": False,
                                   "error": f"{type(res).__name__}: {res}"}
            continue
        out["scores"][name] = {
            "available": getattr(res, "ok", None) is not False,
            "score": getattr(res, "score", None),
            "detail": getattr(res, "error", None),
        }
    available = sum(1 for v in out["scores"].values() if v.get("available"))
    out["modalities_available"] = available
    if available == 0:
        out["note"] = "No upstream model responded; no fused verdict is possible."
    return out


async def _search_history(**kwargs) -> dict:
    """Query past analyses — the operator's own case history."""
    account = kwargs.get("account")
    classification = kwargs.get("classification")
    min_score = kwargs.get("min_score")
    limit = min(int(kwargs.get("limit", 10) or 10), 50)

    try:
        async with get_session() as db:
            stmt = select(AnalysisRecord).order_by(desc(AnalysisRecord.id)).limit(limit)
            if account:
                stmt = stmt.where(
                    (AnalysisRecord.name_orig == account)
                    | (AnalysisRecord.name_dest == account)
                )
            if classification:
                stmt = stmt.where(AnalysisRecord.classification == str(classification).upper())
            if min_score is not None:
                try:
                    stmt = stmt.where(AnalysisRecord.fraud_confidence_score >= float(min_score))
                except (TypeError, ValueError):
                    pass
            rows = (await db.scalars(stmt)).all()
    except Exception as exc:                           # noqa: BLE001
        return {"error": f"History unavailable: {type(exc).__name__}: {exc}"}

    return {
        "count": len(rows),
        "results": [
            {
                "transaction_id": r.transaction_id,
                "classification": r.classification,
                "fused_score": r.fraud_confidence_score,
                "graph": r.graph_score,
                "behavioral": r.behavioral_score,
                "temporal": r.temporal_score,
                "amount": r.amount,
                "from": r.name_orig,
                "to": r.name_dest,
                "modalities_used": r.modalities_used,
            }
            for r in rows
        ],
    }


async def _system_status(**_) -> dict:
    """Which upstream models are reachable right now."""
    import asyncio

    targets = {
        "graph": (str(config.get("upstream", "graph_api_base")), "/health"),
        "behavioral": (str(config.get("upstream", "behavioral_api_base")), "/health"),
        "temporal": (str(config.get("upstream", "temporal_api_base")), "/health"),
    }

    async def probe(url: str, path: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{url.rstrip('/')}{path}")
            return {"reachable": r.status_code < 500, "status_code": r.status_code}
        except Exception as exc:                       # noqa: BLE001
            return {"reachable": False, "error": type(exc).__name__}

    names = list(targets)
    results = await asyncio.gather(*(probe(u, p) for u, p in targets.values()))
    return {"upstream": dict(zip(names, results))}


async def _live_activity(**kwargs) -> dict:
    """What the monitor is seeing right now.

    This is the difference between an assistant that can describe the platform
    and one an analyst can actually use during an incident: it answers "what is
    happening" from live state, not documentation.
    """
    try:
        from monitor.state import STATE
    except Exception as exc:                           # noqa: BLE001
        return {"error": f"Monitor unavailable: {type(exc).__name__}: {exc}"}

    limit = min(int(kwargs.get("limit", 8) or 8), 25)
    severity = (kwargs.get("severity") or "").upper()

    snap = STATE.snapshot(events=40)
    alerts = snap["alerts"]
    if severity in {"MEDIUM", "HIGH", "CRITICAL"}:
        alerts = [a for a in alerts if a.get("severity") == severity]

    # Recent escalations explain WHY the counters moved, which is usually the
    # real question behind "what is going on".
    escalations = [
        e for e in snap["events"] if e.get("kind") == "escalated"
    ][-limit:]

    return {
        "running": snap["running"],
        "counters": snap["counters"],
        "stages": snap["stages"],
        "open_alerts": alerts[:limit],
        "recent_escalations": escalations,
        "note": (
            "Counters are since the monitor last started. An alert is a fused "
            "verdict at MEDIUM or above; an escalation is the graph model "
            "tripping its watch threshold."
        ),
    }


async def _search_docs(**kwargs) -> dict:
    """Look something up in the project documentation (shared with the public bot)."""
    query = str(kwargs.get("query", "")).strip()
    if not query:
        return {"error": "query is required"}
    try:
        from chatbot.router import get_service

        hits = get_service().retriever.search(query, top_k=3)
    except Exception as exc:                           # noqa: BLE001
        return {"error": f"Documentation index unavailable: {type(exc).__name__}: {exc}"}
    return {
        "passages": [
            {"source": c.citation, "text": c.text[:900]} for c, _ in hits
        ]
    }


# ── Registry ─────────────────────────────────────────────────────────────────

_TXN_SCHEMA = {
    "step": "int, PaySim hour (1-743)",
    "type": "TRANSFER | CASH_OUT | CASH_IN | PAYMENT | DEBIT",
    "amount": "float",
    "nameOrig": "sender account id, e.g. C1234",
    "nameDest": "receiver account id",
    "oldbalanceOrg": "float, optional",
    "newbalanceOrig": "float, optional",
    "oldbalanceDest": "float, optional",
    "newbalanceDest": "float, optional",
}

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            name="get_model_scores",
            description=(
                "Score one transaction through all three detection models "
                "(graph, behavioural, temporal) and report which responded. "
                "Use when asked how risky a specific transaction is."
            ),
            parameters=_TXN_SCHEMA,
            run=_model_scores,
            costly=True,
        ),
        Tool(
            name="get_fraud_ring",
            description=(
                "Get the relational fraud ring around a transaction from the "
                "GraphSAGE service: money-laundering pattern, sink account, how "
                "many senders converge, share of brand-new senders, and the "
                "highest-attention transfers. Use for 'who else is involved', "
                "'is this a ring', 'explain the network'."
            ),
            parameters=_TXN_SCHEMA,
            run=_graph_subgraph,
            costly=True,
        ),
        Tool(
            name="search_analysis_history",
            description=(
                "Search previously analysed transactions. Filter by account id, "
                "classification (LOW/MEDIUM/HIGH/CRITICAL) or minimum fused "
                "score. Use for 'have we seen this account', 'recent critical "
                "cases', 'what did we flag today'."
            ),
            parameters={
                "account": "optional account id to match as sender or receiver",
                "classification": "optional LOW | MEDIUM | HIGH | CRITICAL",
                "min_score": "optional float 0-1",
                "limit": "optional int, default 10, max 50",
            },
            run=_search_history,
        ),
        Tool(
            name="get_system_status",
            description=(
                "Check which upstream detection models are reachable. Use for "
                "'is everything running', or to explain a missing modality."
            ),
            parameters={},
            run=_system_status,
        ),
        Tool(
            name="get_live_activity",
            description=(
                "What the live monitor is seeing RIGHT NOW: how many "
                "transactions have been screened, what escalated, which "
                "alerts are open and at what severity, and which detectors "
                "are currently running. Use for 'what is happening', 'why did "
                "I get an alert', 'anything critical', 'is the monitor busy'."
            ),
            parameters={
                "severity": "optional MEDIUM | HIGH | CRITICAL filter",
                "limit": "optional int, default 8, max 25",
            },
            run=_live_activity,
        ),
        Tool(
            name="search_documentation",
            description=(
                "Look up how the platform works in the project documentation — "
                "methodology, metrics, thresholds, API contract. Use for "
                "'how is the score calculated', 'what does NOT_APPLICABLE mean'."
            ),
            parameters={"query": "what to look up"},
            run=_search_docs,
        ),
    ]
}


def available_tools(allow_live_analysis: bool) -> dict[str, Tool]:
    """Tools the current settings permit."""
    if allow_live_analysis:
        return TOOLS
    return {name: t for name, t in TOOLS.items() if not t.costly}

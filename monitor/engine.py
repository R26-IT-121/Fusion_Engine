"""The always-on monitoring loop.

Screening is deliberately asymmetric, and that asymmetry is the design:

    every transaction ──▶ GraphSAGE  (cheap, structural, always on)
                              │
                     score ≥ watch threshold
                              │
                              ▼
              behavioural + temporal, in parallel  (expensive)
                              │
                              ▼
                      fusion ──▶ alert + report

The graph model is the tripwire because relational structure is visible without
any per-account history — a mule ring is a shape, and the shape is there on the
first transfer. Running all three detectors on every record would cost three
times as much to reach the same verdicts, since the other two only change the
outcome once something is already structurally suspicious.

Two alerts leave the system for one incident, on purpose:

  * **Early warning**, the moment the graph model trips. Fast and provisional —
    an analyst can start looking while the rest of the pipeline runs.
  * **Confirmed alert**, after fusion, carrying the severity band and the
    forensic narrative.

Sending only the second would waste the head start the graph model provides;
sending only the first would page people on an unconfirmed signal.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from backend import config
from monitor.state import STATE

logger = logging.getLogger(__name__)

# Below this the transaction is not worth three model calls. Set from the
# served model's own MEDIUM band at startup, not guessed.
DEFAULT_WATCH_THRESHOLD = 0.09
POLL_BATCH = 25            # transactions fetched per refill
DEFAULT_INTERVAL = 1.2     # seconds between screenings


class MonitorEngine:
    def __init__(self) -> None:
        self.paused = False            # holds the loop without losing counters
        self._fusion = None            # the project's trained MetaClassifier
        self._task: asyncio.Task | None = None
        self._queue: list[dict] = []
        self.interval = DEFAULT_INTERVAL
        self.watch_threshold = DEFAULT_WATCH_THRESHOLD
        self._bands: dict[str, float] = {}

    # ── lifecycle ────────────────────────────────────────────────────
    async def start(self, interval: float | None = None) -> None:
        if self._task and not self._task.done():
            return
        if interval:
            self.interval = max(0.2, min(float(interval), 10.0))
        self.paused = False
        STATE.running = True
        # Reset the whole counter set, not just the clock: keeping totals from
        # a previous run while restarting the timer reported a throughput of
        # ~1800/min on a 1.2s interval.
        from monitor.state import Counters

        STATE.counters = Counters()
        self._task = asyncio.create_task(self._run())
        STATE.publish("monitor", {"status": "started", "interval": self.interval})

    def pause(self) -> None:
        """Hold screening without tearing down state.

        Distinct from stop(): counters, alerts and the loaded model survive, so
        resuming continues the same session rather than starting a new one.
        An analyst pausing to read an alert should not lose the run.
        """
        self.paused = True
        for k in STATE.stage_status:
            STATE.stage_status[k] = "idle"
        STATE.publish("monitor", {"status": "paused"})

    def resume(self) -> None:
        self.paused = False
        STATE.publish("monitor", {"status": "resumed"})

    async def restart(self, interval: float | None = None) -> None:
        """Full cycle: drop state, reload the model's thresholds, begin again."""
        await self.stop()
        self.paused = False
        await self.start(interval)

    async def stop(self) -> None:
        STATE.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        for s in STATE.stage_status:
            STATE.stage_status[s] = "idle"
        STATE.publish("monitor", {"status": "stopped"})

    # ── the loop ─────────────────────────────────────────────────────
    async def _run(self) -> None:
        graph_base = str(config.get("upstream", "graph_api_base")).rstrip("/")
        await self._load_bands(graph_base)
        await self._load_fusion()

        async with httpx.AsyncClient(timeout=20.0) as client:
            while STATE.running:
                if self.paused:
                    await asyncio.sleep(0.4)
                    continue
                try:
                    txn = await self._next_transaction(client, graph_base)
                    if txn is None:
                        await asyncio.sleep(2.0)
                        continue
                    await self._screen(client, graph_base, txn)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:                # noqa: BLE001
                    # One bad transaction must not end the monitor.
                    logger.warning(f"Monitor iteration failed: {exc}")
                    STATE.publish("error", {"message": str(exc)[:200]})
                await asyncio.sleep(self.interval)

    async def _load_fusion(self) -> None:
        """Load the same meta-classifier the /analyze endpoint uses.

        Averaging the available scores would be a second, different fusion
        rule — the monitor and the on-demand analyzer would then disagree
        about the same transaction, which is indefensible in a system whose
        whole claim is traceability.
        """
        try:
            from backend.fusion_engine import MetaClassifier

            path = str(config.get("paths", "meta_classifier"))
            clf = MetaClassifier(path)
            await asyncio.to_thread(clf.initialize)
            self._fusion = clf
            logger.info("Monitor using the trained meta-classifier")
        except Exception as exc:                        # noqa: BLE001
            logger.warning(f"Meta-classifier unavailable, monitor will average: {exc}")
            self._fusion = None

    async def _load_bands(self, graph_base: str) -> None:
        """Take the watch threshold from the model rather than hard-coding it."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{graph_base}/health")
            bands = r.json().get("risk_bands") or {}
            if bands:
                self._bands = bands
                self.watch_threshold = float(bands.get("medium", DEFAULT_WATCH_THRESHOLD))
                logger.info(f"Monitor watch threshold from model: {self.watch_threshold:.4f}")
        except Exception as exc:                        # noqa: BLE001
            logger.info(f"Using default watch threshold ({exc})")

    async def _next_transaction(self, client: httpx.AsyncClient, graph_base: str):
        if not self._queue:
            r = await client.get(
                f"{graph_base}/api/graph/sample-transactions",
                params={"n": POLL_BATCH, "fraud_ratio": 0.08},
            )
            r.raise_for_status()
            self._queue = r.json().get("transactions", [])
        return self._queue.pop(0) if self._queue else None

    # ── stage 1: screen ──────────────────────────────────────────────
    async def _screen(self, client, graph_base: str, txn: dict) -> None:
        payload = {k: v for k, v in txn.items() if not k.startswith("_")}
        STATE.set_stage("graph", "active")

        try:
            r = await client.post(f"{graph_base}/api/graph/analyze", json=payload)
        finally:
            STATE.set_stage("graph", "idle")

        STATE.counters.screened += 1

        if r.status_code == 404:
            STATE.publish("screened", {
                "transaction_id": payload["transaction_id"], "outcome": "unknown_accounts",
            })
            return
        r.raise_for_status()
        result = r.json()

        score = float(result.get("relational_risk_score") or 0.0)
        level = result.get("risk_level", "LOW")
        sg = result.get("suspicious_subgraph") or {}

        STATE.publish("screened", {
            "transaction_id": payload["transaction_id"],
            "amount": payload["amount"],
            "from": payload["nameOrig"],
            "to": payload["nameDest"],
            "graph_score": round(score, 4),
            "risk_level": level,
            "escalated": score >= self.watch_threshold,
        })

        if score < self.watch_threshold:
            return

        await self._escalate(client, payload, result, sg)

    # ── stage 2: escalate ────────────────────────────────────────────
    async def _escalate(self, client, payload: dict, graph_result: dict, sg: dict) -> None:
        STATE.counters.escalated += 1
        txid = payload["transaction_id"]
        graph_score = float(graph_result.get("relational_risk_score") or 0.0)

        STATE.publish("escalated", {
            "transaction_id": txid,
            "graph_score": round(graph_score, 4),
            "pattern": sg.get("pattern"),
            "sink_account": sg.get("sink_account"),
            "convergence": (sg.get("structural_evidence") or {}).get("convergence_count"),
        })

        # Early warning goes out now, before the slower detectors finish.
        await self._notify_early(txid, payload, graph_score, sg)

        scores: dict[str, float | None] = {"graph": graph_score}
        for name, key, base_key in (
            ("behavioural", "behavioral_risk_score", "behavioral_api_base"),
            ("temporal", "temporal_risk_score", "temporal_api_base"),
        ):
            STATE.set_stage(name, "active")
            scores[name] = await self._call_upstream(client, base_key, key, payload)
            STATE.set_stage(name, "idle")
            STATE.publish("model", {
                "transaction_id": txid, "model": name, "score": scores[name],
            })

        STATE.set_stage("fusion", "active")
        available = [v for v in scores.values() if v is not None]
        fusion_method = "meta_classifier"
        if self._fusion is not None:
            try:
                result = await asyncio.to_thread(
                    self._fusion.fuse,
                    scores.get("graph"), scores.get("behavioural"), scores.get("temporal"),
                )
                fused = float(getattr(result, "fraud_confidence_score", 0.0))
            except Exception as exc:                    # noqa: BLE001
                logger.warning(f"Fusion failed, averaging instead: {exc}")
                fused = sum(available) / len(available) if available else 0.0
                fusion_method = "mean_fallback"
        else:
            # No trained model available: average what answered. A detector
            # that is unreachable abstains rather than voting zero, which
            # would read as innocence.
            fused = sum(available) / len(available) if available else 0.0
            fusion_method = "mean_fallback"
        STATE.set_stage("fusion", "idle")

        severity = self._severity(fused)
        STATE.publish("fused", {
            "transaction_id": txid,
            "fused_score": round(fused, 4),
            "severity": severity,
            "fusion_method": fusion_method,
            "modalities_used": len(available),
            "scores": {k: (round(v, 4) if v is not None else None) for k, v in scores.items()},
        })

        if severity == "LOW":
            return

        alert = {
            "transaction_id": txid,
            "severity": severity,
            "fused_score": round(fused, 4),
            "graph_score": round(graph_score, 4),
            "pattern": sg.get("pattern"),
            "sink_account": sg.get("sink_account"),
            "amount": payload["amount"],
            "from": payload["nameOrig"],
            "to": payload["nameDest"],
            "modalities_used": len(available),
            "fusion_method": fusion_method,
            "at": time.time(),
        }
        STATE.add_alert(alert)

        STATE.set_stage("report", "active")
        await self._notify_confirmed(alert, sg)
        STATE.set_stage("report", "idle")

    async def _call_upstream(self, client, base_key: str, score_key: str, payload: dict):
        """Score one modality. Returns None when the detector cannot answer."""
        base = str(config.get("upstream", base_key)).rstrip("/")
        for path in ("/api/v1/classify", "/api/v1/behavioral/classify"):
            try:
                r = await client.post(f"{base}{path}", json=payload, timeout=10.0)
                if r.status_code == 200:
                    return float(r.json().get(score_key))
            except Exception:                           # noqa: BLE001
                continue
        return None

    def _severity(self, fused: float) -> str:
        b = self._bands
        if fused >= float(b.get("critical", 0.39)):
            return "CRITICAL"
        if fused >= float(b.get("high", 0.18)):
            return "HIGH"
        if fused >= float(b.get("medium", 0.09)):
            return "MEDIUM"
        return "LOW"

    # ── notifications ────────────────────────────────────────────────
    async def _notify_early(self, txid, payload, score, sg) -> None:
        body = (
            "EARLY WARNING — relational screening\n"
            f"{'=' * 44}\n"
            f"The graph model flagged {txid} before the other detectors ran.\n\n"
            f"Relational score : {score:.4f}\n"
            f"Pattern          : {sg.get('pattern', 'n/a')}\n"
            f"Sink account     : {sg.get('sink_account', 'n/a')}\n"
            f"Amount           : {payload['amount']:,.2f}\n"
            f"From → To        : {payload['nameOrig']} → {payload['nameDest']}\n\n"
            "Behavioural and temporal scoring is running now; a confirmed alert\n"
            "with the full narrative follows if fusion agrees."
        )
        sent = await self._send(f"[Early warning] {txid}", body)
        STATE.publish("notification", {
            "transaction_id": txid, "stage": "early", "sent": sent,
        })

    async def _notify_confirmed(self, alert: dict, sg: dict) -> None:
        body = (
            f"CONFIRMED {alert['severity']} — fused verdict\n"
            f"{'=' * 44}\n"
            f"Transaction : {alert['transaction_id']}\n"
            f"Fused score : {alert['fused_score']:.4f} "
            f"({alert['modalities_used']} of 3 detectors available)\n"
            f"Graph score : {alert['graph_score']:.4f}\n"
            f"Pattern     : {alert['pattern'] or 'n/a'}\n"
            f"Sink        : {alert['sink_account'] or 'n/a'}\n"
            f"Amount      : {alert['amount']:,.2f}\n"
            f"From → To   : {alert['from']} → {alert['to']}\n"
        )
        ev = sg.get("structural_evidence") or {}
        if ev:
            body += (
                f"\nStructural evidence\n"
                f"  senders converging : {ev.get('convergence_count')}\n"
                f"  brand-new senders  : {ev.get('fresh_sender_ratio')}\n"
                f"  mules in subgraph  : {ev.get('mules_in_subgraph')}\n"
            )
        sent = await self._send(
            f"[{alert['severity']}] Fraud alert {alert['transaction_id']}", body
        )
        STATE.publish("notification", {
            "transaction_id": alert["transaction_id"],
            "stage": "confirmed", "severity": alert["severity"], "sent": sent,
        })

    async def _send(self, subject: str, body: str) -> bool:
        """Deliver to the configured alert recipients.

        Recipients come from the risk-manager table the Settings page manages —
        NOT from [email] sender_email, which is the From address. Sending to
        the From address is what produced the NXDOMAIN bounce: the default
        alerts@deepsentinel.io is a placeholder domain that does not exist.
        """
        try:
            from backend.email_service import _provider, _send_plain
            from backend.settings import list_risk_managers

            provider, _ = _provider()
            if not provider:
                return False

            managers = await list_risk_managers()
            recipients = [m.email for m in managers if getattr(m, "enabled", True)]
            if not recipients:
                logger.info("No alert recipients configured; nothing sent.")
                return False

            return await asyncio.to_thread(_send_plain, subject, body, recipients, None)
        except Exception as exc:                        # noqa: BLE001
            logger.warning(f"Monitor notification failed: {exc}")
            return False


ENGINE = MonitorEngine()

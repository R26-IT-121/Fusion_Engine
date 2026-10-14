"""The always-on monitoring loop.

Every transaction goes to all three detectors at once:

                        ┌──▶ GraphSAGE   (structure)  ──┐
    every transaction ──┼──▶ Behaviour   (VAE)        ──┼──▶ fusion ──▶ alert
                        └──▶ Timing      (TCN)        ──┘

This used to be a cascade, with the graph model gating the other two, and the
docstring here described that design long after the code stopped doing it. The
cascade was measured against a 400-transaction replay and it cost half the
frauds: four of the eight it missed scored 0.011–0.107 on the graph model, far
below the gate, while the behavioural model scored those same four between 0.51
and 1.00. A gate in front of an independent detector cannot beat that detector;
it can only hide it. It bought no speed either — the calls were sequential
awaits, so a flagged transaction paid all three in series. Fanned out they cost
whichever one is slowest, once.

A detector that cannot answer **abstains**: its score is excluded from the
fusion rather than counted as low, and the fused confidence is shrunk toward
uncertainty. Absence is not innocence.

Two alerts leave the system for one incident, on purpose:

  * **Early warning**, the moment the graph model trips. Fast and provisional —
    an analyst can start looking while the narrative is still being written.
  * **Confirmed alert**, after fusion, carrying the severity band and the
    forensic report.

Neither the report nor the mail is awaited by the pipeline. Both are slow and
neither changes the verdict; charging them to screening latency once made the
case table report how slow Gmail was as though it were detection time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from backend import config, runlog
from monitor.state import STATE

logger = logging.getLogger(__name__)

# Below this the transaction is not worth three model calls. Set from the
# served model's own MEDIUM band at startup, not guessed.
DEFAULT_WATCH_THRESHOLD = 0.09
POLL_BATCH = 25            # transactions fetched per refill
DEFAULT_INTERVAL = 1.2     # seconds between screenings


# Where the fused verdict changes severity.
#
# These used to fall back to `self._bands` — the risk bands the relational
# model publishes on /health. Those are the operating point for *its* score,
# and reusing them for the fused score is a category error: the two have
# different distributions, and it showed. Measured over a 2,000-transaction
# replay, 0.18, 0.25, 0.39 and 0.55 all selected the identical 49
# transactions, so HIGH and CRITICAL were the same band wearing two names.
#
# The values below sit where the flagged population actually changes:
#
#     band     flagged   recall   false alarms per real fraud
#     0.030        85     27.3%        49 : 1
#     0.090        61     19.7%        48 : 1
#     0.925        35     11.1%        54 : 1
#
# Recall and false-alarm rate are carried over from a fraud-oversampled
# sample; the alert *volume* is stated at PaySim's true 0.129% rate. Note that
# precision barely improves as the band rises — the fused score does not rank
# well at the top end, which is a statement about the detectors and not about
# fusion. An operator can move all three on the Thresholds page.
# The alerting line comes from backend.thresholds, which is where the API and
# the console both read it. Holding a second copy here is how the monitor came
# to fire on a line the page did not display.
from backend.thresholds import DEFAULT_BANDS as FUSED_BANDS


# How long a claim may sit before it is treated as abandoned, and how often to
# look. The claim window has to exceed the slowest realistic screening — one
# transaction costs tens of milliseconds, so three minutes is generous — or a
# live worker's rows would be taken out from under it.
STALE_CLAIM_SECONDS = 180
STALE_SWEEP_SECONDS = 30


# A detector that just failed is left alone for this long. Under the fan-out
# every transaction pays for a dead service, so the breaker is what keeps one
# unreachable model from setting the pace of the whole pipeline.
DETECTOR_BREAKER_SECONDS = 15.0

# Which run-log folder each detector's calls belong in. Keyed by the config
# name so the mapping cannot drift from the setting it describes.
_STREAM_FOR = {
    "graph_api_base": "graph",
    "behavioral_api_base": "behaviour",
    "temporal_api_base": "timing",
}


def _upstream_timeout() -> float:
    """Per-detector call timeout, from config, in seconds."""
    try:
        return max(0.5, float(config.get("upstream", "timeout_ms")) / 1000.0)
    except Exception:                                   # noqa: BLE001
        return 5.0


class MonitorEngine:
    def __init__(self) -> None:
        self.paused = False            # holds the loop without losing counters
        self._fusion = None            # the project's trained MetaClassifier
        self._task: asyncio.Task | None = None
        self._queue: list[dict] = []
        self._source = None          # "queue" | "sample" — published when it changes
        self._label = None           # ground truth from the source file, if present
        self.interval = DEFAULT_INTERVAL
        self.watch_threshold = DEFAULT_WATCH_THRESHOLD
        self._route: dict[str, str] = {}       # detector → the path that answers
        self._down_until: dict[str, float] = {}  # detector → don't call before
        self.upstream_timeout = _upstream_timeout()
        self._background: set[asyncio.Task] = set()  # notifications in flight
        self._last_sweep = 0.0       # when stalled claims were last recovered

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
        runlog.banner(f"monitor started · interval {self.interval}s")
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
        runlog.banner(
            f"monitor stopped · screened {STATE.counters.screened}, "
            f"flagged {STATE.counters.flagged}, alerts {STATE.counters.alerts}")
        STATE.publish("monitor", {"status": "stopped"})

    # ── the loop ─────────────────────────────────────────────────────
    async def _run(self) -> None:
        graph_base = str(config.get("upstream", "graph_api_base")).rstrip("/")
        await self._load_bands(graph_base)
        await self._load_fusion()

        # A previous run may have been killed mid-batch. Those rows are
        # claimed by a worker that no longer exists, so they come back first.
        from monitor import queue as ingest_queue

        recovered = await ingest_queue.release_stale(STALE_CLAIM_SECONDS)
        if recovered:
            logger.info(f"Recovered {recovered} transaction(s) stranded by a previous run.")

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
                # Only the watch threshold is taken from here. The rest of the
                # relational model's bands are deliberately not kept: they
                # describe its own score, and the last time they were held on
                # the engine they ended up being applied to the fused one.
                self.watch_threshold = float(bands.get("medium", DEFAULT_WATCH_THRESHOLD))
                logger.info(f"Monitor watch threshold from model: {self.watch_threshold:.4f}")
        except Exception as exc:                        # noqa: BLE001
            logger.info(f"Using default watch threshold ({exc})")

    async def _next_transaction(self, client: httpx.AsyncClient, graph_base: str):
        """The next arrived transaction, or nothing.

        Only transactions the Query Runner has put into `transactions_live` are
        screened. There is deliberately no other source.

        This used to fall back to the graph service's sample endpoint whenever
        the queue was empty, so a demo always had something moving. That is the
        wrong trade for this system. It meant the live monitor could show a
        throughput, a flag rate and a case queue built entirely from invented
        traffic, labelled "sample replay" in one small caption that nobody
        reads. Worse, it made the two states indistinguishable at a glance:
        screening fifty real arrivals and screening nothing looked identical.

        Empty now means idle, and the monitor says so.
        """
        from monitor import queue as ingest_queue

        if self._queue:
            return self._queue.pop(0)

        # Before concluding the queue is empty, check whether anything is
        # merely stranded. A worker that dies mid-batch leaves its rows
        # claimed, and a claimed row is never screened by anyone — the
        # transaction is silently dropped. release_stale() has always existed
        # for this and was never called; stopping the monitor mid-batch left
        # 23 arrivals stuck that way.
        now = time.monotonic()
        if now - self._last_sweep > STALE_SWEEP_SECONDS:
            self._last_sweep = now
            recovered = await ingest_queue.release_stale(STALE_CLAIM_SECONDS)
            if recovered:
                STATE.publish("source", {
                    "source": "queue",
                    "detail": f"returned {recovered} stalled transaction(s) to the queue",
                })

        claimed = await ingest_queue.claim(POLL_BATCH)
        if claimed:
            self._queue = claimed
            if self._source != "queue":
                self._source = "queue"
                STATE.publish("source", {
                    "source": "queue",
                    "detail": "screening transactions as they arrive",
                })
            return self._queue.pop(0)

        if self._source != "idle":
            self._source = "idle"
            depth = await ingest_queue.depth()
            STATE.publish("source", {
                "source": "idle",
                "detail": ("waiting for transactions — run the Query Runner"
                           if depth.get("available")
                           else "no ingestion queue reachable in this database"),
            })
        return None

    # ── stage 1: screen ──────────────────────────────────────────────
    async def _screen(self, client, graph_base: str, txn: dict) -> None:
        """Score one transaction on all three detectors at once.

        This used to be a cascade: the relational model first, and the other
        two only if it flagged something. The cascade was measured against a
        400-transaction replay and it cost half the frauds in it. Four of the
        eight it missed scored 0.011–0.107 on the relational model — far under
        the 0.183 watch line, so the cascade could never have looked at them —
        while the behavioural model scored those same four between 0.51 and
        1.00. A gate in front of an independent detector cannot do better than
        the detector; it can only hide it.

        It was not buying speed either. The three calls were sequential
        awaits, so a flagged transaction paid graph + timing + behavioural in
        series. Fanned out they cost whichever of the three is slowest, once.

        There is deliberately no "cost gate" option left behind. Any version
        of it either starves the sequence model of the stream its window
        depends on, or skips the behavioural model — which is where those four
        frauds were found. Headroom comes from bounding how many transactions
        are in flight and from the ingestion queue, not from deciding in
        advance which detectors a transaction deserves.
        """
        # row_id marks a claimed queue row; it is bookkeeping, not model input.
        row_id = txn.pop("row_id", None)
        payload = {k: v for k, v in txn.items() if not k.startswith("_")}
        self._label = txn.get("_is_fraud")

        runlog.write("pipeline",
                     f"───── screening  {payload.get('type')} "
                     f"{payload.get('amount')} "
                     f"{payload.get('nameOrig')} → {payload.get('nameDest')}",
                     txn=payload.get("transaction_id"))
        txid = payload["transaction_id"]

        # Measured here rather than estimated later. The case table has always
        # had screening_ms and total_ms columns and the case page has always
        # rendered them; nothing was ever writing them, so every case showed a
        # dash. Timing the calls is the whole fix.
        t0 = time.perf_counter()

        # All three fired together. The relational task is awaited first
        # because the early warning depends only on it, and there is no reason
        # to make that email wait for the other two to come back.
        for name in ("graph", "temporal", "behavioural"):
            STATE.set_stage(name, "active")
        graph_task = asyncio.create_task(self._score_graph(client, graph_base, payload))
        # Held in a dict so the finally below can tell what has not been
        # collected yet, whichever way this method leaves.
        side = {
            "temporal": asyncio.create_task(self._call_upstream(
                client, "temporal_api_base", "temporal_risk_score", payload)),
            "behavioural": asyncio.create_task(self._call_upstream(
                client, "behavioral_api_base", "behavioral_risk_score", payload)),
        }

        try:
            try:
                status, result = await graph_task
            finally:
                STATE.set_stage("graph", "idle")
            screening_ms = int((time.perf_counter() - t0) * 1000)
            STATE.counters.screened += 1

            if status != 200 or result is None:
                # The relational model cannot anchor a subgraph for these
                # accounts, so it abstains — and the transaction goes on to be
                # fused on whatever the other two say.
                #
                # It used to return here, which dropped the transaction
                # entirely: the other two detectors were already running, their
                # scores were collected and thrown away, and nothing was fused
                # or alerted. A behavioural score of 1.00 on an account the
                # graph snapshot has never seen produced no verdict at all.
                # That is the same mistake as the old cascade — one detector
                # deciding whether the others are allowed to matter.
                #
                # Abstaining is the honest reading. "I have no structure for
                # this account" is not "this account is fine", and the
                # uncertainty shrink already exists to price exactly that.
                STATE.publish("screened", {
                    "transaction_id": txid, "outcome": "unknown_accounts",
                })
                runlog.event(("graph", "pipeline"),
                             "graph: ABSTAIN no edge in the snapshot — fusing "
                             "on the remaining detectors", txn=txid)
                temporal = await self._collect(side, "temporal")
                behavioural = await self._collect(side, "behavioural")
                await self._fuse(payload, {}, None,
                                 temporal=temporal, behavioural=behavioural,
                                 watch_flag=False, row_id=row_id,
                                 screening_ms=screening_ms, started=t0)
                return

            score = float(result.get("relational_risk_score") or 0.0)
            level = result.get("risk_level", "LOW")
            sg = result.get("suspicious_subgraph") or {}
            # No longer a gate. It still means something — the relational model
            # saw structure it recognises — and it is still what the early
            # warning goes out on, but nothing downstream is withheld for it.
            watch_flag = score >= self.watch_threshold

            STATE.publish("screened", {
                "transaction_id": txid,
                "amount": payload["amount"],
                "from": payload["nameOrig"],
                "to": payload["nameDest"],
                "graph_score": round(score, 4),
                "risk_level": level,
                "escalated": watch_flag,
            })

            if watch_flag:
                STATE.publish("escalated", {
                    "transaction_id": txid,
                    "graph_score": round(score, 4),
                    "pattern": sg.get("pattern"),
                    "sink_account": sg.get("sink_account"),
                    "convergence": (sg.get("structural_evidence") or {}).get("convergence_count"),
                })
                # Goes out now, while the other two detectors are still in
                # flight — and it is not awaited. An SMTP handshake takes about
                # 3.7 seconds on this connection, and awaiting it here put that
                # into total_ms for every early-flagged transaction, so the case
                # table was reporting mail latency as detection latency. The
                # notification is a consequence of the verdict, not part of it.
                self._spawn(self._notify_early(txid, payload, score, sg), "early warning")

            temporal = await self._collect(side, "temporal")
            behavioural = await self._collect(side, "behavioural")

            await self._fuse(payload, sg, score,
                             temporal=temporal, behavioural=behavioural,
                             watch_flag=watch_flag, row_id=row_id,
                             screening_ms=screening_ms, started=t0)
        finally:
            # Whatever is still in flight — because the relational call raised,
            # or the accounts were unknown — is collected here. Skipping it
            # leaves the detector's lamp lit in the live monitor forever and
            # loses its exception to asyncio's "never retrieved" warning.
            for name in list(side):
                await self._collect(side, name)

    async def _collect(self, side: dict, name: str):
        """Take one detector's answer, once. Idempotent: a second call is a
        no-op, so the cleanup path can ask for everything unconditionally."""
        task = side.pop(name, None)
        if task is None:
            return None, None
        try:
            return await task
        except Exception as exc:                        # noqa: BLE001
            logger.warning(f"{name} detector failed: {exc}")
            return None, None
        finally:
            STATE.set_stage(name, "idle")

    def _spawn(self, coro, what: str) -> None:
        """Run something alongside the pipeline without holding it up.

        The reference is kept until the task finishes: a bare create_task can
        be collected mid-flight, which loses the work silently.
        """
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

        def _log(t: asyncio.Task) -> None:
            if not t.cancelled() and t.exception() is not None:
                logger.warning(f"{what} failed: {t.exception()}")
        task.add_done_callback(_log)

    async def _score_graph(self, client, graph_base: str, payload: dict):
        """The relational model. Returns `(status, body)`, body None on 404.

        Logged here rather than in `_call_upstream`: this detector has its own
        endpoint and its own 404 contract, so it never passes through that
        path — which is why the graph folder stayed empty when only the other
        two were instrumented.
        """
        txid = payload.get("transaction_id")
        started = time.perf_counter()
        try:
            r = await client.post(f"{graph_base}/api/graph/analyze", json=payload)
        except Exception as exc:                        # noqa: BLE001
            runlog.event(("graph", "pipeline"),
                         f"graph: FAIL   unreachable after "
                         f"{(time.perf_counter() - started) * 1000:.0f}ms — "
                         f"{type(exc).__name__}: {str(exc)[:100]}",
                         txn=txid, level="ERROR")
            raise

        ms = (time.perf_counter() - started) * 1000
        if r.status_code == 404:            # accounts not in the trained graph
            runlog.event(("graph", "pipeline"),
                         f"graph: MISS   neither account is in the trained graph "
                         f"({ms:.0f}ms) — abstaining", txn=txid)
            return 404, None
        if r.status_code != 200:
            runlog.event(("graph", "pipeline"),
                         f"graph: FAIL   HTTP {r.status_code} in {ms:.0f}ms",
                         txn=txid, level="ERROR")
        r.raise_for_status()
        body = r.json()
        sg = (body or {}).get("suspicious_subgraph") or {}
        # relational_risk_score, which is what the service actually returns and
        # what _screen reads a few lines below. Guessing graph_risk_score here
        # logged "score=None" for a call that had in fact succeeded.
        score = body.get("relational_risk_score")
        if score is None:
            line = f"graph: OK     200 without relational_risk_score in {ms:.0f}ms"
        else:
            line = (f"graph: OK     score={float(score):.6f} "
                    f"risk={body.get('risk_level') or '—'} "
                    f"pattern={sg.get('pattern') or '—'} "
                    f"sink={sg.get('sink_account') or '—'} in {ms:.0f}ms")
        runlog.event(("graph", "pipeline"), line, txn=txid)
        return r.status_code, body

    # ── stage 2: fuse ────────────────────────────────────────────────
    async def _fuse(self, payload: dict, sg: dict, graph_score: float | None, *,
                    temporal: tuple, behavioural: tuple, watch_flag: bool,
                    row_id: int | None = None,
                    screening_ms: int | None = None,
                    started: float | None = None) -> None:
        txid = payload["transaction_id"]

        # The response bodies are kept, not just the scores. A detector answers
        # with its reasoning attached, and that reasoning is what a reviewer
        # opens the case for — dropping it here is why it never reached the
        # case record.
        scores: dict[str, float | None] = {"graph": graph_score}
        bodies: dict[str, dict | None] = {}
        scores["temporal"], bodies["temporal"] = temporal
        scores["behavioural"], bodies["behavioural"] = behavioural
        for name in ("temporal", "behavioural"):
            if scores[name] is not None:
                STATE.publish("model", {
                    "transaction_id": txid, "model": name, "score": scores[name],
                })

        STATE.set_stage("fusion", "active")
        available = [v for v in scores.values() if v is not None]
        fusion_method = "meta_classifier"
        contributions: dict = {}
        driver = None
        if self._fusion is not None:
            try:
                result = await asyncio.to_thread(
                    self._fusion.fuse,
                    scores.get("graph"), scores.get("behavioural"), scores.get("temporal"),
                )
                # `confidence_score`, not `fraud_confidence_score` — the
                # latter is the API response field, not the dataclass one. Read
                # it directly rather than through getattr with a default: a
                # silent 0.0 on a renamed attribute reads as "nothing
                # suspicious" and suppresses every alert.
                fused = float(result.confidence_score)
                # Which detector actually argued for this verdict. Exact for a
                # linear meta-classifier; empty when averaging, because an
                # average has no such thing as a driver.
                contributions = getattr(result, "contributions", {}) or {}
                driver = getattr(result, "driver", None)
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
        if severity != "LOW":
            STATE.counters.flagged += 1

        runlog.event(("fusion", "pipeline"),
                     f"fusion: VERDICT {severity} fused={fused:.4f} "
                     f"({fusion_method}, {len(available)}/3"
                     + (f", driver={driver}" if driver else "") + ")"
                     + (("  contributions " + " ".join(
                         f"{k}={v:+.2f}" for k, v in contributions.items()))
                        if contributions else ""),
                     txn=txid)

        STATE.publish("fused", {
            "transaction_id": txid,
            "fused_score": round(fused, 4),
            "severity": severity,
            "fusion_method": fusion_method,
            "modalities_used": len(available),
            "scores": {k: (round(v, 4) if v is not None else None) for k, v in scores.items()},
            # The arithmetic, not just the answer. These were already computed
            # for the runlog and then dropped on the floor; carrying them on
            # the event lets a watcher see how each verdict was arrived at
            # while it happens, rather than reconstructing it afterwards.
            "contributions": contributions,
            "driver": driver,
        })

        # The queue row is closed on the fused answer, not the relational one.
        # It is the verdict the system actually stands behind.
        if row_id is not None:
            from monitor import queue as ingest_queue
            await ingest_queue.mark_done(row_id, escalated=severity != "LOW")

        # Everything is fused now, so recording everything would make the case
        # table a log of traffic. A case is opened when the fused verdict is
        # above LOW, or when the relational model raised the transaction even
        # though fusion settled it — the second kind is a disagreement between
        # detectors, which is exactly what a reviewer should see.
        if severity == "LOW" and not watch_flag:
            return

        from backend.adapters.upstream import behavioural_evidence
        from monitor import cases

        # Built through the same function the analyzer's panel is fed from, so
        # a case opened later shows the decomposition in the shape the panel
        # already knows how to render.
        b_body = bodies.get("behavioural")

        case_ref = await cases.record(
            transaction_id=txid,
            classification=severity,
            fused_score=fused,
            scores=scores,
            available_flags={k: (v is not None) for k, v in scores.items()},
            modalities_used=len(available),
            payload=payload,
            graph_evidence=sg,
            behavioral_evidence=behavioural_evidence(b_body) if b_body else None,
            # False, always. This runs before the email is attempted, so any
            # other value here is a claim about something that has not
            # happened yet. _notify_confirmed updates it with what actually
            # occurred — the table previously said "sent" for every non-LOW
            # case, including the eight in a row where the SMTP handshake
            # timed out and nothing was delivered.
            alert_sent=False,
            label_is_fraud=self._label,
            screening_ms=screening_ms,
            total_ms=(int((time.perf_counter() - started) * 1000)
                      if started is not None else None),
        )

        if severity == "LOW":
            return

        alert = {
            "transaction_id": txid,
            "severity": severity,
            "fused_score": round(fused, 4),
            "graph_score": (round(graph_score, 4)
                            if graph_score is not None else None),
            "pattern": sg.get("pattern"),
            "sink_account": sg.get("sink_account"),
            "amount": payload["amount"],
            "from": payload["nameOrig"],
            "to": payload["nameDest"],
            "modalities_used": len(available),
            "fusion_method": fusion_method,
            # What each detector said, and which of them drove the verdict.
            # The alert used to carry the graph score alone, which made every
            # row look like a graph finding even when behaviour was the reason
            # it fired.
            "scores": {k: (round(v, 4) if v is not None else None)
                       for k, v in scores.items()},
            "contributions": contributions,
            "driver": driver,
            "at": time.time(),
        }
        STATE.add_alert(alert)

        # Not awaited. This now generates the forensic narrative and renders the
        # subgraph before it sends, which is tens of seconds of LLM and SMTP —
        # none of it part of the verdict, all of it previously charged to the
        # pipeline. The stage lamp is set inside so the monitor still shows it
        # working.
        self._spawn(self._report_and_notify(alert, sg, case_ref, scores),
                    "confirmed alert")


    async def _call_upstream(self, client, base_key: str, score_key: str, payload: dict):
        """Score one modality.

        Returns `(score, body)`. The body comes back alongside the score
        because the detectors answer with their attribution attached and the
        case record needs it; `(None, None)` when the detector cannot answer.

        A 200 without the score key is a contract violation rather than a
        measurement, so it falls through to the next path and ultimately
        counts as unavailable — the same reading the request adapters take.

        Two things here exist only because of the fan-out. The working route
        is remembered per detector, and a detector that has just failed is not
        asked again for a few seconds. Probing both paths at full timeout used
        to cost a dead service twenty seconds per transaction; that was
        survivable when only the escalated few reached it and is not now that
        every transaction does. The sequence model has been unloadable on this
        machine for days, which is exactly the case this has to absorb.
        """
        stream = _STREAM_FOR.get(base_key, "pipeline")
        txid = payload.get("transaction_id")

        now = time.monotonic()
        if now < self._down_until.get(base_key, 0.0):
            # Worth a line of its own. Silence here reads as "the detector was
            # never asked", when in fact it was skipped on purpose.
            runlog.event((stream, "pipeline"),
                         f"{stream}: SKIP   circuit breaker open for another "
                         f"{self._down_until[base_key] - now:.1f}s", txn=txid)
            return None, None

        base = str(config.get("upstream", base_key)).rstrip("/")
        known = self._route.get(base_key)
        candidates = ("/api/v1/classify", "/api/v1/behavioral/classify")
        paths = ([known] if known else []) + [p for p in candidates if p != known]

        started = time.perf_counter()
        # Every attempt, not just the last one. Reporting only the last was
        # actively misleading: the timing detector answers its real path with a
        # 500 (TensorFlow missing) and the fallback path with a 404, so the log
        # blamed a 404 on a path that detector does not even serve.
        attempts: list[str] = []
        for path in paths:
            try:
                r = await client.post(f"{base}{path}", json=payload,
                                      timeout=self.upstream_timeout)
                if r.status_code == 200:
                    data = r.json()
                    raw = data.get(score_key)
                    if raw is None:
                        attempts.append(f"{path} 200 without '{score_key}'")
                        continue
                    self._route[base_key] = path
                    ms = (time.perf_counter() - started) * 1000
                    runlog.event((stream, "pipeline"),
                                 f"{stream}: OK     {path} → {score_key}={float(raw):.6f} "
                                 f"in {ms:.0f}ms", txn=txid)
                    return float(raw), data
                # 503 is the sequence detector saying its 32-transaction
                # window is still filling — a normal start-up state, and the
                # call that got this answer still advanced that window.
                #
                # Arming the breaker here suppressed the very calls that fill
                # it. The breaker check returns before the POST is sent, so a
                # held-out detector receives nothing and its window stops
                # advancing: at one call per 15s the buffer needed about eight
                # minutes of wall-clock screening rather than 32 transactions,
                # and a replay of any realistic length finished with the
                # detector still warming and every alert showing no timing
                # score. Retry on the next transaction instead.
                #
                # This is not a cosmetic gap. On a transaction where the
                # sequence model carries the verdict, fusion imputes its
                # training mean and the fused score collapses far below the
                # alerting band — the fraud is cleared rather than flagged.
                #
                # A path that answers 503 is a path that exists, so remember it
                # and stop probing the alternates; otherwise every screening
                # also sent a doomed request to the other candidate.
                if r.status_code == 503:
                    self._route[base_key] = path
                    ms = (time.perf_counter() - started) * 1000
                    runlog.event((stream, "pipeline"),
                                 f"{stream}: WARM   {path} still filling its "
                                 f"window ({ms:.0f}ms); abstaining, breaker "
                                 f"left closed", txn=txid)
                    return None, None
                # The body usually names the real cause — a model that failed
                # to load says so here, and that is the line worth keeping.
                detail = ""
                try:
                    detail = f" · {str(r.json())[:110]}"
                except Exception:                       # noqa: BLE001
                    detail = f" · {r.text[:110]}" if r.text else ""
                attempts.append(f"{path} HTTP {r.status_code}{detail}")
            except Exception as exc:                    # noqa: BLE001
                attempts.append(f"{path} {type(exc).__name__}: {str(exc)[:80]}")
                continue

        # Forget the route as well as opening the breaker: a service that came
        # back at a different path should be found again, not written off.
        self._route.pop(base_key, None)
        # Measured from now, not from when the call started. A detector that
        # fails by timing out burns most of the window before it gets here, so
        # dating the breaker from entry would leave it barely closed — in
        # exactly the case the breaker exists for.
        self._down_until[base_key] = time.monotonic() + DETECTOR_BREAKER_SECONDS
        ms = (time.perf_counter() - started) * 1000
        runlog.event((stream, "pipeline"),
                     f"{stream}: FAIL   no answer after {ms:.0f}ms; abstaining, "
                     f"breaker open {DETECTOR_BREAKER_SECONDS:.0f}s"
                     + "".join(f"\n{' ' * 64}tried {a}" for a in attempts),
                     txn=txid, level="ERROR")
        return None, None

    def _severity(self, fused: float) -> str:
        # An operator-set line wins: someone looked at the replay and decided.
        # Otherwise the measured fused bands below — NOT self._bands, which
        # belong to the relational model.
        from backend import thresholds

        b = thresholds.current() or FUSED_BANDS
        if fused >= float(b.get("critical", FUSED_BANDS["critical"])):
            return "CRITICAL"
        if fused >= float(b.get("high", FUSED_BANDS["high"])):
            return "HIGH"
        if fused >= float(b.get("medium", FUSED_BANDS["medium"])):
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

    async def _report_and_notify(self, alert: dict, sg: dict,
                                 case_ref: str | None, scores: dict) -> None:
        STATE.set_stage("report", "active")
        try:
            await self._notify_confirmed(alert, sg, case_ref, scores)
        finally:
            STATE.set_stage("report", "idle")

    async def _forensic_report(self, alert: dict, scores: dict) -> str | None:
        """Generate the case narrative, or None if it cannot be produced.

        Built from what the pipeline already has rather than by re-running it:
        calling run_pipeline here would score the transaction a second time and
        push it into the sequence model's window twice, which would corrupt the
        very window that model depends on.

        Best-effort by design. It is one LLM call on a shared quota, and an
        alert that arrives without its narrative is far better than one that
        does not arrive.
        """
        txid = alert["transaction_id"]
        try:
            from backend import main as backend_main
            from backend.rag.prompt_builder import build_chain_of_evidence_prompt

            reporter, retriever = backend_main.forensic_reporter, backend_main.retriever
            if reporter is None or retriever is None:
                runlog.write("fusion",
                             "SKIP   forensic report — reporter or retriever not "
                             "initialised", txn=txid)
                return None

            g = scores.get("graph") or 0.0
            b = scores.get("behavioural")
            t = scores.get("temporal")
            with runlog.stage("fusion", "rag_retrieval", txn=txid) as d:
                hits = await asyncio.wait_for(
                    asyncio.to_thread(retriever.retrieve, g, b or 0.0, t or 0.0,
                                      float(alert["fused_score"])),
                    timeout=20.0,
                )
                # RetrievalResult is a dataclass, not a dict. Reading it with
                # .get() raised inside the try that wraps report generation,
                # so a logging line silently cost every alert its narrative.
                top = hits[0] if hits else None
                d["note"] = (
                    f'matched "{getattr(top, "typology_name", "?")}" '
                    f'(similarity {getattr(top, "similarity_score", 0.0):.3f})'
                    if top is not None else "no typology matched"
                )
            if not hits:
                return None

            package = build_chain_of_evidence_prompt(
                transaction_id=alert["transaction_id"],
                graph_score=g,
                behavioral_score=b if b is not None else 0.0,
                temporal_score=t if t is not None else 0.0,
                confidence_score=float(alert["fused_score"]),
                graph_available=scores.get("graph") is not None,
                behavioral_available=b is not None,
                temporal_available=t is not None,
                retrieval=hits[0],
                classification=alert["severity"],
            )
            with runlog.stage("fusion", "llm_narrative", txn=txid) as d:
                report = await asyncio.wait_for(
                    asyncio.to_thread(reporter.generate_report, package), timeout=45.0
                )
                d["note"] = f"{len(report or '')} chars"
            return report
        except Exception as exc:                        # noqa: BLE001
            logger.info(f"No forensic report for {alert['transaction_id']}: {exc}")
            runlog.write("fusion",
                         f"FAIL   forensic report — {type(exc).__name__}: {str(exc)[:140]}",
                         txn=txid, level="ERROR")
            return None

    async def _notify_confirmed(self, alert: dict, sg: dict,
                                case_ref: str | None = None,
                                scores: dict | None = None) -> None:
        from backend.email_service import _send_rich
        from monitor import alert_email
        from monitor import assets as alert_assets
        from monitor.alert_render import render_subgraph

        scores = scores or {"graph": alert.get("graph_score")}

        # The narrative and the diagram are produced together, before the send,
        # because both belong in the same message.
        txid = alert["transaction_id"]
        report = await self._forensic_report(alert, scores)
        try:
            with runlog.stage("fusion", "subgraph_diagram", txn=txid) as d:
                png = await asyncio.to_thread(render_subgraph, sg)
                d["note"] = f"{len(png) / 1024:.1f} KB" if png else "no diagram produced"
        except Exception as exc:                        # noqa: BLE001
            logger.info(f"Subgraph diagram failed: {exc}")
            png = None

        attachments = []
        if report:
            try:
                with runlog.stage("fusion", "pdf_generation", txn=txid) as d:
                    pdf = _report_pdf(alert, report)
                    d["note"] = f"{len(pdf) / 1024:.1f} KB"
                attachments.append((
                    "application", "pdf",
                    f"forensic-report-{txid}.pdf",
                    pdf,
                ))
            except Exception as exc:                    # noqa: BLE001
                # An alert with its narrative missing still has to go out.
                logger.warning(f"Forensic PDF failed for {txid}: {exc}")
                report = None

        from backend import thresholds

        html = alert_email.build(
            alert=alert, sg=sg, scores=scores,
            bands=thresholds.current() or FUSED_BANDS,
            has_image=png is not None, case_ref=case_ref,
            console_url=str(config.get("upstream", "console_url")).rstrip("/"),
            report_attached=bool(report),
        )
        text = alert_email.build_text(alert, sg, scores, bool(report))

        from backend.settings import list_risk_managers

        managers = await list_risk_managers()
        recipients = [m.email for m in managers if getattr(m, "enabled", True)]
        if not recipients:
            logger.info("No alert recipients configured; nothing sent.")
            runlog.write("fusion",
                         "SKIP   email — no risk managers configured", txn=txid)
            sent = False
        else:
            with runlog.stage("fusion", "email_delivery", txn=txid,
                              also=("pipeline",)) as d:
                sent = await asyncio.to_thread(
                    _send_rich,
                    f"[{alert['severity']}] Fraud alert {txid}",
                    text, html, recipients,
                    # The severity banner and the mark travel with the
                    # subgraph as CID parts. Built from what is on disk, so a
                    # missing file drops that one image and nothing else.
                    {**alert_assets.inline_for(alert["severity"]),
                     **({"subgraph": png} if png else {})} or None,
                    attachments or None,
                )
                # Whether it was accepted, not whether it was attempted. The
                # case table used to claim "sent" for every non-LOW verdict,
                # including eight in a row where the handshake timed out.
                d["note"] = (
                    f"{'delivered to' if sent else 'REJECTED for'} "
                    f"{len(recipients)} recipient(s)"
                    + (", pdf attached" if attachments else "")
                    + (", diagram inline" if png else "")
                )

        # Record what happened, not what was intended. An operator reading a
        # case needs to know whether anyone was actually told.
        if case_ref:
            from monitor import cases as _cases

            await _cases.mark_alerted(case_ref, sent)

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


SEV_RGB = {
    "CRITICAL": (0.69, 0.22, 0.17),
    "HIGH":     (0.65, 0.42, 0.03),
    "MEDIUM":   (0.33, 0.28, 0.60),
    "LOW":      (0.08, 0.48, 0.24),
}


def _report_sections(report: str) -> list[tuple[str | None, str]]:
    """Split the narrative into (heading, body) pairs.

    The reporter returns its sections with the heading run together with the
    first sentence — "SECTION 2 - MULTI-MODAL EVIDENCE ANALYSIS The Graph
    Network Analysis Score is..." — and wraps the whole in --- fences. Left
    alone that reads as a wall of text. The headings are lifted out here rather
    than by asking the model to format itself, which it does not do reliably.
    """
    import re as _re

    heading = _re.compile(
        r"^(SECTION\s+\d+\s*[\u2014\u2013-]\s*"
        r"(?:[A-Z][A-Z\-/&.]*(?:[ \t]+|$)){1,6})"
    )
    preamble = _re.compile(r"^(Transaction ID|Classification):", _re.M)

    out: list[tuple[str | None, str]] = []
    for para in _re.split(r"\n\s*\n", report.replace("---", "").strip()):
        para = " ".join(para.split())
        if not para:
            continue
        head = None
        m = heading.match(para)
        if m:
            head = " ".join(m.group(1).split()).title()
            para = para[m.end():].strip()
        if preamble.search(para) or para.upper().startswith("CASE INVESTIGATION REPORT"):
            # This block only repeats the facts already tabulated above it.
            keep = _re.search(r"(FATF Typology Match:.*)$", para)
            para = keep.group(1).strip() if keep else ""
        if head or para:
            out.append((head, para))
    return out


def _report_pdf(alert: dict, report: str, style: str | None = None) -> bytes:
    """The forensic report as a filed document.

    A PDF rather than the HTML page this used to attach: what a compliance
    officer does with this is save it, print it and cite it, and an .html
    attachment is none of those things. Built with the in-tree writer, so
    nothing has to be installed to produce one.

    Laid out from the design study — warm ground, a dark masthead, the fused
    score as the one large number, a segmented scale, and numbered sections. It
    is the study's palette and structure rather than its typography: the writer
    uses PDF's standard fonts so that nothing has to be embedded, and every
    reader in the world can open it.
    """
    from backend import report_styles
    from backend.pdf import Document

    st = report_styles.resolve(style)

    class P:                       # the palette for this render
        GROUND, INK, MUTED = st["ground"], st["ink"], st["muted"]
        FAINT, RULE, WASH = st["faint"], st["rule"], st["wash"]
        DEEP = st["header"]

    sev = str(alert.get("severity") or "LOW").upper()
    accent = SEV_RGB.get(sev, (0.37, 0.41, 0.42))
    tint = {"CRITICAL": (1.000, 0.960, 0.952), "HIGH": (0.996, 0.965, 0.918),
            "MEDIUM": (0.945, 0.941, 0.976), "LOW": (0.925, 0.957, 0.937)}.get(
                sev, (0.949, 0.953, 0.953))

    doc = Document(
        footer="DeepSentinel \u00b7 generated from the record for this transaction",
        ground=st["ground"],
        rail=st.get("rail", 0.0), rail_rgb=st.get("header"))
    txid = str(alert["transaction_id"])
    fused = float(alert["fused_score"])
    used = int(alert.get("modalities_used") or 0)

    # A style may pin its own accent. Severity still shows in the words and the
    # band; letting an amber HIGH tint a deep-green report just makes it look
    # like two designs stapled together.
    if st.get("accent"):
        accent = st["accent"]

    # ── the rail, when the style has one ─────────────────────────────────
    #
    # Everything here is what a masthead and a hero would otherwise carry: who
    # produced this, how sure it is, and which accounts it concerns. Putting
    # them down the side leaves the whole width to the narrative, which is the
    # part anyone actually reads.
    if st.get("rail"):
        RI = st.get("rail_ink", (0.95, 0.96, 0.96))
        RM = st.get("rail_muted", (0.61, 0.70, 0.65))
        RT = st.get("rail_track", (0.18, 0.29, 0.23))

        doc.rail_label("Chain-of-evidence", RM, size=7.0, gap=2.0)
        doc.rail_label("Forensic report", RM, size=7.0, gap=1.0)
        doc.ry -= 10
        doc.rail_label(f"{sev} priority", RI, size=7.5, gap=4.0)

        doc.ry -= 12
        doc.rail_label("Fused fraud confidence", RM)
        doc.rail_hero(f"{fused * 100:.1f}", "%", RI, accent, RT, fused)
        doc.ry -= 6
        doc.rail_value(f"{used} of 3 detectors", RM, size=7.5)

        doc.rail_rule(RT)
        doc.rail_label("Transaction", RM)
        doc.rail_value(txid, RI, size=8.0)

        if alert.get("pattern"):
            doc.rail_rule(RT)
            doc.rail_label("Type", RM)
            doc.rail_value(str(alert["pattern"]).replace(" ", "_").upper(), RI, size=8.5)

        if alert.get("amount") is not None:
            doc.rail_rule(RT)
            doc.rail_label("Amount", RM)
            doc.rail_value(f"{float(alert['amount']):,.2f}", RI,
                           size=13.0, font="Helvetica-Bold")

        if alert.get("from") or alert.get("to"):
            doc.rail_rule(RT)
            doc.rail_label("From \u2192 to", RM)
            doc.rail_value(str(alert.get("from") or "\u2014"), RI, size=8.0)
            doc.rail_value("\u2193", RM, size=8.0)
            doc.rail_value(str(alert.get("to") or "\u2014"), RI, size=8.0)

        doc.heading("Suspicious transaction", size=20.0)
        doc.rule(rgb=P.RULE)

    if st["masthead"]:
        doc.masthead("Chain-of-evidence forensic report",
                     "Suspicious transaction", txid, accent, deep=st["header"])
    elif not st.get("rail"):
        doc.band(accent)
        doc.label(f"{sev}  \u00b7  chain-of-evidence forensic report", accent)
        doc.heading(f"Transaction {txid}")

    # Where this verdict sits on the operating range, in twelve blocks. The
    # bands are far apart in probability, so the fill is by band index and
    # position within it — a linear scale would put nearly every transaction
    # in the first block.
    from backend import thresholds

    b = thresholds.current() or FUSED_BANDS
    stops = [0.0, float(b.get("medium", 0.03)), float(b.get("high", 0.09)),
             float(b.get("critical", 0.925)), 1.0]
    seg = max(i for i in range(4) if fused >= stops[i])
    lo, hi = stops[seg], stops[seg + 1]
    within = (fused - lo) / (hi - lo) if hi > lo else 0.0
    lit = max(1, min(12, round(((seg + within) / 4) * 12)))

    if st["hero"]:
        doc.hero(f"{fused * 100:.1f}", "%", "Fused fraud confidence", accent,
                 segments=12, lit=lit, muted=P.MUTED, ink=P.INK, wash=P.WASH)
        doc.pill(f"{sev}  \u00b7  {used} of 3 detectors", accent, tint)
    elif not st.get("rail"):
        doc.para(
            f"Fused confidence {fused:.4f}  \u00b7  {used} of 3 detectors available",
            size=9.5, font="Courier", rgb=P.MUTED)

    # With a rail, these already appear down the side.
    if not st.get("rail"):
        doc.rule(rgb=P.RULE)
        doc.label("Transaction", P.MUTED)
        doc.kv([
            ("Amount", f"{alert['amount']:,.2f}"),
            ("Originating account", str(alert.get("from") or "\u2014")),
            ("Collection account", str(alert.get("to") or "\u2014")),
            ("Pattern", str(alert.get("pattern") or "\u2014").replace("_", " ").title()),
            ("Sink", str(alert.get("sink_account") or "\u2014")),
        ])

    scores = alert.get("scores") or {}
    if scores:
        doc.rule(rgb=P.RULE)
        doc.label("Sub-model risk scores", P.MUTED)
        rows = [(key, name) for key, name in
                (("graph", "Graph network analysis"),
                 ("behavioural", "Behavioural anomaly"),
                 ("temporal", "Temporal pattern analysis"))
                if key in scores]
        if st.get("meters"):
            # Same three numbers, drawn instead of listed. The dominant
            # detector is named on its own row rather than in a sentence
            # underneath, so the loudest signal is visible before it is read.
            driver = alert.get("driver")
            for key, name in rows:
                v = scores.get(key)
                if v is None:
                    doc.kv([(name, "did not answer")])
                    continue
                note = ("dominant \u00b7 signal rating high" if key == driver
                        else "rating high" if v >= 0.5
                        else "rating medium" if v >= 0.18
                        else "minimal \u00b7 rating low")
                doc.meter(name, float(v), note,
                          accent=st.get("accent", accent),
                          track=P.WASH, ink=P.INK, muted=P.MUTED)
        else:
            doc.kv([
                (name, f"{scores[key]:.4f}" if scores.get(key) is not None
                       else "did not answer")
                for key, name in rows
            ])
        if alert.get("driver"):
            doc.para(
                f"Largest contribution to the fused verdict: "
                f"{alert['driver']}. The meta-classifier is linear, so this is "
                f"the exact share of the log-odds, not an estimate.",
                size=8.5, rgb=P.MUTED)

    # The counter advances on headings only. Enumerating every pair numbered
    # the untitled paragraphs too, so a report with any preamble came out
    # numbered 01, 03, 05.
    n = 0
    for head, body in _report_sections(report):
        if head:
            n += 1
            if st["numbered"]:
                doc.numbered(n, head, muted=P.MUTED, rule=P.RULE, ink=P.INK)
            else:
                doc.subheading(head)
        if body:
            doc.para(body, rgb=P.INK)

    doc.rule(gap=14.0, rgb=P.RULE)
    doc.para(
        "Generated by DeepSentinel from the scores and the retrieved typology on record "
        "for this transaction. Every claim above traces to one of them; nothing in it is "
        "inferred beyond what was measured. Review before acting.",
        size=8.5, rgb=P.FAINT, keep_together=True,
    )
    return doc.render()


ENGINE = MonitorEngine()

"""Live state for the monitor.

One process-wide store holding what is happening right now: counters, the
recent event log, and open alerts. Subscribers (the dashboard over SSE, and the
operator assistant) read from here, so everyone sees the same truth rather than
each polling a different endpoint and disagreeing.

Bounded on purpose — a monitor that runs for days must not grow without limit,
so the event log and alert list are ring buffers.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

MAX_EVENTS = 200
MAX_ALERTS = 50


@dataclass
class Counters:
    screened: int = 0            # transactions the graph model has seen
    escalated: int = 0           # sent on to the other two detectors
    alerts: int = 0              # fused verdicts at or above MEDIUM
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        elapsed = max(time.time() - self.started_at, 1e-6)
        return {
            "screened": self.screened,
            "escalated": self.escalated,
            "alerts": self.alerts,
            "uptime_seconds": round(elapsed, 1),
            "throughput_per_min": round(self.screened / elapsed * 60, 1),
            # The screening funnel is the story: a lot in, few escalated,
            # fewer still alerted.
            "escalation_rate": round(self.escalated / self.screened, 4) if self.screened else 0.0,
        }


class MonitorState:
    def __init__(self) -> None:
        self.running = False
        self.counters = Counters()
        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self.alerts: deque[dict] = deque(maxlen=MAX_ALERTS)
        self.stage_status: dict[str, str] = {
            "graph": "idle", "behavioural": "idle",
            "temporal": "idle", "fusion": "idle", "report": "idle",
        }
        self._subscribers: set[asyncio.Queue] = set()

    # -- pub/sub ------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, kind: str, payload: dict) -> None:
        """Fan out an event. A slow consumer is dropped from, not blocked on."""
        event = {"kind": kind, "at": time.time(), **payload}
        if kind != "heartbeat":
            self.events.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # The dashboard fell behind; skip rather than stall the engine.
                pass

    # -- mutations ----------------------------------------------------
    def set_stage(self, stage: str, status: str) -> None:
        self.stage_status[stage] = status
        self.publish("stage", {"stage": stage, "status": status})

    def add_alert(self, alert: dict) -> None:
        self.alerts.appendleft(alert)
        self.counters.alerts += 1
        self.publish("alert", alert)

    def snapshot(self, events: int = 40) -> dict:
        return {
            "running": self.running,
            "counters": self.counters.as_dict(),
            "stages": dict(self.stage_status),
            "alerts": list(self.alerts)[:20],
            "events": list(self.events)[-events:],
        }


STATE = MonitorState()

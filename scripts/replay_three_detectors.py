"""Replay the held-out test window through all three detectors.

Produces the score triples the meta-classifier should be fitted on, which
until now did not exist: every previous fit used synthetic triples, and every
previous measurement ran with the timing detector unavailable.

Order matters. The timing model holds a rolling 32-transaction window and
scores the stream, so rows are sent in chronological order and its first 31
answers are warm-up. Shuffling here would measure a different model.

Writes CSV: transaction_id, step, graph, behavioural, temporal, is_fraud
with empty cells where a detector abstained.
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request

GRAPH = "http://127.0.0.1:8002/api/graph/analyze"
BEHAV = "http://127.0.0.1:8001/api/v1/behavioral/classify"
TEMPO = "http://127.0.0.1:8003/api/v1/classify"

SRC = sys.argv[1]
OUT = sys.argv[2]


def post(url: str, payload: dict, timeout: float = 30.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"__status": e.code}
    except Exception as e:                               # noqa: BLE001
        return {"__err": type(e).__name__}


def main() -> int:
    rows = list(csv.DictReader(open(SRC, newline="")))
    # NO sorting. The input is already in original PaySim order, and that is
    # the point: the timing model's window IS the sequence, so re-ordering
    # within a step — even by something as innocent as account name — makes it
    # score a stream that never existed.
    print(f"  {len(rows)} rows, steps "
          f"{int(float(rows[0]['step']))}–{int(float(rows[-1]['step']))}")

    out = open(OUT, "w", newline="")
    w = csv.writer(out)
    w.writerow(["transaction_id", "step", "graph", "behavioural", "temporal",
                "is_fraud"])

    counts = {"graph": 0, "behavioural": 0, "temporal": 0}
    started = time.time()

    for i, r in enumerate(rows):
        txid = f"replay-{i:06d}"
        p = {
            "transaction_id": txid, "step": int(float(r["step"])),
            "type": r["type"], "amount": float(r["amount"]),
            "nameOrig": r["nameOrig"], "nameDest": r["nameDest"],
            "oldbalanceOrg": float(r["oldbalanceOrg"]),
            "newbalanceOrig": float(r["newbalanceOrig"]),
            "oldbalanceDest": float(r["oldbalanceDest"]),
            "newbalanceDest": float(r["newbalanceDest"]),
            "isFlaggedFraud": 0,
        }

        g = post(GRAPH, p)
        gs = g.get("relational_risk_score") if "__status" not in g else None

        b = post(BEHAV, p)
        bs = b.get("behavioral_risk_score") if isinstance(b, dict) else None

        t = post(TEMPO, p)
        ts = t.get("temporal_risk_score") if isinstance(t, dict) else None

        for k, v in (("graph", gs), ("behavioural", bs), ("temporal", ts)):
            if v is not None:
                counts[k] += 1

        w.writerow([txid, p["step"],
                    "" if gs is None else f"{float(gs):.6f}",
                    "" if bs is None else f"{float(bs):.6f}",
                    "" if ts is None else f"{float(ts):.6f}",
                    1 if r.get("isFraud") in ("1", "1.0") else 0])

        if i and i % 200 == 0:
            rate = i / (time.time() - started)
            print(f"    {i}/{len(rows)}  {rate:.0f}/s  "
                  f"g={counts['graph']} b={counts['behavioural']} "
                  f"t={counts['temporal']}", flush=True)

    out.close()
    el = time.time() - started
    print()
    print(f"  wrote {OUT} in {el:.0f}s ({len(rows)/el:.0f}/s)")
    for k, v in counts.items():
        print(f"    {k:<12} answered {v}/{len(rows)} ({v/len(rows):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

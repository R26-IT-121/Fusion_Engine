"""Replay ordinary traffic through all three detectors and fuse every one.

The bands were set when fusion only ever saw escalated transactions. Under a
parallel fan-out it sees the whole stream, so the question this answers is:
what does the meta-classifier output on traffic that used to be cleared, and
where does the medium band have to sit so the case table does not flood.
"""
import json, statistics, sys, time
from pathlib import Path
import httpx

ROOT = Path("/Users/sachinthabhashitha/Downloads/Fusion Engine/R26-IT-121/fusion_engine/DeepSentinel")
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

from backend.fusion_engine import MetaClassifier

N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
GRAPH, BEHAV, TEMPORAL = "http://127.0.0.1:8002", "http://127.0.0.1:8001", "http://127.0.0.1:8003"

clf = MetaClassifier("./models/meta_classifier.joblib")
clf.initialize()

def score(c, base, keys):
    for path in ("/api/v1/classify", "/api/v1/behavioral/classify"):
        try:
            r = c.post(f"{base}{path}", json=P, timeout=6.0)
            if r.status_code == 200:
                d = r.json()
                for k in keys:
                    if d.get(k) is not None:
                        return float(d[k])
        except Exception:
            continue
    return None

rows, seen = [], set()
with httpx.Client() as c:
    while len(rows) < N:
        r = c.get(f"{GRAPH}/api/graph/sample-transactions", params={"count": 50}, timeout=20.0)
        batch = r.json().get("transactions", [])
        if not batch:
            break
        for P in batch:
            if P["transaction_id"] in seen or len(rows) >= N:
                continue
            seen.add(P["transaction_id"])
            label = bool(P.pop("_is_fraud", False))
            try:
                g = c.post(f"{GRAPH}/api/graph/analyze", json=P, timeout=20.0)
                if g.status_code != 200:
                    continue
                gs = float(g.json().get("relational_risk_score") or 0.0)
            except Exception:
                continue
            bs = score(c, BEHAV, ("behavioral_risk_score",))
            ts = score(c, TEMPORAL, ("temporal_risk_score",))
            fused = clf.fuse(gs, bs, ts).confidence_score
            rows.append({"g": gs, "b": bs, "t": ts, "f": fused, "y": label, "step": int(P["step"])})
        print(f"  … {len(rows)}", file=sys.stderr, flush=True)

out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("replay.json")
out.write_text(json.dumps(rows))
print(f"\n{len(rows)} transactions, {sum(r['y'] for r in rows)} fraudulent")
print(f"modalities answering: graph {sum(r['g'] is not None for r in rows)}, "
      f"behavioural {sum(r['b'] is not None for r in rows)}, "
      f"temporal {sum(r['t'] is not None for r in rows)}")

WATCH = 0.1830238699913025
old_gate = [r for r in rows if r["g"] >= WATCH]
print(f"\nunder the OLD cascade: {len(old_gate)}/{len(rows)} "
      f"({100*len(old_gate)/len(rows):.1f}%) reached fusion at all")

fs = sorted(r["f"] for r in rows)
def q(p): return fs[min(int(p * len(fs)), len(fs) - 1)]
print(f"\nfused score over the WHOLE stream:")
print(f"  min {fs[0]:.4f}  p50 {q(.5):.4f}  p90 {q(.9):.4f}  p99 {q(.99):.4f}  max {fs[-1]:.4f}")
print(f"  mean {statistics.mean(fs):.4f}")

print(f"\nwhat each band would flag (cases opened per {len(rows)} screened):")
print(f"  {'band':>7}  {'flagged':>7}  {'rate':>7}  {'frauds caught':>13}  {'precision':>9}")
frauds = sum(r["y"] for r in rows) or 1
for band in (0.09, 0.15, 0.20, 0.25, 0.30, 0.39, 0.50, 0.60, 0.70, 0.80):
    hit = [r for r in rows if r["f"] >= band]
    tp = sum(r["y"] for r in hit)
    prec = tp / len(hit) if hit else 0.0
    print(f"  {band:>7.2f}  {len(hit):>7}  {100*len(hit)/len(rows):>6.1f}%  "
          f"{tp:>6}/{frauds:<6}  {prec:>8.1%}")

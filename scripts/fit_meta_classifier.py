"""Fit the meta-classifier on real detector output and see whether it is better.

The shipped model was fitted on synthetic score triples drawn from beta
distributions — a stand-in written before the detectors existed. This asks
whether a model fitted on what the detectors actually emit beats it, on a
held-out slice, and refuses to adopt it if it does not.

The split is by PaySim `step`, not at random. The inputs here are already model
outputs, so the leakage risk is milder than it is upstream, but the project's
whole evaluation protocol is temporal and a random split here would be the one
inconsistent measurement in it.
"""
import json, sys, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = "/Users/sachinthabhashitha/Downloads/Fusion Engine/R26-IT-121/fusion_engine/DeepSentinel"
os.chdir(ROOT); sys.path.insert(0, ROOT)
from backend.fusion_engine import MetaClassifier

rows = json.load(open(sys.argv[1]))
rows = [r for r in rows if r["g"] is not None and r["b"] is not None]
steps = np.array([r["step"] for r in rows])
cut = int(np.quantile(steps, 0.70))
tr = [r for r in rows if r["step"] <= cut]
te = [r for r in rows if r["step"] > cut]

Xtr = np.array([[r["g"], r["b"]] for r in tr]); ytr = np.array([int(r["y"]) for r in tr])
Xte = np.array([[r["g"], r["b"]] for r in te]); yte = np.array([int(r["y"]) for r in te])

print(f"{len(rows)} scored transactions · split at step {cut}")
print(f"  train {len(tr):>5}  ({ytr.sum()} fraud, {ytr.mean():.2%})")
print(f"  test  {len(te):>5}  ({yte.sum()} fraud, {yte.mean():.2%})")
if ytr.sum() < 10 or yte.sum() < 5:
    print("\nNot enough fraud either side of the split to fit or judge anything.")
    sys.exit(2)

# Incumbent: the shipped synthetic model, scored on the same held-out slice.
inc = MetaClassifier('./models/meta_classifier.joblib'); inc.initialize()
p_inc = np.array([inc.fuse(r["g"], r["b"], None).confidence_score for r in te])

# Challenger: fitted on real output from the two detectors that answer here.
chal = Pipeline([("scaler", StandardScaler()),
                 ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                            class_weight="balanced", random_state=42))])
chal.fit(Xtr, ytr)
p_chal = chal.predict_proba(Xte)[:, 1]

def report(name, p):
    ap, auc = average_precision_score(yte, p), roc_auc_score(yte, p)
    print(f"  {name:<34} PR-AUC {ap:.4f}   AUC-ROC {auc:.4f}   distinct {len(set(np.round(p,4))):>4}")
    return ap

print("\nheld-out performance:")
ap_i = report("shipped (synthetic, 3-feature)", p_inc)
ap_c = report("fitted on real output (2-feature)", p_chal)

base = yte.mean()
print(f"\n  base rate on the test slice: {base:.4f}  (PR-AUC of a coin flip)")
print(f"  lift — shipped {ap_i/base:.2f}x · challenger {ap_c/base:.2f}x")

verdict = "ADOPT" if ap_c > ap_i * 1.02 else "KEEP THE SHIPPED MODEL"
print(f"\n  => {verdict}")
if verdict.startswith("ADOPT"):
    import joblib
    joblib.dump({"pipeline": chal, "features": ["graph", "behavioural"],
                 "trained_on": len(tr), "test_pr_auc": float(ap_c),
                 "split_step": int(cut)}, "./models/meta_gb.joblib")
    print("     written to models/meta_gb.joblib")

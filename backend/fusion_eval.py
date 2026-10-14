"""How the fused verdict scores against each detector on its own.

Computed from the replayed test window rather than quoted: the triples file
holds every detector's score for the held-out steps plus the ground truth, so
the comparison that justifies fusing at all can be recomputed whenever the
meta-classifier changes instead of ageing in a slide.

Two metrics, because they disagree and the disagreement is the honest part.
At 9.4% positives, average precision is the metric that reflects what an
analyst experiences; AUROC flatters any model on an imbalanced set. Both are
reported so neither can be cherry-picked.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE: dict | None = None
TRIPLES = Path(__file__).resolve().parent.parent / "data" / "datasets" / \
    "replay_triples_test_window.csv"


def _num(v):
    try:
        return float(v) if v not in ("", None) else None
    except (TypeError, ValueError):
        return None


def evaluate(meta_classifier, bands: dict) -> dict | None:
    """Recompute the comparison. Cached — it fuses five thousand rows."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if meta_classifier is None or not TRIPLES.exists():
        return None

    try:
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return None

    try:
        rows = list(csv.DictReader(TRIPLES.open()))
        # A meta-classifier fitted on the replay's earlier steps is judged only
        # on the later ones; scoring it on its own training rows would flatter it.
        pipe = getattr(meta_classifier, "_pipeline", None)
        fit_until = getattr(pipe, "ds_fit_until_step", None) if getattr(pipe, "ds_source", None) == "recorded" else None
        if fit_until is not None:
            rows = [r for r in rows if int(r["step"]) > fit_until]
        y = np.array([int(r["is_fraud"]) for r in rows], dtype=bool)
        fused = np.array([
            meta_classifier.fuse(_num(r["graph"]), _num(r["behavioural"]),
                                 _num(r["temporal"])).confidence_score
            for r in rows
        ])

        def pair(scores):
            ok = ~np.isnan(scores)
            if not ok.any() or len(set(y[ok].tolist())) < 2:
                return None
            return {"pr_auc": round(float(average_precision_score(y[ok], scores[ok])), 4),
                    "auroc": round(float(roc_auc_score(y[ok], scores[ok])), 4),
                    "scored": int(ok.sum())}

        detectors = {}
        for key, label in (("graph", "Network"), ("behavioural", "Behaviour"),
                           ("temporal", "Timing")):
            col = np.array([_num(r[key]) if _num(r[key]) is not None else np.nan
                            for r in rows])
            got = pair(col)
            if got:
                detectors[label] = got

        # At the line the monitor actually alerts on — read from the one
        # definition rather than defaulted to here, which is how a fourth copy
        # of this number got into the codebase in the first place.
        from backend.thresholds import DEFAULT_BANDS

        thr = float((bands or {}).get("critical") or DEFAULT_BANDS["critical"])
        flagged = fused >= thr
        tp = int((flagged & y).sum())
        fp = int((flagged & ~y).sum())
        fn = int((~flagged & y).sum())
        tn = int((~flagged & ~y).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        _CACHE = {
            "window": {"name": "held-out test" if fit_until is None else f"held-out test, steps {fit_until + 1}-743 (the fusion was fitted on earlier steps)",
                       "rows": len(rows), "frauds": int(y.sum())},
            "fusion": pair(fused),
            "detectors": detectors,
            "at_threshold": {
                "threshold": round(thr, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4)
                if precision + recall else 0.0,
                "accuracy": round((tp + tn) / max(1, len(y)), 4),
                "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            },
            "note": ("Replayed test window. Average precision is the metric to "
                     "read at this class balance; AUROC is shown beside it "
                     "because the two rank the detectors differently."),
        }
        return _CACHE
    except Exception as exc:                            # noqa: BLE001
        logger.warning(f"Could not evaluate the fusion engine: {exc}")
        return None

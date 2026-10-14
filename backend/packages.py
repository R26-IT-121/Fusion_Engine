"""Which package a deployment is licensed for, and what that unlocks.

Generalises the single-feature gate in `assistant/entitlement.py`. That module
proved the shape — a master switch plus a role list, persisted in the runtime
settings JSON so it survives a restart without a migration — and this applies
the same shape to every commercial feature.

The hard rule this module encodes
---------------------------------
Detection is never gated. Screening, fusion, alerting and the monitor run
identically on every package, because a fraud detector that stops detecting
when a licence lapses is not a fraud detector. What tiers is how much the
product helps a customer *understand, act on and govern* a result — never
whether the result is produced.

`ALWAYS_INCLUDED` names those capabilities explicitly so the rule is checkable
rather than a convention someone erodes later; `require()` refuses to gate them
even if a caller asks.
"""

from __future__ import annotations

import json
import logging
import threading
from enum import Enum
from pathlib import Path

from fastapi import HTTPException

from backend import config

logger = logging.getLogger(__name__)

SETTINGS_KEY = "package"
_LOCK = threading.Lock()


class Package(str, Enum):
    ESSENTIAL = "essential"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


# Ordered weakest to strongest. Membership is cumulative: Enterprise includes
# everything Professional unlocks.
_RANK = {Package.ESSENTIAL: 0, Package.PROFESSIONAL: 1, Package.ENTERPRISE: 2}

DEFAULT_PACKAGE = Package.PROFESSIONAL

# Capabilities that no package may withhold. Listed by name so the guarantee is
# testable, not merely documented.
ALWAYS_INCLUDED = frozenset({
    "detection",        # all three modalities, every transaction
    "fusion",           # combining them into one confidence
    "alerting",         # email on a confirmed alert
    "monitoring",       # the always-on screening loop
    "analysis_history", # the record that a transaction was screened
})

# Everything else, and the minimum package that unlocks it.
FEATURES: dict[str, Package] = {
    "attribution_panels": Package.PROFESSIONAL,
    "forensic_report":    Package.PROFESSIONAL,
    "ablation_view":      Package.PROFESSIONAL,
    "ai_assistant":       Package.PROFESSIONAL,
    "batch_analysis":     Package.PROFESSIONAL,
    "sar_draft":          Package.PROFESSIONAL,
    "threshold_sim":      Package.PROFESSIONAL,
    "governance_pack":    Package.ENTERPRISE,
    "drift_monitoring":   Package.ENTERPRISE,
    "custom_training":    Package.ENTERPRISE,
}

_LABELS = {
    Package.ESSENTIAL: "Essential",
    Package.PROFESSIONAL: "Professional",
    Package.ENTERPRISE: "Enterprise",
}


def upsell(feature: str) -> str:
    """The message a locked feature shows instead of failing silently."""
    needed = FEATURES.get(feature)
    if needed is None:
        return "This feature is not available on your package."
    return (
        f"This is part of the {_LABELS[needed]} package. "
        "Ask your administrator to upgrade to enable it."
    )


# ── persistence ───────────────────────────────────────────────────────────────

def _settings_path() -> Path:
    try:
        return Path(str(config.get("paths", "runtime_settings")))
    except Exception:                                   # noqa: BLE001
        return Path("./settings.json")


def _read_all() -> dict:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text() or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read runtime settings ({exc}); using defaults")
        return {}


def current() -> Package:
    raw = _read_all().get(SETTINGS_KEY)
    if isinstance(raw, str):
        try:
            return Package(raw)
        except ValueError:
            logger.warning(f"Unknown package {raw!r} in settings; using default")
    return DEFAULT_PACKAGE


def set_package(name: str, actor: str | None = None) -> Package:
    try:
        pkg = Package(str(name).lower())
    except ValueError:
        valid = ", ".join(p.value for p in Package)
        raise HTTPException(422, f"Unknown package {name!r}. Valid: {valid}")

    with _LOCK:
        data = _read_all()
        data[SETTINGS_KEY] = pkg.value
        path = _settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    logger.info(f"Package set to {pkg.value}" + (f" by {actor}" if actor else ""))
    return pkg


# ── the gate ──────────────────────────────────────────────────────────────────

def has(feature: str) -> bool:
    """Is `feature` unlocked on the licensed package?"""
    if feature in ALWAYS_INCLUDED:
        return True
    needed = FEATURES.get(feature)
    if needed is None:
        # Unknown features are open. A typo must not silently disable something
        # that was working; the alternative fails closed in the worst place.
        logger.warning(f"Unknown feature {feature!r} treated as included")
        return True
    return _RANK[current()] >= _RANK[needed]


def require(feature: str) -> None:
    """Raise 403 with an explanation unless `feature` is unlocked.

    402 would arguably be more correct for a payment-gated resource, but 403 is
    what the existing assistant gate returns and what the web app already
    handles; one convention is worth more than the nicer status code.
    """
    if feature in ALWAYS_INCLUDED:
        return          # never gateable, whatever the caller passed
    if not has(feature):
        raise HTTPException(403, upsell(feature))


def status() -> dict:
    """Everything the frontend needs to render locks and upsells in one call."""
    pkg = current()
    return {
        "package": pkg.value,
        "label": _LABELS[pkg],
        "features": {name: has(name) for name in FEATURES},
        "always_included": sorted(ALWAYS_INCLUDED),
        "upsells": {name: upsell(name) for name in FEATURES if not has(name)},
    }

# ── What a buyer sees ────────────────────────────────────────────────────────
#
# The plan names, prices and included features the public pricing page renders.
# It lives here, beside the gate, on purpose: a pricing page maintained
# separately drifts from what the software actually enforces, and then the site
# is selling something the product does not do. `catalogue()` reads FEATURES
# and ALWAYS_INCLUDED directly, so a feature moved between tiers moves on the
# website in the same commit.
#
# Figures are indicative and derived from measured cost drivers, not from
# surveyed market rates. Nothing here charges anyone — see backend/billing.py.

FEATURE_LABELS: dict[str, str] = {
    "detection":          "All three detectors on every transaction",
    "fusion":             "Combined into one confidence score",
    "alerting":           "Email alerts to nominated risk managers",
    "monitoring":         "Always-on live monitor",
    "analysis_history":   "Analysis history and audit log",
    "attribution_panels": "Per-detector attribution — which signal fired, and why",
    "forensic_report":    "Grounded forensic report, as a filed PDF",
    "ablation_view":      "Grounded-vs-ungrounded comparison",
    "ai_assistant":       "Operator assistant",
    "batch_analysis":     "Batch analysis — score a file of transactions",
    "sar_draft":          "Suspicious activity report drafting",
    "threshold_sim":      "Threshold simulator",
    "governance_pack":    "Governance pack — model cards and validation evidence",
    "drift_monitoring":   "Drift monitoring",
    "custom_training":    "Training on the institution's own data",
}

_PLANS = {
    Package.ESSENTIAL: {
        "tagline": "Detection and alerting",
        "who": "A small institution that needs transactions screened and staff notified.",
        "price": "$40", "unit": "per analyst / month",
        "volume": "100,000 transactions a month included",
        "overage": "$0.30 per 1,000 after that",
        "term": "Monthly",
    },
    Package.PROFESSIONAL: {
        "tagline": "Investigation and evidence",
        "who": "A bank with a compliance function that has to justify its decisions.",
        "price": "$120", "unit": "per analyst / month",
        "volume": "1,000,000 transactions a month included",
        "overage": "$0.20 per 1,000 after that · 500 reports, then $0.40 each",
        "term": "Annual",
    },
    Package.ENTERPRISE: {
        "tagline": "Your institution, your governance",
        "who": "A bank that needs the system trained on its own data and governed under its own regulator.",
        "price": "From $25,000", "unit": "engagement, then annual licence",
        "volume": "Negotiated",
        "overage": "Negotiated",
        "term": "Annual",
    },
}


def catalogue() -> dict:
    """The plans, priced, with each one's features read off the gate table."""
    plans = []
    for pkg in (Package.ESSENTIAL, Package.PROFESSIONAL, Package.ENTERPRISE):
        included = sorted(ALWAYS_INCLUDED) + [
            f for f, need in FEATURES.items() if _RANK[need] <= _RANK[pkg]
        ]
        plans.append({
            "id": pkg.value,
            "name": _LABELS[pkg],
            **_PLANS[pkg],
            "features": [
                {"key": f, "label": FEATURE_LABELS.get(f, f),
                 "always_included": f in ALWAYS_INCLUDED}
                for f in included
            ],
        })
    return {
        "plans": plans,
        "current": current().value,
        # The claim the whole structure rests on, served rather than retyped
        # into the page, so it cannot quietly diverge from the code.
        "never_gated": [FEATURE_LABELS.get(f, f) for f in sorted(ALWAYS_INCLUDED)],
        "note": ("Indicative pricing, derived from measured cost drivers rather "
                 "than surveyed market rates."),
    }

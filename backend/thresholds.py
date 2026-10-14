"""The operating point the monitor actually alerts on.

The threshold page replays stored decisions at a different line, which is real
measurement — genuine scores against genuine labels. But a page that can only
measure is a calculator, not a control: an operator can conclude the line
should move and then has nowhere to move it.

So the chosen fused line is persisted here and the monitor prefers it over the
model's own defaults. Setting it changes which transactions raise an alert.

Only the fused verdict is settable. Each detector's internal threshold belongs
to that detector's service — the relational model already publishes its bands
on /health and the monitor reads them — so overriding those here would mean
this file quietly disagreeing with the model that owns them.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from backend import config

logger = logging.getLogger(__name__)

SETTINGS_KEY = "fused_bands"
_LOCK = threading.Lock()

# The operating point, and the only definition of it.
#
# There were three: this file reported 0.39/0.18/0.09, the monitor alerted on
# 0.925/0.09/0.03, and /analyze classified at 0.80/0.65/0.50 — so the page
# showed one line, the monitor fired on a second, and every stored verdict was
# labelled by a third. The values kept below are the ones that actually
# classified the data: every analysis on record was banded at 0.80/0.65/0.50,
# which makes them the operating point in fact, whatever the other two said.
#
# They are a chosen operating point, not a derived one — the fused score has no
# calibration of its own. What justifies moving them is the threshold page,
# which replays stored decisions at a new line and shows what it would have
# caught and missed. The relational detector's own 0.1830 is different: that
# one is tuned on the validation window and published by the model.
DEFAULT_BANDS = {"critical": 0.80, "high": 0.65, "medium": 0.50}
ORDER = ("critical", "high", "medium")


def _path() -> Path:
    try:
        return Path(str(config.get("paths", "runtime_settings")))
    except Exception:                                     # noqa: BLE001
        return Path("./settings.json")


def _read_all() -> dict:
    p = _path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text() or "{}")
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read runtime settings ({exc}); using defaults")
        return {}


def current() -> Optional[dict]:
    """The operator-set bands, or None if nobody has set any."""
    raw = _read_all().get(SETTINGS_KEY)
    if isinstance(raw, dict) and all(k in raw for k in ORDER):
        try:
            return {k: float(raw[k]) for k in ORDER}
        except (TypeError, ValueError):
            logger.warning("Stored fused bands are malformed; ignoring them")
    return None


def set_bands(bands: dict, actor: str | None = None) -> dict:
    """Persist the fused operating point. Validated, ordered, and audited."""
    try:
        parsed = {k: float(bands[k]) for k in ORDER}
    except (KeyError, TypeError, ValueError):
        raise HTTPException(422, f"Body must contain numeric {', '.join(ORDER)}.")

    for k, v in parsed.items():
        if not 0.0 <= v <= 1.0:
            raise HTTPException(422, f"{k} must be between 0 and 1, got {v}.")

    # A band that sits below the one beneath it would make the severity ladder
    # unorderable, and every case would land in whichever rung was checked first.
    if not parsed["critical"] >= parsed["high"] >= parsed["medium"]:
        raise HTTPException(
            422, "Bands must descend: critical ≥ high ≥ medium.")

    with _LOCK:
        data = _read_all()
        data[SETTINGS_KEY] = parsed
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2))

    logger.info(f"Fused bands set to {parsed}" + (f" by {actor}" if actor else ""))
    return parsed


def clear(actor: str | None = None) -> None:
    """Hand the operating point back to the model's own calibration."""
    with _LOCK:
        data = _read_all()
        if data.pop(SETTINGS_KEY, None) is not None:
            _path().write_text(json.dumps(data, indent=2))
            logger.info("Fused bands cleared" + (f" by {actor}" if actor else ""))

"""Entitlement for the operator assistant.

The assistant is a paid tier, not a default capability, so access is decided in
one place with two independent checks:

1. **Master switch** — an admin can turn the feature off for the whole
   deployment without a redeploy.
2. **Plan** — which roles the licence covers. Roles map to packages: an analyst
   seat is the base package, a risk-manager seat the professional one.

State lives in the runtime settings JSON the app already uses (`[paths]
runtime_settings`), so it survives restarts without a database migration and an
operator can inspect it as plain text.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import HTTPException

from backend import config

logger = logging.getLogger(__name__)

SETTINGS_KEY = "assistant"
_LOCK = threading.Lock()

# Roles are the licence unit. Analysts (read-only, base package) are excluded by
# default — enabling them is a deliberate admin action, which is what makes this
# a tier rather than a toggle.
DEFAULT_ALLOWED_ROLES = ["admin", "risk_manager"]

UPSELL = (
    "The AI assistant is part of the Professional package. Ask your "
    "administrator to enable it for your account."
)


@dataclass
class AssistantSettings:
    enabled: bool = False
    allowed_roles: list[str] = field(default_factory=lambda: list(DEFAULT_ALLOWED_ROLES))
    max_steps: int = 4           # tool calls per question — bounds cost and latency
    allow_live_analysis: bool = True   # may the assistant spend upstream calls?

    def to_dict(self) -> dict:
        return asdict(self)


def _settings_path() -> Path:
    try:
        return Path(str(config.get("paths", "runtime_settings")))
    except Exception:                                  # noqa: BLE001
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


def load() -> AssistantSettings:
    raw = _read_all().get(SETTINGS_KEY) or {}
    defaults = AssistantSettings()
    roles = raw.get("allowed_roles")
    return AssistantSettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        allowed_roles=[str(r) for r in roles] if isinstance(roles, list) else defaults.allowed_roles,
        max_steps=int(raw.get("max_steps", defaults.max_steps)),
        allow_live_analysis=bool(raw.get("allow_live_analysis", defaults.allow_live_analysis)),
    )


def save(changes: dict) -> AssistantSettings:
    """Apply a partial update. Unknown keys are rejected so a typo surfaces."""
    allowed = {"enabled", "allowed_roles", "max_steps", "allow_live_analysis"}
    unknown = set(changes) - allowed
    if unknown:
        raise HTTPException(
            422,
            f"Unknown setting(s): {', '.join(sorted(unknown))}. "
            f"Valid: {', '.join(sorted(allowed))}",
        )

    if "max_steps" in changes:
        try:
            steps = int(changes["max_steps"])
        except (TypeError, ValueError):
            raise HTTPException(422, "max_steps must be an integer")
        if not 1 <= steps <= 8:
            raise HTTPException(422, "max_steps must be between 1 and 8")
        changes["max_steps"] = steps

    if "allowed_roles" in changes:
        roles = changes["allowed_roles"]
        if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
            raise HTTPException(422, "allowed_roles must be a list of role names")
        valid = {"admin", "risk_manager", "analyst"}
        bad = set(roles) - valid
        if bad:
            raise HTTPException(422, f"Unknown role(s): {', '.join(sorted(bad))}")

    with _LOCK:
        data = _read_all()
        current = {**load().to_dict(), **changes}
        data[SETTINGS_KEY] = current
        path = _settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    return load()


def status_for(role: str) -> dict:
    """What the frontend needs to decide whether to render the assistant."""
    s = load()
    entitled = s.enabled and role in s.allowed_roles
    return {
        "available": entitled,
        "enabled": s.enabled,
        "entitled": role in s.allowed_roles,
        "reason": None if entitled else (
            "The AI assistant is disabled for this deployment."
            if not s.enabled else UPSELL
        ),
    }


def require_entitled(role: str) -> AssistantSettings:
    """Raise 403 unless this role may use the assistant."""
    s = load()
    if not s.enabled:
        raise HTTPException(403, "The AI assistant is disabled for this deployment.")
    if role not in s.allowed_roles:
        raise HTTPException(403, UPSELL)
    return s

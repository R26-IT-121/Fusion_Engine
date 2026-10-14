"""Whether the graph explorer is switched on, and how far it may reach.

The explorer is the one screen that can put real load on the network detector:
every search walks the served graph and every expansion walks it again. On a
laptop running all six services that is noticeable, and during a demo it is
the last thing anyone wants competing for memory with the pipeline.

So it has a switch. Off by default is the wrong choice — a feature nobody can
find is a feature nobody uses — but an administrator being able to turn it off
for the duration of a demo, without a deploy, is worth the twenty lines.

The reach limits live here too rather than in the UI. A limit the browser
enforces is a suggestion; this one is applied where the request is served.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)
_lock = threading.Lock()

DEFAULTS = {
    "enabled": True,
    # One hop is the honest default. Two is available, but on this graph most
    # senders are one-shot accounts with no second hop to find, so it usually
    # costs time and returns the same picture.
    "max_hops": 2,
    "max_edges": 150,
}


def _path() -> Path:
    from backend import config

    return Path(str(config.get("paths", "runtime_settings"))).with_name(
        "graph_explorer.json")


def current() -> dict:
    """Never raises. An unreadable settings file must not take the page down."""
    out = dict(DEFAULTS)
    try:
        raw = json.loads(_path().read_text())
        if isinstance(raw, dict):
            if "enabled" in raw:
                out["enabled"] = bool(raw["enabled"])
            if "max_hops" in raw:
                out["max_hops"] = max(1, min(int(raw["max_hops"]), 2))
            if "max_edges" in raw:
                out["max_edges"] = max(10, min(int(raw["max_edges"]), 400))
    except FileNotFoundError:
        pass
    except Exception as exc:                             # noqa: BLE001
        logger.warning(f"Could not read the graph-explorer settings ({exc}).")
    return out


def update(enabled: bool | None = None, max_hops: int | None = None,
           max_edges: int | None = None, actor: str | None = None) -> dict:
    cfg = current()
    if enabled is not None:
        cfg["enabled"] = bool(enabled)
    if max_hops is not None:
        cfg["max_hops"] = max(1, min(int(max_hops), 2))
    if max_edges is not None:
        cfg["max_edges"] = max(10, min(int(max_edges), 400))

    with _lock:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**cfg, "set_by": actor}, indent=2))
    logger.info(f"Graph explorer settings changed by {actor}: {cfg}")
    return cfg

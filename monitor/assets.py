"""Images embedded in the alert email.

Attached as CID parts rather than linked. Remote images are blocked by
default in most inboxes, so a hosted banner would show as a broken box in
exactly the message that most needs to be read at a glance.

Read once and cached: an alert is sent per case, and re-reading four files
from disk each time buys nothing.

Every lookup returns None rather than raising when a file is absent. An
alert that arrives without its banner is a cosmetic loss; an alert that does
not arrive because a JPEG was missing is a real one.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets"

# Content-IDs the HTML refers to as cid:<name>.
LOGO_CID = "ds-logo"
BANNER_CID = "ds-banner"

# Rendered width in the email, in CSS pixels. The files are twice this so
# they stay sharp on a retina display; the width attribute is what holds the
# layout in Outlook, which ignores max-width.
BANNER_WIDTH = 548
LOGO_WIDTH = 26

_cache: dict[str, bytes | None] = {}


def _looks_like_an_image(data: bytes) -> bool:
    """JPEG or PNG magic bytes.

    A file that is present but not an image is worse than one that is absent:
    absent degrades to alt text, whereas a truncated JPEG is attached, declared
    as some content type, and renders as a broken box in the alert. Checked
    here so the two failures behave the same way.
    """
    return data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"


def _read(name: str) -> bytes | None:
    if name in _cache:
        return _cache[name]
    path = ASSETS / name
    try:
        data = path.read_bytes()
        if not _looks_like_an_image(data):
            logger.warning(
                f"Alert asset {name} is not a JPEG or PNG ({len(data)} bytes); "
                "skipping it rather than attaching a broken image.")
            data = None
    except Exception as exc:                             # noqa: BLE001
        logger.info(f"Alert asset {name} unavailable: {exc}")
        data = None
    _cache[name] = data
    return data


def banner(severity: str) -> bytes | None:
    """The severity strip for the top of the message."""
    return _read(f"banner-{(severity or '').lower()}.jpg")


def logo() -> bytes | None:
    return _read("logo.png")


def inline_for(severity: str) -> dict[str, bytes]:
    """The CID map to hand the mailer, skipping anything that is missing."""
    out: dict[str, bytes] = {}
    b = banner(severity)
    if b:
        out[BANNER_CID] = b
    lg = logo()
    if lg:
        out[LOGO_CID] = lg
    return out

"""Draw the extracted subgraph as a PNG for the alert email.

Why a raster image and not SVG: Gmail strips inline SVG entirely, and blocks
remote images by default. A PNG attached with a Content-ID is the only form
that reliably renders in an inbox, so that is what this produces.

What it draws is the real thing — the accounts, roles, amounts and per-edge
attention weights that the relational model's extractor returned for this
transaction. Nothing here is illustrative.
"""
from __future__ import annotations

import io
import logging
import math

logger = logging.getLogger("deepsentinel")

# Rendered at 2x and displayed at half size, so it stays sharp on a phone.
SCALE = 2
W, H = 560, 250

INK = (20, 24, 26)
MUTED = (94, 105, 108)
FAINT = (140, 148, 151)
RULE = (220, 227, 227)
PAPER = (255, 255, 255)
TEAL = (15, 118, 110)
CORAL = (176, 57, 44)
AMBER = (166, 106, 8)

# The extractor's own vocabulary. Anything unrecognised draws neutral rather
# than guessing at a meaning.
ROLE_COLOUR = {
    "MULE_CENTRAL": CORAL,
    "SINK": CORAL,
    "FRESH_SENDER": AMBER,
    "SENDER": TEAL,
    "INTERMEDIARY": TEAL,
}

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    for path in (_BOLD_CANDIDATES if bold else _FONT_CANDIDATES):
        try:
            return ImageFont.truetype(path, size * SCALE)
        except Exception:                               # noqa: BLE001
            continue
    # Better a bitmap font than no diagram.
    return ImageFont.load_default()


def _short(account: str, keep: int = 11) -> str:
    return account if len(account) <= keep else account[: keep - 1] + "\u2026"


def _money(v: float) -> str:
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return f"{v:.0f}"


def render_subgraph(sg: dict) -> bytes | None:
    """PNG bytes for the alert email, or None if there is nothing to draw.

    Returning None is a normal outcome — a transaction between two accounts
    the model has never linked has no structure worth a picture, and the email
    is written to work without one.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:                                 # noqa: BLE001
        return None

    nodes = list(sg.get("nodes") or [])
    edges = list(sg.get("edges") or [])
    sink_id = sg.get("sink_account")
    if not nodes or not edges or not sink_id:
        return None

    by_id = {n.get("account_id"): n for n in nodes}
    senders = [n for n in nodes if n.get("account_id") != sink_id]
    if not senders:
        return None

    # Busiest senders first, so a truncated drawing keeps the ones that matter.
    weight = {}
    for e in edges:
        weight[e.get("src")] = max(weight.get(e.get("src"), 0.0),
                                   float(e.get("edge_attention_weight") or 0.0))
    senders.sort(key=lambda n: weight.get(n.get("account_id"), 0.0), reverse=True)
    hidden = max(0, len(senders) - 8)
    shown = senders[:8]

    img = Image.new("RGB", (W * SCALE, H * SCALE), PAPER)
    d = ImageDraw.Draw(img)
    f_lbl = _font(9)
    f_acct = _font(9)
    f_bold = _font(10, bold=True)
    f_cap = _font(8)

    d.rectangle([0, 0, W * SCALE - 1, H * SCALE - 1], outline=RULE, width=1 * SCALE)

    top, bottom = 54, H - 34
    sink_x, sink_y = int(W * 0.72), (top + bottom) // 2
    send_x = int(W * 0.26)
    span = bottom - top
    step = span / max(len(shown) - 1, 1) if len(shown) > 1 else 0
    y0 = top if len(shown) > 1 else sink_y

    pos = {}
    for i, n in enumerate(shown):
        pos[n["account_id"]] = (send_x, int(y0 + i * step))
    pos[sink_id] = (sink_x, sink_y)

    # ── edges first, so nodes sit on top of them ──
    max_attn = max((float(e.get("edge_attention_weight") or 0.0) for e in edges), default=0.0) or 1.0
    for e in edges:
        a, b = pos.get(e.get("src")), pos.get(e.get("dst"))
        if not a or not b:
            continue
        trigger = bool(e.get("is_trigger_edge"))
        attn = float(e.get("edge_attention_weight") or 0.0)
        width = 1 + round(2.6 * (attn / max_attn))
        colour = CORAL if trigger else (176, 190, 190)
        d.line([a[0] * SCALE, a[1] * SCALE, b[0] * SCALE, b[1] * SCALE],
               fill=colour, width=max(1, width) * SCALE)

        # The transfer that triggered this alert is the one worth labelling.
        if trigger:
            mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 9
            label = _money(float(e.get("amount") or 0.0))
            tw = d.textlength(label, font=f_bold)
            d.rectangle([mx * SCALE - tw / 2 - 5 * SCALE, my * SCALE - 2 * SCALE,
                         mx * SCALE + tw / 2 + 5 * SCALE, my * SCALE + 13 * SCALE],
                        fill=PAPER)
            d.text((mx * SCALE - tw / 2, my * SCALE), label, font=f_bold, fill=CORAL)

    # ── nodes ──
    for n in shown:
        x, y = pos[n["account_id"]]
        colour = ROLE_COLOUR.get(str(n.get("role") or "").upper(), MUTED)
        r = 6
        d.ellipse([(x - r) * SCALE, (y - r) * SCALE, (x + r) * SCALE, (y + r) * SCALE],
                  fill=PAPER, outline=colour, width=2 * SCALE)
        # Labels go to the left of the dots. The fan of edges opens rightward,
        # so anything written on that side gets struck through by its own
        # evidence.
        acct = _short(n["account_id"])
        role = str(n.get("role") or "").replace("_", " ").lower()
        aw = d.textlength(acct, font=f_acct)
        rw = d.textlength(role, font=f_cap)
        d.text(((x - 12) * SCALE - aw, (y - 6) * SCALE), acct, font=f_acct, fill=INK)
        d.text(((x - 12) * SCALE - rw, (y + 3) * SCALE), role, font=f_cap, fill=FAINT)

    sx, sy = pos[sink_id]
    sink_node = by_id.get(sink_id, {})
    r = 13
    d.ellipse([(sx - r - 4) * SCALE, (sy - r - 4) * SCALE,
               (sx + r + 4) * SCALE, (sy + r + 4) * SCALE], outline=(244, 214, 209),
              width=2 * SCALE)
    d.ellipse([(sx - r) * SCALE, (sy - r) * SCALE, (sx + r) * SCALE, (sy + r) * SCALE],
              fill=CORAL, outline=CORAL)
    d.text(((sx + r + 12) * SCALE, (sy - 11) * SCALE), _short(sink_id, 14),
           font=f_bold, fill=INK)
    d.text(((sx + r + 12) * SCALE, (sy + 1) * SCALE), "collection account",
           font=f_cap, fill=CORAL)
    recv = float(sink_node.get("total_received_amount") or 0.0)
    if recv:
        d.text(((sx + r + 12) * SCALE, (sy + 11) * SCALE),
               f"received {_money(recv)} in total", font=f_cap, fill=FAINT)

    # ── header and footnote ──
    pattern = str(sg.get("pattern") or "structure").replace("_", " ").title()
    d.text((18 * SCALE, 18 * SCALE), pattern, font=_font(13, bold=True), fill=INK)
    ev = sg.get("structural_evidence") or {}
    conv = ev.get("convergence_count")
    sub = f"{conv} senders converging" if conv else f"{len(nodes)} accounts, {len(edges)} transfers"
    d.text((18 * SCALE, 36 * SCALE), sub, font=f_lbl, fill=MUTED)

    foot = "line thickness = edge attention  ·  red = the transfer that triggered this alert"
    if hidden:
        foot = f"{hidden} further sender(s) not drawn  ·  " + foot
    d.text((18 * SCALE, (H - 20) * SCALE), foot, font=f_cap, fill=FAINT)

    img = img.resize((W, H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()

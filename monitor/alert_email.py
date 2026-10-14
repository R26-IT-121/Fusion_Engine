"""The HTML body of a monitor alert.

Written to the rules of email, not of the web. Tables for layout, styles
inline, no external stylesheet, no SVG, no flexbox — Gmail and Outlook strip or
ignore all of those, and an alert that renders as a wall of unstyled text is
worse than the plain-text one it replaced.

It commits to a light palette. Email clients apply their own dark-mode
inversion with no way to opt in properly, so every surface here paints an
explicit background rather than inheriting one.

The plain-text alternative is built alongside it and carries the same facts, so
a text-only client loses the styling and nothing else.
"""
from __future__ import annotations

from monitor import assets

SEVERITY = {
    "CRITICAL": ("#B0392C", "#FAEBE8"),
    "HIGH":     ("#A66A08", "#FBF0DF"),
    "MEDIUM":   ("#544799", "#EDEBF8"),
    "LOW":      ("#157A3D", "#E7F3EC"),
}
INK, MUTED, FAINT, RULE = "#14181A", "#5E696C", "#8A9497", "#DCE3E3"
PAPER, GROUND = "#FFFFFF", "#F5F7F7"
MONO = "'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"

DETECTORS = [
    ("graph", "Network", "the payment graph around it"),
    ("behavioural", "Behaviour", "how it fits its transaction type"),
    ("temporal", "Timing", "the transactions just before it"),
]


def _bar(value: float | None, colour: str) -> str:
    """A score bar built from table cells, which is the only bar an email client
    can be trusted to draw."""
    if value is None:
        return (f'<span style="font:12px {SANS};color:{FAINT};font-style:italic;">'
                f'did not answer</span>')
    pct = max(2, min(100, round(value * 100)))
    return (
        f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        f'style="width:150px;border-collapse:collapse;"><tr>'
        f'<td style="height:7px;width:{pct}%;background:{colour};'
        f'border-radius:4px 0 0 4px;font-size:0;line-height:0;">&nbsp;</td>'
        f'<td style="height:7px;background:{RULE};border-radius:0 4px 4px 0;'
        f'font-size:0;line-height:0;">&nbsp;</td>'
        f'</tr></table>'
    )


def _scale(fused: float, bands: dict) -> str:
    """Where this verdict sits against the operating point, as a marked rule."""
    med, high, crit = (float(bands.get("medium", 0.03)),
                       float(bands.get("high", 0.09)),
                       float(bands.get("critical", 0.925)))
    # The bands are far apart in probability but the interesting action is at
    # the bottom, so the rule is drawn on the band index rather than linearly —
    # a linear axis would put every ordinary transaction in the same pixel.
    stops = [0.0, med, high, crit, 1.0]
    seg = 0
    for i in range(4):
        if fused >= stops[i]:
            seg = i
    within = 0.0
    lo, hi = stops[seg], stops[seg + 1]
    if hi > lo:
        within = min(1.0, max(0.0, (fused - lo) / (hi - lo)))
    pos = round(((seg + within) / 4) * 100)

    cells = ""
    for label, colour in (("low", "#C8D2D2"), ("medium", SEVERITY["MEDIUM"][0]),
                          ("high", SEVERITY["HIGH"][0]), ("critical", SEVERITY["CRITICAL"][0])):
        cells += (f'<td style="width:25%;height:6px;background:{colour};'
                  f'font-size:0;line-height:0;">&nbsp;</td>'
                  f'<td style="width:2px;font-size:0;line-height:0;background:{PAPER};">&nbsp;</td>')
    return (
        f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        f'style="width:100%;border-collapse:collapse;"><tr>'
        f'<td style="width:{pos}%;font-size:0;line-height:0;">&nbsp;</td>'
        f'<td style="font:700 11px {MONO};color:{INK};white-space:nowrap;'
        f'padding-bottom:3px;">&#9660;</td>'
        f'<td style="font-size:0;line-height:0;">&nbsp;</td></tr></table>'
        f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        f'style="width:100%;border-collapse:collapse;"><tr>{cells}</tr></table>'
        f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
        f'style="width:100%;border-collapse:collapse;margin-top:5px;"><tr>'
        f'<td style="font:10px {MONO};color:{FAINT};letter-spacing:.08em;">LOW</td>'
        f'<td style="font:10px {MONO};color:{FAINT};letter-spacing:.08em;text-align:center;">'
        f'MEDIUM {med:g}</td>'
        f'<td style="font:10px {MONO};color:{FAINT};letter-spacing:.08em;text-align:center;">'
        f'HIGH {high:g}</td>'
        f'<td style="font:10px {MONO};color:{FAINT};letter-spacing:.08em;text-align:right;">'
        f'CRITICAL {crit:g}</td>'
        f'</tr></table>'
    )


def _row(label: str, value: str, mono: bool = False) -> str:
    face = MONO if mono else SANS
    return (
        f'<tr>'
        f'<td style="padding:8px 0;border-bottom:1px solid {RULE};font:12px {SANS};'
        f'color:{MUTED};width:38%;">{label}</td>'
        f'<td style="padding:8px 0;border-bottom:1px solid {RULE};font:13px {face};'
        f'color:{INK};text-align:right;">{value}</td>'
        f'</tr>'
    )


def build(alert: dict, sg: dict, scores: dict, bands: dict,
          has_image: bool, case_ref: str | None, console_url: str,
          report_attached: bool) -> str:
    sev = alert["severity"]
    colour, tint = SEVERITY.get(sev, (MUTED, GROUND))
    fused = float(alert["fused_score"])
    ev = sg.get("structural_evidence") or {}

    detector_rows = ""
    for key, name, what in DETECTORS:
        v = scores.get(key)
        shown = f"{v:.4f}" if v is not None else "—"
        detector_rows += (
            f'<tr>'
            f'<td style="padding:9px 12px 9px 0;border-bottom:1px solid {RULE};">'
            f'<div style="font:600 13px {SANS};color:{INK};">{name}</div>'
            f'<div style="font:11px {SANS};color:{FAINT};padding-top:2px;">{what}</div></td>'
            f'<td style="padding:9px 12px;border-bottom:1px solid {RULE};width:150px;">'
            f'{_bar(v, colour)}</td>'
            f'<td style="padding:9px 0;border-bottom:1px solid {RULE};font:13px {MONO};'
            f'color:{INK if v is not None else FAINT};text-align:right;width:62px;">{shown}</td>'
            f'</tr>'
        )

    missing = 3 - int(alert.get("modalities_used") or 0)
    caveat = ""
    if missing:
        caveat = (
            f'<tr><td style="padding:12px 16px;background:{GROUND};border-radius:6px;'
            f'font:12px {SANS};color:{MUTED};line-height:1.5;">'
            f'<strong style="color:{INK};">{missing} detector'
            f'{"s" if missing > 1 else ""} did not answer.</strong> '
            f'Their signal is excluded rather than counted as low, and the confidence '
            f'above is deliberately less certain in both directions because of it.'
            f'</td></tr><tr><td style="height:18px;"></td></tr>'
        )

    # The severity strip across the top, and the mark beside the wordmark.
    #
    # Both are rendered only when the file is actually there. An inbox that
    # blocks images then falls back to the alt text and the coloured rule
    # below, so the severity is still legible with nothing loaded — which is
    # how a good number of recipients will first see it.
    banner_block = ""
    if assets.banner(sev):
        banner_block = (
            f'<tr><td style="font-size:0;line-height:0;">'
            f'<img src="cid:{assets.BANNER_CID}" width="{assets.BANNER_WIDTH}" '
            f'alt="{sev.title()} severity" '
            f'style="display:block;width:100%;max-width:600px;height:auto;border:0;">'
            f'</td></tr>'
        )

    logo_img = ""
    if assets.logo():
        logo_img = (
            f'<img src="cid:{assets.LOGO_CID}" width="{assets.LOGO_WIDTH}" '
            f'height="{assets.LOGO_WIDTH}" alt="" '
            f'style="vertical-align:middle;margin-right:7px;border:0;'
            f'width:{assets.LOGO_WIDTH}px;height:{assets.LOGO_WIDTH}px;">'
        )

    image_block = ""
    if has_image:
        image_block = (
            f'<tr><td style="padding-bottom:6px;font:600 11px {MONO};color:{FAINT};'
            f'letter-spacing:.11em;">THE STRUCTURE THAT WAS FOUND</td></tr>'
            f'<tr><td style="padding-bottom:20px;">'
            f'<img src="cid:subgraph" width="560" alt="Extracted subgraph: senders '
            f'converging on {sg.get("sink_account", "a collection account")}" '
            f'style="display:block;width:100%;max-width:560px;height:auto;'
            f'border:1px solid {RULE};border-radius:6px;"></td></tr>'
        )

    evidence = ""
    if ev:
        evidence = (
            f'<tr><td style="padding-bottom:6px;font:600 11px {MONO};color:{FAINT};'
            f'letter-spacing:.11em;">STRUCTURAL EVIDENCE</td></tr>'
            f'<tr><td style="padding-bottom:20px;">'
            f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" '
            f'style="width:100%;border-collapse:collapse;">'
            + _row("Senders converging on the sink", str(ev.get("convergence_count", "—")), True)
            + _row("Share of them brand new", f'{float(ev.get("fresh_sender_ratio") or 0) * 100:.0f}%', True)
            + _row("Known mules in the subgraph", str(ev.get("mules_in_subgraph", "—")), True)
            + '</table></td></tr>'
        )

    case_button = ""
    if case_ref:
        case_button = (
            f'<tr><td style="padding-bottom:22px;">'
            f'<a href="{console_url}/cases" '
            f'style="display:inline-block;background:{INK};color:#FFFFFF;'
            f'text-decoration:none;font:600 13px {SANS};padding:11px 20px;'
            f'border-radius:6px;">Open case {case_ref}</a></td></tr>'
        )

    attach_note = ""
    if report_attached:
        attach_note = (
            f'<tr><td style="padding:12px 16px;background:{tint};border-radius:6px;'
            f'font:12px {SANS};color:{MUTED};line-height:1.5;">'
            f'The forensic report is attached. Every claim in it is grounded in the '
            f'scores above and the retrieved typology — it cites evidence rather '
            f'than speculating.</td></tr><tr><td style="height:18px;"></td></tr>'
        )

    return f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light only">
<title>{sev} fraud alert {alert['transaction_id']}</title></head>
<body style="margin:0;padding:0;background:{GROUND};">
<table role="presentation" cellpadding="0" cellspacing="0" border="0"
       style="width:100%;background:{GROUND};border-collapse:collapse;">
<tr><td align="center" style="padding:26px 12px;">

<table role="presentation" cellpadding="0" cellspacing="0" border="0"
       style="width:100%;max-width:600px;background:{PAPER};border:1px solid {RULE};
              border-radius:10px;border-collapse:separate;overflow:hidden;">

  {banner_block}
  <tr><td style="height:4px;background:{colour};font-size:0;line-height:0;">&nbsp;</td></tr>

  <tr><td style="padding:22px 26px 0;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" style="width:100%;">
      <tr>
        <td style="font:700 11px {MONO};color:{colour};letter-spacing:.13em;">
          {sev} &nbsp;&middot;&nbsp; FUSED VERDICT
        </td>
        <td style="text-align:right;white-space:nowrap;font:11px {MONO};color:{FAINT};">
          {logo_img}DeepSentinel
        </td>
      </tr>
    </table>
  </td></tr>

  <tr><td style="padding:14px 26px 0;">
    <div style="font:700 40px {MONO};color:{INK};letter-spacing:-.02em;line-height:1;">
      {fused:.4f}
    </div>
    <div style="font:13px {SANS};color:{MUTED};padding-top:7px;line-height:1.5;">
      Fused confidence for
      <span style="font-family:{MONO};color:{INK};">{alert['transaction_id']}</span>,
      from {alert.get('modalities_used', 0)} of 3 detectors.
    </div>
  </td></tr>

  <tr><td style="padding:20px 26px 24px;">{_scale(fused, bands)}</td></tr>

  <tr><td style="padding:0 26px;">
    <table role="presentation" cellpadding="0" cellspacing="0" border="0"
           style="width:100%;border-collapse:collapse;">
      {caveat}
      {image_block}

      <tr><td style="padding-bottom:6px;font:600 11px {MONO};color:{FAINT};
                     letter-spacing:.11em;">WHAT EACH DETECTOR SAID</td></tr>
      <tr><td style="padding-bottom:22px;">
        <table role="presentation" cellpadding="0" cellspacing="0" border="0"
               style="width:100%;border-collapse:collapse;">{detector_rows}</table>
      </td></tr>

      <tr><td style="padding-bottom:6px;font:600 11px {MONO};color:{FAINT};
                     letter-spacing:.11em;">THE TRANSACTION</td></tr>
      <tr><td style="padding-bottom:22px;">
        <table role="presentation" cellpadding="0" cellspacing="0" border="0"
               style="width:100%;border-collapse:collapse;">
          {_row("Amount", f"{alert['amount']:,.2f}", True)}
          {_row("From", alert['from'], True)}
          {_row("To", alert['to'], True)}
          {_row("Pattern", str(alert.get('pattern') or '—').replace('_', ' ').title())}
          {_row("Collection account", str(alert.get('sink_account') or '—'), True)}
        </table>
      </td></tr>

      {evidence}
      {attach_note}
      {case_button}
    </table>
  </td></tr>

  <tr><td style="padding:16px 26px 22px;border-top:1px solid {RULE};
                 font:11px {SANS};color:{FAINT};line-height:1.6;">
    Sent because this transaction fused at or above the MEDIUM band. Thresholds are
    set on the Thresholds page and apply from the next transaction onward.
  </td></tr>

</table>
</td></tr></table>
</body></html>"""


def build_text(alert: dict, sg: dict, scores: dict, report_attached: bool) -> str:
    """The plain-text alternative. Same facts, no styling."""
    ev = sg.get("structural_evidence") or {}
    lines = [
        f"CONFIRMED {alert['severity']} - fused verdict",
        "=" * 44,
        f"Transaction : {alert['transaction_id']}",
        f"Fused score : {alert['fused_score']:.4f} "
        f"({alert.get('modalities_used', 0)} of 3 detectors available)",
        "",
        "What each detector said",
    ]
    for key, name, _ in DETECTORS:
        v = scores.get(key)
        lines.append(f"  {name:<11} : {f'{v:.4f}' if v is not None else 'did not answer'}")
    lines += [
        "",
        f"Amount      : {alert['amount']:,.2f}",
        f"From -> To  : {alert['from']} -> {alert['to']}",
        f"Pattern     : {alert.get('pattern') or 'n/a'}",
        f"Sink        : {alert.get('sink_account') or 'n/a'}",
    ]
    if ev:
        lines += [
            "",
            "Structural evidence",
            f"  senders converging : {ev.get('convergence_count')}",
            f"  brand-new senders  : {ev.get('fresh_sender_ratio')}",
            f"  mules in subgraph  : {ev.get('mules_in_subgraph')}",
        ]
    if report_attached:
        lines += ["", "The forensic report is attached."]
    return "\n".join(lines) + "\n"


def sample(severity: str = "HIGH") -> tuple[dict, dict, dict]:
    """A representative alert, for the Settings preview and the test send.

    Both used to build their own FraudAlert against a second, older template in
    email_service. That meant three different emails existed — what the preview
    showed, what the test button sent, and what the monitor actually delivered —
    and only the third was real. A preview that does not render the shipping
    template is worse than no preview, because it is believed.

    Numbers are plausible rather than round, so the preview exercises the same
    formatting the live path does.
    """
    fused = {"CRITICAL": 0.9612, "HIGH": 0.5373,
             "MEDIUM": 0.1401, "LOW": 0.0083}.get(severity.upper(), 0.5373)
    alert = {
        "transaction_id": "PREVIEW-004821",
        "severity": severity.upper(),
        "fused_score": fused,
        "graph_score": 0.0313,
        "pattern": "HUB_AND_SPOKE",
        "sink_account": "C1166671647",
        "amount": 1326310.62,
        "from": "C2028127673",
        "to": "C1166671647",
        "modalities_used": 3,
        "fusion_method": "meta_classifier",
        "driver": "behavioural",
        "scores": {"graph": 0.0313, "behavioural": 0.9012, "temporal": 0.7745},
        "at": 0,
    }
    sg = {
        "pattern": "HUB_AND_SPOKE",
        "sink_account": "C1166671647",
        "structural_evidence": {
            "convergence_count": 7,
            "fresh_sender_ratio": 0.86,
            "mules_in_subgraph": 3,
        },
    }
    scores = {"graph": 0.0313, "behavioural": 0.9012, "temporal": 0.7745}
    return alert, sg, scores

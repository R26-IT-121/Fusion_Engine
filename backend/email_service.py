"""
Fraud alert email.

Sending requires an identity to send *from*. Two are supported:

  SMTP      A dedicated sending account and an app password. Simplest to set up
            and works immediately. The app password is a scoped credential
            revocable on its own, not the account password — treat it like an
            API key and never use a personal account.

  SendGrid  An API key. Better deliverability at volume and no SMTP port to
            worry about, but the sender address must be verified first.

With neither configured, nothing is sent and the caller is told so. An earlier
version returned success in that case, so the UI reported delivery for mail that
was never sent.
"""

import asyncio
import logging
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from enum import Enum
from typing import List, Optional

import httpx

from backend import config

logger = logging.getLogger(__name__)

SENDER_EMAIL = config.get("email", "sender_email")
SENDER_NAME = config.get("email", "sender_name")


class SendOutcome(str, Enum):
    SENT = "sent"
    NOT_CONFIGURED = "not_configured"
    FAILED = "failed"


@dataclass
class SendResult:
    outcome: SendOutcome
    provider: Optional[str] = None
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.outcome is SendOutcome.SENT


@dataclass
class FraudAlert:
    """Fraud alert data for email"""

    transaction_id: str
    fraud_confidence: float
    classification: str  # CRITICAL, HIGH, MEDIUM, LOW
    timestamp: str
    graph_score: Optional[float] = None
    behavioral_score: Optional[float] = None
    temporal_score: Optional[float] = None
    graph_signal: Optional[str] = None
    behavioral_signal: Optional[str] = None
    temporal_signal: Optional[str] = None
    forensic_report: Optional[str] = None
    typology_name: Optional[str] = None
    typology_id: Optional[str] = None


def _provider() -> tuple[Optional[str], dict]:
    """
    Resolve which provider is configured. Read per call rather than at import so
    editing config.ini and restarting is enough — and so a test can change it.
    """
    smtp_host = config.get("email", "smtp_host")
    smtp_user = config.get("secrets", "smtp_username")
    smtp_pass = config.get("secrets", "smtp_password")
    if smtp_host and smtp_user and smtp_pass:
        return "smtp", {
            "host": smtp_host,
            "port": config.get("email", "smtp_port"),
            "username": smtp_user,
            "password": smtp_pass,
            "use_tls": config.get("email", "smtp_use_tls"),
        }

    sendgrid_key = config.get("secrets", "sendgrid_api_key")
    if sendgrid_key:
        return "sendgrid", {"api_key": sendgrid_key}

    return None, {}


async def send_fraud_alert(
    alert: FraudAlert,
    recipient_emails: List[str],
    backend_url: str = "http://localhost:8000",
) -> SendResult:
    """Send a fraud alert. Reports what actually happened — never a false success."""
    if not recipient_emails:
        return SendResult(
            SendOutcome.NOT_CONFIGURED,
            detail="No recipients were given.",
        )

    provider, settings = _provider()
    if provider is None:
        logger.warning(
            "Email not sent for %s — no provider configured.", alert.transaction_id
        )
        return SendResult(
            SendOutcome.NOT_CONFIGURED,
            detail=(
                "No email provider is configured, so nothing was sent. Set either "
                "SMTP credentials or a SendGrid API key in config.ini."
            ),
        )

    html_body = _build_email_html(alert, backend_url)

    if provider == "smtp":
        # smtplib is blocking; keep it off the event loop.
        return await asyncio.to_thread(
            _send_via_smtp, alert, recipient_emails, html_body, settings
        )
    return await _send_via_sendgrid(
        alert, recipient_emails, html_body, settings["api_key"]
    )


def _send_via_smtp(
    alert: FraudAlert, recipients: List[str], html_body: str, s: dict
) -> SendResult:
    message = EmailMessage()
    message["Subject"] = f"Fraud alert: {alert.classification} risk on {alert.transaction_id}"
    message["From"] = f"{SENDER_NAME} <{s['username']}>"
    message["To"] = ", ".join(recipients)
    # A plain-text part matters: some clients and most spam filters penalise
    # HTML-only mail.
    message.set_content(_build_email_text(alert))
    message.add_alternative(html_body, subtype="html")

    try:
        context = ssl.create_default_context()
        if int(s["port"]) == 465:
            with smtplib.SMTP_SSL(s["host"], int(s["port"]), context=context, timeout=20) as srv:
                srv.login(s["username"], s["password"])
                srv.send_message(message)
        else:
            with smtplib.SMTP(s["host"], int(s["port"]), timeout=20) as srv:
                if s["use_tls"]:
                    srv.starttls(context=context)
                srv.login(s["username"], s["password"])
                srv.send_message(message)

        logger.info(f"Alert sent via SMTP: {alert.transaction_id} → {recipients}")
        return SendResult(SendOutcome.SENT, provider="smtp")

    except smtplib.SMTPAuthenticationError:
        return SendResult(
            SendOutcome.FAILED,
            provider="smtp",
            detail=(
                "The mail server rejected those credentials. For Gmail this "
                "usually means a normal account password was used instead of an "
                "app password, or two-factor authentication is not enabled on "
                "the sending account."
            ),
        )
    except (smtplib.SMTPException, OSError) as e:
        return SendResult(
            SendOutcome.FAILED,
            provider="smtp",
            detail=f"{type(e).__name__}: {e}",
        )


async def _send_via_sendgrid(
    alert: FraudAlert, recipients: List[str], html_body: str, api_key: str
) -> SendResult:
    payload = {
        "personalizations": [{"to": [{"email": e} for e in recipients]}],
        "from": {"email": SENDER_EMAIL, "name": SENDER_NAME},
        "subject": f"Fraud alert: {alert.classification} risk on {alert.transaction_id}",
        "content": [
            {"type": "text/plain", "value": _build_email_text(alert)},
            {"type": "text/html", "value": html_body},
        ],
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15.0,
            )

        if resp.status_code in (200, 202):
            logger.info(f"Alert sent via SendGrid: {alert.transaction_id} → {recipients}")
            return SendResult(SendOutcome.SENT, provider="sendgrid")

        # 403 here is nearly always an unverified sender, which the generic
        # message does not make obvious.
        detail = f"SendGrid returned {resp.status_code}: {resp.text[:300]}"
        if resp.status_code == 403:
            detail += (
                f" — most often this means '{SENDER_EMAIL}' has not been verified "
                f"as a sender in the SendGrid account."
            )
        logger.error(detail)
        return SendResult(SendOutcome.FAILED, provider="sendgrid", detail=detail)

    except Exception as e:
        logger.error(f"SendGrid request failed: {type(e).__name__}: {e}")
        return SendResult(
            SendOutcome.FAILED, provider="sendgrid", detail=f"{type(e).__name__}: {e}"
        )


def _build_email_text(alert: FraudAlert) -> str:
    """Plain-text alternative. HTML-only mail is penalised by spam filters."""
    lines = [
        f"{alert.classification} RISK — fraud confidence {alert.fraud_confidence:.0%}",
        "",
        f"Transaction: {alert.transaction_id}",
        f"Detected:    {alert.timestamp}",
        "",
        "Model scores",
    ]
    for label, score, signal in (
        ("Network (GraphSAGE)", alert.graph_score, alert.graph_signal),
        ("Behaviour (VAE)", alert.behavioral_score, alert.behavioral_signal),
        ("Timing (TCN)", alert.temporal_score, alert.temporal_signal),
    ):
        if score is None:
            continue
        lines.append(f"  {label}: {score:.3f}")
        if signal:
            lines.append(f"    {signal}")

    if alert.typology_name:
        lines += ["", f"FATF typology: {alert.typology_name} ({alert.typology_id})"]
    if alert.forensic_report:
        lines += ["", "Forensic analysis", alert.forensic_report]

    lines += ["", "— DeepSentinel"]
    return "\n".join(lines)


def build_email_html(alert: FraudAlert, backend_url: str) -> str:
    return _build_email_html_impl(alert, backend_url)


def _build_email_html(alert: FraudAlert, backend_url: str) -> str:
    return _build_email_html_impl(alert, backend_url)


def _build_email_html_impl(alert: FraudAlert, backend_url: str) -> str:
    """Build beautiful HTML email."""

    classification_color = {
        "CRITICAL": "#ef4444",
        "HIGH": "#f97316",
        "MEDIUM": "#eab308",
        "LOW": "#22c55e",
    }.get(alert.classification, "#6b7280")

    risk_icon = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
    }.get(alert.classification, "⚪")

    modality_signals = ""
    if alert.graph_score is not None:
        modality_signals += f"""
        <tr>
            <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">
                <span style="font-weight: 600; color: #1f2937;">🕸️ Graph Neural Network</span>
            </td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; text-align: right;">
                <span style="font-size: 24px; font-weight: 700; color: {classification_color};">
                    {alert.graph_score * 100:.1f}%
                </span>
            </td>
        </tr>
        """
        if alert.graph_signal:
            modality_signals += f"""
            <tr>
                <td colspan="2" style="padding: 8px 16px; border-bottom: 1px solid #e5e7eb; background: #f9fafb;">
                    <span style="font-size: 13px; color: #6b7280;">{alert.graph_signal}</span>
                </td>
            </tr>
            """

    if alert.behavioral_score is not None:
        modality_signals += f"""
        <tr>
            <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">
                <span style="font-weight: 600; color: #1f2937;">📊 Behavioral VAE</span>
            </td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; text-align: right;">
                <span style="font-size: 24px; font-weight: 700; color: {classification_color};">
                    {alert.behavioral_score * 100:.1f}%
                </span>
            </td>
        </tr>
        """
        if alert.behavioral_signal:
            modality_signals += f"""
            <tr>
                <td colspan="2" style="padding: 8px 16px; border-bottom: 1px solid #e5e7eb; background: #f9fafb;">
                    <span style="font-size: 13px; color: #6b7280;">{alert.behavioral_signal}</span>
                </td>
            </tr>
            """

    if alert.temporal_score is not None:
        modality_signals += f"""
        <tr>
            <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb;">
                <span style="font-weight: 600; color: #1f2937;">⏱️ Temporal CNN</span>
            </td>
            <td style="padding: 12px 16px; border-bottom: 1px solid #e5e7eb; text-align: right;">
                <span style="font-size: 24px; font-weight: 700; color: {classification_color};">
                    {alert.temporal_score * 100:.1f}%
                </span>
            </td>
        </tr>
        """
        if alert.temporal_signal:
            modality_signals += f"""
            <tr>
                <td colspan="2" style="padding: 8px 16px; border-bottom: 1px solid #e5e7eb; background: #f9fafb;">
                    <span style="font-size: 13px; color: #6b7280;">{alert.temporal_signal}</span>
                </td>
            </tr>
            """

    typology_section = ""
    if alert.typology_name:
        typology_section = f"""
        <div style="margin-top: 24px; padding: 16px; background: #f3f4f6; border-radius: 8px;">
            <p style="margin: 0 0 8px 0; font-size: 12px; font-weight: 600; color: #6b7280; text-transform: uppercase;">
                FATF Typology Match
            </p>
            <p style="margin: 0; font-size: 16px; font-weight: 600; color: #1f2937;">
                {alert.typology_name}
            </p>
            <p style="margin: 4px 0 0 0; font-size: 12px; color: #6b7280;">
                ID: <code style="background: white; padding: 2px 6px; border-radius: 4px; font-family: monospace;">{alert.typology_id}</code>
            </p>
        </div>
        """

    forensic_section = ""
    if alert.forensic_report:
        forensic_section = f"""
        <div style="margin-top: 24px;">
            <p style="margin: 0 0 12px 0; font-size: 12px; font-weight: 600; color: #6b7280; text-transform: uppercase;">
                LLM Forensic Analysis
            </p>
            <div style="padding: 16px; background: #f9fafb; border-left: 4px solid {classification_color}; border-radius: 4px;">
                <p style="margin: 0; font-size: 14px; line-height: 1.6; color: #374151;">
                    {alert.forensic_report}
                </p>
            </div>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f3f4f6;">
        <div style="max-width: 600px; margin: 0 auto; padding: 24px 16px;">

            <!-- Header -->
            <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px 12px 0 0; padding: 32px 24px; text-align: center;">
                <p style="margin: 0; font-size: 48px;">🚨</p>
                <h1 style="margin: 16px 0 0 0; font-size: 24px; font-weight: 700; color: white;">
                    Fraud Alert Detected
                </h1>
                <p style="margin: 8px 0 0 0; font-size: 14px; color: #cbd5e1;">
                    Transaction flagged by DeepSentinel Fusion Engine
                </p>
            </div>

            <!-- Risk Badge -->
            <div style="background: {classification_color}; padding: 24px; text-align: center; border-bottom: 1px solid #e5e7eb;">
                <p style="margin: 0; font-size: 13px; color: white; text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">
                    {risk_icon} {alert.classification} RISK
                </p>
                <p style="margin: 12px 0 0 0; font-size: 48px; font-weight: 700; color: white;">
                    {alert.fraud_confidence * 100:.0f}%
                </p>
                <p style="margin: 8px 0 0 0; font-size: 12px; color: rgba(255,255,255,0.8);">
                    Fraud Confidence Score
                </p>
            </div>

            <!-- Transaction Details -->
            <div style="background: white; padding: 24px; border-bottom: 1px solid #e5e7eb;">
                <p style="margin: 0 0 16px 0; font-size: 12px; font-weight: 600; color: #6b7280; text-transform: uppercase;">
                    Transaction Details
                </p>
                <div style="background: #f9fafb; padding: 16px; border-radius: 8px;">
                    <p style="margin: 0 0 8px 0; font-size: 13px; color: #6b7280;">
                        <span style="font-weight: 600;">Transaction ID:</span>
                        <code style="background: white; padding: 4px 8px; border-radius: 4px; font-family: monospace; color: #1f2937;">
                            {alert.transaction_id}
                        </code>
                    </p>
                    <p style="margin: 0; font-size: 13px; color: #6b7280;">
                        <span style="font-weight: 600;">Timestamp:</span> {alert.timestamp}
                    </p>
                </div>
            </div>

            <!-- Model Scores -->
            <div style="background: white; padding: 24px; border-bottom: 1px solid #e5e7eb;">
                <p style="margin: 0 0 16px 0; font-size: 12px; font-weight: 600; color: #6b7280; text-transform: uppercase;">
                    AI Model Scores
                </p>
                <table style="width: 100%; border-collapse: collapse;">
                    {modality_signals}
                </table>
            </div>

            <!-- Typology & Forensic -->
            <div style="background: white; padding: 24px; border-bottom: 1px solid #e5e7eb;">
                {typology_section}
                {forensic_section}
            </div>

            <!-- Footer -->
            <div style="background: #f3f4f6; padding: 24px; border-radius: 0 0 12px 12px; text-align: center;">
                <p style="margin: 0 0 12px 0; font-size: 12px; color: #6b7280;">
                    Review this transaction in DeepSentinel:
                </p>
                <a href="{backend_url}" style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #2563eb, #0891b2); color: white; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 13px;">
                    Open Dashboard
                </a>
                <p style="margin: 16px 0 0 0; font-size: 11px; color: #9ca3af;">
                    DeepSentinel Fraud Detection Platform<br/>
                    Multi-modal AI fusion engine with LLM forensic reports
                </p>
            </div>

        </div>
    </body>
    </html>
    """


def _send_plain(
    subject: str, body: str, recipients: list[str], reply_to: str | None = None
) -> bool:
    """Send a plain-text message through whichever provider is configured.

    Separate from send_fraud_alert because that one builds an alert-shaped
    HTML template; this is for operational mail such as enquiries, where the
    text IS the content and a Reply-To matters more than styling.
    """
    provider, s_cfg = _provider()
    if provider != "smtp" or not recipients:
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"{SENDER_NAME} <{s_cfg['username']}>"
    message["To"] = ", ".join(recipients)
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)

    try:
        context = ssl.create_default_context()
        if int(s_cfg["port"]) == 465:
            with smtplib.SMTP_SSL(
                s_cfg["host"], int(s_cfg["port"]), context=context, timeout=20
            ) as srv:
                srv.login(s_cfg["username"], s_cfg["password"])
                srv.send_message(message)
        else:
            with smtplib.SMTP(s_cfg["host"], int(s_cfg["port"]), timeout=20) as srv:
                if s_cfg.get("use_tls"):
                    srv.starttls(context=context)
                srv.login(s_cfg["username"], s_cfg["password"])
                srv.send_message(message)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Plain email send failed: {exc}")
        return False

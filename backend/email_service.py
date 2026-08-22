"""
Email service for fraud alert notifications.
Sends beautiful HTML reports to configured risk managers via SendGrid.
No passwords stored - only API key required.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

# SendGrid (recommended for production - no password needed)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "alerts@deepsentinel.io")
SENDER_NAME = os.getenv("SENDER_NAME", "DeepSentinel")

# Gmail Bot Account (optional - uses service account, no user password)
GMAIL_BOT_EMAIL = os.getenv("GMAIL_BOT_EMAIL", "")  # e.g., alerts@deepsentinel-bot.iam.gserviceaccount.com
GMAIL_SERVICE_ACCOUNT_KEY = os.getenv("GMAIL_SERVICE_ACCOUNT_KEY", "")  # Path to service account JSON


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


async def send_fraud_alert(
    alert: FraudAlert,
    recipient_emails: List[str],
    backend_url: str = "http://localhost:8000",
) -> bool:
    """Send fraud alert email via SendGrid (no password required)."""

    if not recipient_emails:
        logger.warning("No recipient emails configured")
        return False

    html_body = _build_email_html(alert, backend_url)

    # Use SendGrid (production - API key only, no password)
    if SENDGRID_API_KEY:
        return await _send_via_sendgrid(alert, recipient_emails, html_body)
    # Mock send if no credentials (for development/testing)
    else:
        logger.info(
            f"Mock email send (SendGrid not configured): {alert.transaction_id} → {recipient_emails}"
        )
        return True


async def _send_via_sendgrid(
    alert: FraudAlert, recipient_emails: List[str], html_body: str
) -> bool:
    """Send via SendGrid API."""
    try:
        payload = {
            "personalizations": [
                {"to": [{"email": email} for email in recipient_emails]}
            ],
            "from": {"email": SENDER_EMAIL, "name": SENDER_NAME},
            "subject": f"🚨 Fraud Alert: Transaction {alert.transaction_id}",
            "content": [{"type": "text/html", "value": html_body}],
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.sendgrid.com/v3/mail/send",
                json=payload,
                headers={"Authorization": f"Bearer {SENDGRID_API_KEY}"},
                timeout=10.0,
            )

            if resp.status_code in (200, 202):
                logger.info(f"Email sent: {alert.transaction_id} → {recipient_emails}")
                return True
            else:
                logger.error(f"SendGrid error {resp.status_code}: {resp.text}")
                return False

    except Exception as e:
        logger.error(f"Email send failed: {type(e).__name__}: {e}")
        return False


def build_email_html(alert: FraudAlert, backend_url: str) -> str:
    """Build beautiful HTML email (public API)."""
    return _build_email_html_impl(alert, backend_url)


def _build_email_html(alert: FraudAlert, backend_url: str) -> str:
    """Deprecated: use build_email_html instead."""
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

"""
Configuration management for risk managers, thresholds, and alert settings.
Uses JSON file for persistence (can be upgraded to database).
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

SETTINGS_FILE = Path(os.getenv("SETTINGS_FILE", "./settings.json"))


@dataclass
class RiskManager:
    """Risk manager contact info"""

    name: str
    email: str
    role: str = "Risk Manager"
    enabled: bool = True


@dataclass
class AlertSettings:
    """Fraud alert configuration"""

    fraud_threshold: float = 0.6  # Alert if confidence > this
    include_low_risk: bool = False
    include_medium_risk: bool = True
    include_high_risk: bool = True
    include_critical_risk: bool = True
    send_to_all: bool = True  # vs. role-based routing


@dataclass
class Configuration:
    """Complete system configuration"""

    risk_managers: List[RiskManager]
    alert_settings: AlertSettings
    backend_url: str = "http://localhost:8000"


def load_config() -> Configuration:
    """Load configuration from file or return defaults."""
    if not SETTINGS_FILE.exists():
        logger.info("Settings file not found, creating with defaults")
        default = Configuration(
            risk_managers=[
                RiskManager(name="Default Admin", email="admin@deepsentinel.io"),
            ],
            alert_settings=AlertSettings(),
            backend_url="http://localhost:8000",
        )
        save_config(default)
        return default

    try:
        with open(SETTINGS_FILE) as f:
            data = json.load(f)
            risk_managers = [
                RiskManager(**rm) for rm in data.get("risk_managers", [])
            ]
            alert_settings_data = data.get("alert_settings", {})
            alert_settings = AlertSettings(**alert_settings_data)
            backend_url = data.get("backend_url", "http://localhost:8000")
            return Configuration(
                risk_managers=risk_managers,
                alert_settings=alert_settings,
                backend_url=backend_url,
            )
    except Exception as e:
        logger.error(f"Failed to load config: {e}, using defaults")
        return Configuration(
            risk_managers=[
                RiskManager(name="Default Admin", email="admin@deepsentinel.io"),
            ],
            alert_settings=AlertSettings(),
        )


def save_config(config: Configuration) -> bool:
    """Save configuration to file."""
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "risk_managers": [asdict(rm) for rm in config.risk_managers],
            "alert_settings": asdict(config.alert_settings),
            "backend_url": config.backend_url,
        }
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Config saved: {SETTINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        return False


def add_risk_manager(name: str, email: str, role: str = "Risk Manager") -> bool:
    """Add a new risk manager."""
    config = load_config()
    if any(rm.email == email for rm in config.risk_managers):
        logger.warning(f"Risk manager {email} already exists")
        return False
    config.risk_managers.append(RiskManager(name=name, email=email, role=role))
    return save_config(config)


def remove_risk_manager(email: str) -> bool:
    """Remove a risk manager by email."""
    config = load_config()
    original_count = len(config.risk_managers)
    config.risk_managers = [rm for rm in config.risk_managers if rm.email != email]
    if len(config.risk_managers) < original_count:
        return save_config(config)
    logger.warning(f"Risk manager {email} not found")
    return False


def get_enabled_risk_manager_emails() -> List[str]:
    """Get list of enabled risk manager emails."""
    config = load_config()
    return [rm.email for rm in config.risk_managers if rm.enabled]


def get_all_risk_managers() -> List[RiskManager]:
    """Get all risk managers."""
    config = load_config()
    return config.risk_managers


def should_alert(classification: str) -> bool:
    """Check if should send alert for this classification."""
    config = load_config()
    alerts = config.alert_settings
    return {
        "CRITICAL": alerts.include_critical_risk,
        "HIGH": alerts.include_high_risk,
        "MEDIUM": alerts.include_medium_risk,
        "LOW": alerts.include_low_risk,
    }.get(classification, False)


def get_backend_url() -> str:
    """Get configured backend URL for emails."""
    config = load_config()
    return config.backend_url

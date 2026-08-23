"""
Centralised configuration.

Values resolve in this order, first match wins:

    1. Environment variable   (cloud deploys, secret managers, CI)
    2. config.ini             (operational settings, committed to git)
    3. Built-in default

Secrets (API keys, the JWT signing key) are declared `secret=True`. They are
read from the environment first and are never written to config.ini by us — a
bank deploys those through its secret manager, not a file on the container.
config.ini may still hold them for local development, which is why the [secrets]
section exists in config.example.ini but the real config.ini is gitignored.

Add a new setting by appending one Setting(...) to SETTINGS below. It is then
available as get("section", "key") and validated at startup.
"""

import configparser
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(os.getenv("DEEPSENTINEL_CONFIG", "./config.ini"))


@dataclass(frozen=True)
class Setting:
    section: str
    key: str
    env_var: str
    default: Any
    cast: Callable[[str], Any] = str
    secret: bool = False
    required_in_prod: bool = False
    description: str = ""


def _as_bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


# ── Setting registry ─────────────────────────────────────────────────────────
# env_var names are kept identical to the previous .env keys so existing
# deployments and docker-compose files keep working unchanged.

SETTINGS: tuple[Setting, ...] = (
    # --- paths ---
    Setting("paths", "chroma_db", "CHROMA_DB_PATH", "./chroma_store",
            description="ChromaDB vector store location"),
    Setting("paths", "fatf_data", "FATF_DATA_PATH", "./data/fatf_typologies.json",
            description="FATF typology knowledge base"),
    Setting("paths", "meta_classifier", "MODEL_SAVE_PATH", "./models/meta_classifier.joblib",
            description="Trained fusion meta-classifier"),
    Setting("paths", "users_db", "USERS_FILE", "./users.json",
            description="User database (contains password hashes)"),
    Setting("paths", "runtime_settings", "SETTINGS_FILE", "./settings.json",
            description="Risk-manager and alert configuration written at runtime"),

    # --- upstream model APIs ---
    Setting("upstream", "behavioral_api_base", "BEHAVIORAL_API_BASE", "http://localhost:8001",
            description="M1 Wijesinghe — VAE/DSAA"),
    Setting("upstream", "graph_api_base", "GRAPH_API_BASE", "http://localhost:8002",
            description="M2 Ewaduge — Edge-Enhanced GraphSAGE"),
    Setting("upstream", "temporal_api_base", "TEMPORAL_API_BASE", "http://localhost:8003",
            description="M3 Pathirana — TCN/TSCFD"),
    Setting("upstream", "timeout_ms", "UPSTREAM_TIMEOUT_MS", 5000, cast=int,
            description="Per-call timeout for upstream model inference"),

    # --- LLM ---
    Setting("llm", "provider", "LLM_PROVIDER", "gemini",
            description="gemini | ollama"),
    Setting("llm", "gemini_model", "GEMINI_MODEL", "gemini-2.0-flash"),
    Setting("llm", "ollama_base_url", "OLLAMA_BASE_URL", "http://localhost:11434"),
    Setting("llm", "ollama_model", "OLLAMA_MODEL", "llama3"),

    # --- database ---
    # SQLite default means a fresh clone runs with zero setup. Point this at a
    # Postgres URL for anything shared between machines or people.
    Setting("database", "url", "DATABASE_URL", "sqlite+aiosqlite:///./deepsentinel.db",
            secret=True,
            description="SQLAlchemy URL. Postgres for shared/cloud, SQLite for local-only"),
    Setting("database", "pool_size", "DB_POOL_SIZE", 5, cast=int,
            description="Connection pool size (Postgres only)"),
    Setting("database", "max_overflow", "DB_MAX_OVERFLOW", 10, cast=int,
            description="Connections allowed beyond pool_size (Postgres only)"),
    Setting("database", "echo_sql", "DB_ECHO_SQL", False, cast=_as_bool,
            description="Log every SQL statement — debugging only, noisy"),

    # --- auth ---
    Setting("auth", "access_token_expire_minutes", "ACCESS_TOKEN_EXPIRE_MINUTES", 480, cast=int,
            description="Session lifetime in minutes"),
    Setting("auth", "cors_origins", "CORS_ORIGINS", "*",
            description="Comma-separated allowed origins. '*' is development only"),
    Setting("auth", "max_failed_logins", "MAX_FAILED_LOGINS", 5, cast=int,
            description="Failed attempts before an account is temporarily locked"),
    Setting("auth", "lockout_minutes", "LOCKOUT_MINUTES", 15, cast=int,
            description="How long an account stays locked"),

    # --- email ---
    Setting("email", "sender_email", "SENDER_EMAIL", "alerts@deepsentinel.io"),
    Setting("email", "sender_name", "SENDER_NAME", "DeepSentinel Fraud Alerts"),
    Setting("email", "smtp_host", "SMTP_HOST", "",
            description="e.g. smtp.gmail.com. Leave blank to use SendGrid instead"),
    Setting("email", "smtp_port", "SMTP_PORT", 587, cast=int,
            description="587 for STARTTLS, 465 for implicit TLS"),
    Setting("email", "smtp_use_tls", "SMTP_USE_TLS", True, cast=_as_bool,
            description="STARTTLS on port 587; ignored on 465"),

    # --- secrets (environment first; never committed) ---
    Setting("secrets", "jwt_secret_key", "JWT_SECRET_KEY", "", secret=True,
            required_in_prod=True,
            description="JWT signing key — generate with secrets.token_urlsafe(32)"),
    Setting("secrets", "admin_bootstrap_password", "ADMIN_BOOTSTRAP_PASSWORD", "admin123",
            secret=True,
            description="Password for the admin account created on first run"),
    Setting("secrets", "gemini_api_key", "GEMINI_API_KEY", "", secret=True,
            description="Google AI Studio API key"),
    Setting("secrets", "chatbot_gemini_api_key", "CHATBOT_GEMINI_API_KEY", "", secret=True,
            description="Separate key for the chatbots, so their traffic does not "
                        "consume the quota the forensic reports need. Falls back "
                        "to gemini_api_key when unset"),
    Setting("secrets", "sendgrid_api_key", "SENDGRID_API_KEY", "", secret=True,
            description="SendGrid API key for fraud alert email"),
    Setting("secrets", "smtp_username", "SMTP_USERNAME", "", secret=True,
            description="Sending account address, e.g. alerts@yourdomain.com"),
    Setting("secrets", "smtp_password", "SMTP_PASSWORD", "", secret=True,
            description="App password for the sending account, never a personal password"),
)

_INDEX: dict[tuple[str, str], Setting] = {(s.section, s.key): s for s in SETTINGS}

_parser: Optional[configparser.ConfigParser] = None
_cache: dict[tuple[str, str], Any] = {}


def _load_parser() -> configparser.ConfigParser:
    global _parser
    if _parser is not None:
        return _parser

    parser = configparser.ConfigParser(interpolation=None)
    if CONFIG_FILE.exists():
        parser.read(CONFIG_FILE, encoding="utf-8")
        logger.info(f"Loaded configuration from {CONFIG_FILE}")
    else:
        logger.warning(
            f"{CONFIG_FILE} not found — using environment variables and defaults. "
            f"Copy config.example.ini to config.ini to customise."
        )
    _parser = parser
    return parser


def get(section: str, key: str) -> Any:
    """Resolve a setting: environment > config.ini > default."""
    cached = _cache.get((section, key))
    if cached is not None:
        return cached

    setting = _INDEX.get((section, key))
    if setting is None:
        raise KeyError(f"Unknown setting [{section}] {key} — add it to SETTINGS in config.py")

    raw = os.getenv(setting.env_var)
    source = "env"

    if raw is None or raw == "":
        parser = _load_parser()
        if parser.has_option(section, key):
            raw = parser.get(section, key)
            source = "config.ini"

    if raw is None or raw == "":
        value = setting.default
        source = "default"
    else:
        try:
            value = setting.cast(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"[{section}] {key} (from {source}) is not a valid "
                f"{setting.cast.__name__}: {raw!r}"
            ) from e

    _cache[(section, key)] = value
    if not setting.secret:
        logger.debug(f"config [{section}] {key} = {value!r} (from {source})")
    return value


def get_path(section: str, key: str) -> Path:
    return Path(get(section, key))


def reload() -> None:
    """Drop cached values so the next get() re-reads config.ini and the env."""
    global _parser
    _parser = None
    _cache.clear()


def validate(strict: bool = False) -> list[str]:
    """
    Resolve every setting and report problems.

    strict=True treats missing production-required secrets as errors. Call with
    strict=True when DEEPSENTINEL_ENV=production so a misconfigured deploy fails
    at startup rather than at the first request.
    """
    problems: list[str] = []

    for setting in SETTINGS:
        try:
            value = get(setting.section, setting.key)
        except ValueError as e:
            problems.append(str(e))
            continue

        if setting.required_in_prod and not value:
            msg = (
                f"[{setting.section}] {setting.key} is not set "
                f"(env {setting.env_var}) — {setting.description}"
            )
            if strict:
                problems.append(msg)
            else:
                logger.warning(f"{msg}. Required before deploying to production.")

    return problems


def is_production() -> bool:
    return os.getenv("DEEPSENTINEL_ENV", "development").lower() == "production"


def describe() -> str:
    """Render resolved configuration for startup logs. Secrets are masked."""
    lines = ["Resolved configuration:"]
    current_section = None
    for setting in SETTINGS:
        if setting.section != current_section:
            current_section = setting.section
            lines.append(f"  [{current_section}]")
        value = get(setting.section, setting.key)
        if setting.secret:
            shown = f"<set, {len(str(value))} chars>" if value else "<not set>"
        else:
            shown = repr(value)
        lines.append(f"    {setting.key} = {shown}")
    return "\n".join(lines)

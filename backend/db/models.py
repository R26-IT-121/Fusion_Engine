"""
Database schema.

Runs on PostgreSQL in production (Neon/RDS) and SQLite locally, from the same
model definitions — SQLAlchemy handles the dialect difference. Column types are
chosen to behave identically on both.
"""

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime | None) -> datetime | None:
    """
    Normalise a datetime read from the database to timezone-aware UTC.

    SQLite has no timezone type, so DateTime(timezone=True) round-trips as a
    naive value there while PostgreSQL returns an aware one. Comparing the two
    forms raises TypeError, so every comparison against a stored timestamp goes
    through this.
    """
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class UserRole(str, Enum):
    """
    Role-based access control.

    ADMIN         — DeepSentinel team. Full access including system configuration.
    RISK_MANAGER  — Bank risk manager. Transactions, monitoring, alerts, and
                    risk-manager recipients; not system configuration.
    ANALYST       — Bank assistant manager. Read-only.
    """

    ADMIN = "admin"
    RISK_MANAGER = "risk_manager"
    ANALYST = "analyst"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default=UserRole.ANALYST.value)

    # bcrypt hash — never the password itself
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)

    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Lockout state for brute-force protection
    failed_login_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Forces re-authentication everywhere when a password changes: tokens issued
    # before this timestamp are rejected.
    credentials_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


class RiskManager(Base):
    """A recipient of fraud alert email. Distinct from a User — a risk manager
    can receive alerts without holding a login."""

    __tablename__ = "risk_managers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(64), nullable=False, default="Risk Manager")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow
    )

    def __repr__(self) -> str:
        return f"<RiskManager {self.email}>"


class AlertSettings(Base):
    """Singleton row (id=1) holding alert thresholds."""

    __tablename__ = "alert_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    fraud_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.6)
    include_low_risk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    include_medium_risk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    include_high_risk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    include_critical_risk: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    send_to_all: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    backend_url: Mapped[str] = mapped_column(
        String(500), nullable=False, default="http://localhost:8000"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )


class AnalysisRecord(Base):
    """
    One fraud analysis. This is the transaction history the risk manager
    monitors, and the evidence trail an audit asks for.

    Account identifiers are stored because a fraud investigation needs them.
    Balances are not — they are not needed after scoring and would widen the
    blast radius of a breach.
    """

    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    # Transaction context
    tx_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    amount: Mapped[float | None] = mapped_column(Float, nullable=True)
    name_orig: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    name_dest: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Fusion result
    fraud_confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    classification: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    modalities_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Per-modality scores; null means that model was unavailable
    graph_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    behavioral_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    temporal_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    graph_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    behavioral_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    temporal_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # RAG + LLM output
    typology_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    typology_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    similarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    forensic_report: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Operational
    alert_sent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    mock_scenario: Mapped[str | None] = mapped_column(String(64), nullable=True)
    analysed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )

    __table_args__ = (
        Index("ix_analysis_created_classification", "created_at", "classification"),
    )

    def __repr__(self) -> str:
        return f"<AnalysisRecord {self.transaction_id} {self.classification}>"


class AuditLog(Base):
    """
    Append-only record of security-relevant actions.

    Never updated or deleted by application code. A compliance audit reads this
    to answer who did what, when, and from where.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utcnow, index=True
    )
    actor: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    target: Mapped[str | None] = mapped_column(String(320), nullable=True)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False, default="success")
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Context. Must never contain passwords, tokens, or full card/account data.
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<AuditLog {self.timestamp} {self.actor} {self.action} {self.outcome}>"

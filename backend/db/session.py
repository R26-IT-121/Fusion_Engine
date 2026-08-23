"""
Async engine and session management.

One code path for both backends — the URL decides:

    postgresql+asyncpg://user:pass@host/db    cloud (Neon, RDS)
    sqlite+aiosqlite:///./deepsentinel.db     local, zero setup

Neon and most managed Postgres require TLS. asyncpg does not understand the
`?sslmode=` query parameter that libpq uses, so it is translated here rather
than making every developer remember to strip it from the connection string
they copied out of the dashboard.
"""

import logging
import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend import config
from backend.db.models import Base

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

# libpq sslmode values that mean "encrypt, but do not verify the certificate"
_SSL_NO_VERIFY = {"require", "allow", "prefer"}
_SSL_VERIFY = {"verify-ca", "verify-full"}


def _normalise_url(raw: str) -> tuple[str, dict]:
    """
    Return (url, connect_args) with the driver set for async use and any
    libpq-style sslmode translated into asyncpg's ssl argument.
    """
    url = raw.strip()
    connect_args: dict = {}

    # Accept the plain URL the provider's dashboard shows and upgrade the driver
    if url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    elif url.startswith("sqlite://") and "+aiosqlite" not in url:
        url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    if "asyncpg" in url:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))

        sslmode = query.pop("sslmode", None)
        # Neon puts the endpoint id here for older clients; asyncpg rejects it
        query.pop("options", None)
        query.pop("channel_binding", None)

        if sslmode == "disable":
            connect_args["ssl"] = False
        elif sslmode in _SSL_VERIFY:
            connect_args["ssl"] = ssl.create_default_context()
        elif sslmode in _SSL_NO_VERIFY:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            connect_args["ssl"] = ctx

        # Connection poolers (Neon's -pooler endpoint, Supabase's :6543, any
        # PgBouncer in transaction mode) hand each transaction a different
        # backend connection. asyncpg prepares statements and caches them by
        # name against the connection it prepared them on, so a cached name
        # resolves on a backend that never saw the PREPARE — surfacing as
        # intermittent "prepared statement _asyncpg_stmt_N does not exist"
        # under concurrency. Disabling the cache is the supported fix.
        if "-pooler." in parts.netloc or parts.port == 6543:
            connect_args["statement_cache_size"] = 0
            logger.info(
                "Pooled endpoint detected — disabled asyncpg statement cache."
            )

        url = urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )

    return url, connect_args


def _redact(url: str) -> str:
    """Strip credentials so a connection string can be logged."""
    parts = urlsplit(url)
    if "@" in parts.netloc:
        host = parts.netloc.rsplit("@", 1)[1]
        netloc = f"***@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    return url


def get_engine() -> AsyncEngine:
    global _engine, _session_factory
    if _engine is not None:
        return _engine

    raw = config.get("database", "url")
    url, connect_args = _normalise_url(raw)
    is_sqlite = url.startswith("sqlite")

    kwargs: dict = {
        "echo": bool(config.get("database", "echo_sql")),
        "connect_args": connect_args,
        "pool_pre_ping": True,  # drop connections a serverless DB closed while idle
    }
    if not is_sqlite:
        kwargs["pool_size"] = config.get("database", "pool_size")
        kwargs["max_overflow"] = config.get("database", "max_overflow")
        kwargs["pool_recycle"] = 300  # Neon closes idle connections; stay under it

    _engine = create_async_engine(url, **kwargs)
    _session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )

    logger.info(
        f"Database engine created: {_redact(url)} "
        f"({'SQLite (local)' if is_sqlite else 'PostgreSQL'})"
    )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    Session scope. Commits on clean exit, rolls back on exception.

        async with get_session() as db:
            db.add(user)
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """
    Create tables that do not exist yet, then seed the singleton settings row.

    create_all only adds missing tables — it never alters an existing one. Once
    the schema is in production, column changes go through Alembic
    (`poetry install --extras postgres`), not by editing models and restarting.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema ready.")

    from sqlalchemy import select

    from backend.db.models import AlertSettings

    async with get_session() as db:
        existing = await db.scalar(select(AlertSettings).where(AlertSettings.id == 1))
        if existing is None:
            db.add(AlertSettings(id=1))
            logger.info("Seeded default alert settings.")


async def close_db() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed.")


async def healthcheck() -> bool:
    """True if the database answers a trivial query."""
    from sqlalchemy import text

    try:
        async with get_session() as db:
            await db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database healthcheck failed: {type(e).__name__}: {e}")
        return False

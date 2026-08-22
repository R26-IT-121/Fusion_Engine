"""
One-shot import of the legacy JSON files into the database.

Run once after configuring DATABASE_URL:

    python -m backend.db.migrate_from_json

Safe to re-run: existing usernames and email addresses are skipped, never
overwritten. The JSON files are left on disk — delete them yourself once you
have confirmed the import.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select

from backend import config
from backend.db.models import AlertSettings, RiskManager, User
from backend.db.session import close_db, get_session, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("migrate")


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


async def migrate_users(path: Path) -> tuple[int, int]:
    if not path.exists():
        logger.info(f"No {path} — skipping users.")
        return 0, 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Could not read {path}: {e}")
        return 0, 0

    imported = skipped = 0
    async with get_session() as db:
        for username, record in data.items():
            email = str(record.get("email", "")).lower()
            clash = await db.scalar(
                select(User).where(
                    (User.username == username) | (func.lower(User.email) == email)
                )
            )
            if clash is not None:
                logger.info(f"  skip {username} — already present")
                skipped += 1
                continue

            # The stored value is already a bcrypt hash; carry it across so
            # existing passwords keep working.
            db.add(
                User(
                    username=username,
                    email=email,
                    full_name=record.get("full_name", username),
                    role=record.get("role", "analyst"),
                    hashed_password=record["hashed_password"],
                    enabled=bool(record.get("enabled", True)),
                    created_at=_parse_dt(record.get("created_at")) or datetime.now(timezone.utc),
                    last_login=_parse_dt(record.get("last_login")),
                )
            )
            logger.info(f"  import {username} ({record.get('role')})")
            imported += 1

    return imported, skipped


async def migrate_settings(path: Path) -> tuple[int, int]:
    if not path.exists():
        logger.info(f"No {path} — skipping settings.")
        return 0, 0

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Could not read {path}: {e}")
        return 0, 0

    imported = skipped = 0
    async with get_session() as db:
        for record in data.get("risk_managers", []):
            email = str(record.get("email", "")).lower()
            if not email:
                continue
            existing = await db.scalar(
                select(RiskManager).where(func.lower(RiskManager.email) == email)
            )
            if existing is not None:
                logger.info(f"  skip {email} — already present")
                skipped += 1
                continue

            db.add(
                RiskManager(
                    name=record.get("name", email),
                    email=email,
                    role=record.get("role", "Risk Manager"),
                    enabled=bool(record.get("enabled", True)),
                )
            )
            logger.info(f"  import {email}")
            imported += 1

        alerts = data.get("alert_settings") or {}
        if alerts:
            settings = await db.scalar(select(AlertSettings).where(AlertSettings.id == 1))
            if settings is None:
                settings = AlertSettings(id=1)
                db.add(settings)
            for key, value in alerts.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
            if data.get("backend_url"):
                settings.backend_url = data["backend_url"]
            logger.info("  imported alert settings")

    return imported, skipped


async def main() -> int:
    url = config.get("database", "url")
    target = "SQLite (local only)" if url.startswith("sqlite") else "PostgreSQL"
    logger.info(f"Target: {target}")

    if url.startswith("sqlite"):
        logger.warning(
            "DATABASE_URL points at SQLite. Data will stay on this machine only. "
            "Set a Postgres URL for shared multi-device access."
        )

    await init_db()

    logger.info("Users:")
    u_imported, u_skipped = await migrate_users(config.get_path("paths", "users_db"))

    logger.info("Risk managers and alert settings:")
    r_imported, r_skipped = await migrate_settings(config.get_path("paths", "runtime_settings"))

    await close_db()

    logger.info(
        f"Done. Users: {u_imported} imported, {u_skipped} skipped. "
        f"Risk managers: {r_imported} imported, {r_skipped} skipped."
    )
    logger.info(
        "The JSON files were left in place. Delete them once you have verified "
        "the import — they are no longer read."
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

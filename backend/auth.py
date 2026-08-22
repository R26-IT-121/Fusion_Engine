"""
Authentication and role-based access control.

Users live in the database, so an account created on one machine works from
every machine — which JSON file storage could not do.

Roles:
    ADMIN         DeepSentinel team. Everything, including system configuration.
    RISK_MANAGER  Bank risk manager. Transactions, monitoring, alerts, and alert
                  recipients. Not system configuration.
    ANALYST       Bank assistant manager. Read-only.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, Header, HTTPException, Request
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import func, select

from backend import config
from backend.db.models import AuditLog, User, UserRole, as_utc
from backend.db.session import get_session

logger = logging.getLogger(__name__)

# --- Security configuration ---
SECRET_KEY = config.get("secrets", "jwt_secret_key")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_urlsafe(32)
    logger.warning(
        "No JWT signing key configured — generated an ephemeral one. Sessions "
        "will be invalidated on restart. Set JWT_SECRET_KEY (or [secrets] "
        "jwt_secret_key in config.ini) before deploying."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = config.get("auth", "access_token_expire_minutes")
MAX_FAILED_LOGINS = config.get("auth", "max_failed_logins")
LOCKOUT_MINUTES = config.get("auth", "lockout_minutes")

# bcrypt truncates silently past 72 bytes — reject rather than accept a password
# whose tail is ignored.
MAX_PASSWORD_BYTES = 72
MIN_PASSWORD_LENGTH = 8


# ── Schemas ──────────────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    password: str
    role: UserRole = UserRole.ANALYST

    @field_validator("username")
    @classmethod
    def _username_ok(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("username must be at least 3 characters")
        if not v.replace("_", "").replace("-", "").replace(".", "").isalnum():
            raise ValueError("username may contain letters, digits, . _ - only")
        return v

    @field_validator("password")
    @classmethod
    def _password_ok(cls, v: str) -> str:
        if len(v) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
        if len(v.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordChange(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def _password_ok(cls, v: str) -> str:
        if len(v) < MIN_PASSWORD_LENGTH:
            raise ValueError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")
        if len(v.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
        return v


class UserOut(BaseModel):
    username: str
    email: str
    full_name: str
    role: str
    enabled: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    @classmethod
    def from_model(cls, u: User) -> "UserOut":
        return cls(
            username=u.username,
            email=u.email,
            full_name=u.full_name,
            role=u.role,
            enabled=u.enabled,
            created_at=u.created_at,
            last_login=u.last_login,
        )


# ── Password hashing ─────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password. Returns False for a malformed hash rather than raising."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ── Tokens ───────────────────────────────────────────────────────────────────


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.username,
        "role": user.role,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired, please sign in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Audit ────────────────────────────────────────────────────────────────────


async def audit(
    action: str,
    actor: Optional[str] = None,
    target: Optional[str] = None,
    outcome: str = "success",
    detail: Optional[str] = None,
    client_ip: Optional[str] = None,
) -> None:
    """Append to the audit log. Never raises — a logging failure must not break
    the request it is recording."""
    try:
        async with get_session() as db:
            db.add(
                AuditLog(
                    actor=actor,
                    action=action,
                    target=target,
                    outcome=outcome,
                    detail=detail,
                    client_ip=client_ip,
                )
            )
    except Exception as e:
        logger.error(f"Audit write failed for {action}: {type(e).__name__}: {e}")


# ── User operations ──────────────────────────────────────────────────────────


async def get_user(username: str) -> Optional[User]:
    async with get_session() as db:
        return await db.scalar(select(User).where(User.username == username))


async def list_users() -> list[User]:
    async with get_session() as db:
        result = await db.scalars(select(User).order_by(User.created_at))
        return list(result)


async def create_user(data: UserCreate, created_by: Optional[str] = None) -> User:
    async with get_session() as db:
        clash = await db.scalar(
            select(User).where(
                (User.username == data.username)
                | (func.lower(User.email) == data.email.lower())
            )
        )
        if clash is not None:
            field = "username" if clash.username == data.username else "email"
            raise HTTPException(status_code=409, detail=f"That {field} is already registered")

        user = User(
            username=data.username,
            email=data.email.lower(),
            full_name=data.full_name,
            role=data.role.value,
            hashed_password=hash_password(data.password),
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)

    await audit("user.create", actor=created_by, target=data.username,
                detail=f"role={data.role.value}")
    logger.info(f"User created: {data.username} ({data.role.value}) by {created_by}")
    return user


async def authenticate_user(
    username: str, password: str, client_ip: Optional[str] = None
) -> User:
    """
    Verify credentials. Raises HTTPException on any failure.

    Failure messages are deliberately identical for unknown user and wrong
    password so the endpoint cannot be used to enumerate valid usernames.
    """
    now = datetime.now(timezone.utc)

    # The outcome is decided inside the session but raised after it closes.
    # Raising inside would roll the session back, discarding the failed-attempt
    # counter we just incremented — which would silently disable lockout.
    failure: Optional[HTTPException] = None
    audit_outcome = "success"
    audit_detail: Optional[str] = None
    authenticated: Optional[User] = None

    async with get_session() as db:
        user = await db.scalar(select(User).where(User.username == username))

        if user is None:
            # Spend comparable time hashing so response timing does not reveal
            # whether the username exists.
            verify_password(password, "$2b$12$" + "x" * 53)
            failure = HTTPException(
                status_code=401, detail="Invalid username or password"
            )
            audit_outcome, audit_detail = "failure", "unknown user"

        elif (locked := as_utc(user.locked_until)) is not None and locked > now:
            remaining = int((locked - now).total_seconds() // 60) + 1
            failure = HTTPException(
                status_code=423,
                detail=f"Account locked after too many failed attempts. "
                       f"Try again in {remaining} minute(s).",
            )
            audit_outcome, audit_detail = "blocked", "account locked"

        elif not user.enabled:
            failure = HTTPException(status_code=403, detail="This account is disabled")
            audit_outcome, audit_detail = "blocked", "account disabled"

        elif not verify_password(password, user.hashed_password):
            user.failed_login_count += 1
            if user.failed_login_count >= MAX_FAILED_LOGINS:
                user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
                user.failed_login_count = 0
                logger.warning(
                    f"Account locked for {LOCKOUT_MINUTES} minutes after "
                    f"{MAX_FAILED_LOGINS} failed attempts: {username}"
                )
            failure = HTTPException(
                status_code=401, detail="Invalid username or password"
            )
            audit_outcome, audit_detail = "failure", "bad password"

        else:
            user.failed_login_count = 0
            user.locked_until = None
            user.last_login = now
            await db.flush()
            await db.refresh(user)
            authenticated = user

    # Session has committed; safe to record and raise.
    await audit("auth.login", actor=username, outcome=audit_outcome,
                detail=audit_detail, client_ip=client_ip)

    if failure is not None:
        raise failure

    assert authenticated is not None
    return authenticated


async def change_password(username: str, current: str, new: str) -> None:
    # As in authenticate_user: decide inside the session, raise after it closes,
    # so a rejection cannot roll back state we meant to keep.
    failure: Optional[HTTPException] = None

    async with get_session() as db:
        user = await db.scalar(select(User).where(User.username == username))
        if user is None:
            failure = HTTPException(status_code=404, detail="User not found")
        elif not verify_password(current, user.hashed_password):
            failure = HTTPException(
                status_code=401, detail="Current password is incorrect"
            )
        elif verify_password(new, user.hashed_password):
            failure = HTTPException(
                status_code=422,
                detail="New password must be different from the current one",
            )
        else:
            user.hashed_password = hash_password(new)
            # Invalidate every token issued before this moment
            user.credentials_changed_at = datetime.now(timezone.utc)

    await audit(
        "auth.password_change",
        actor=username,
        outcome="failure" if failure else "success",
    )

    if failure is not None:
        raise failure

    logger.info(f"Password changed: {username}")


async def set_user_enabled(username: str, enabled: bool, actor: str) -> None:
    async with get_session() as db:
        user = await db.scalar(select(User).where(User.username == username))
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        user.enabled = enabled
        if not enabled:
            user.credentials_changed_at = datetime.now(timezone.utc)

    await audit("user.enable" if enabled else "user.disable", actor=actor, target=username)


async def delete_user(username: str, actor: str) -> None:
    async with get_session() as db:
        user = await db.scalar(select(User).where(User.username == username))
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        if user.role == UserRole.ADMIN.value:
            remaining = await db.scalar(
                select(func.count()).select_from(User).where(
                    User.role == UserRole.ADMIN.value, User.enabled.is_(True)
                )
            )
            if remaining is not None and remaining <= 1:
                raise HTTPException(
                    status_code=409,
                    detail="Cannot delete the last admin — the system would become "
                           "unconfigurable. Create another admin first.",
                )
        await db.delete(user)

    await audit("user.delete", actor=actor, target=username)
    logger.info(f"User deleted: {username} by {actor}")


# ── Request dependencies ─────────────────────────────────────────────────────


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
) -> User:
    """Resolve the caller from the Authorization header. Use as a dependency."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = await get_user(username)
    if user is None or not user.enabled:
        raise HTTPException(status_code=401, detail="Account not found or disabled")

    # Reject tokens issued before the last credential change.
    #
    # JWT `iat` is a whole-second integer, so a token minted at 12:00:00.900
    # carries iat=12:00:00. Comparing that directly against a sub-second
    # credentials_changed_at would reject tokens issued moments after the
    # change — including the one just handed out at login. Allow one second of
    # slack, which is the truncation error and nothing more.
    issued_at = payload.get("iat")
    changed = as_utc(user.credentials_changed_at)
    if issued_at is not None and changed is not None:
        issued = datetime.fromtimestamp(issued_at, tz=timezone.utc)
        if issued < changed - timedelta(seconds=1):
            raise HTTPException(
                status_code=401, detail="Credentials changed, please sign in again"
            )

    request.state.username = user.username
    return user


def require_role(*allowed: UserRole):
    """
    Dependency factory restricting an endpoint to specific roles.

        @app.post("/settings", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """
    allowed_values = {r.value for r in allowed}

    async def _guard(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_values:
            await audit(
                "authz.denied",
                actor=user.username,
                outcome="denied",
                detail=f"role={user.role} needs one of {sorted(allowed_values)}",
            )
            raise HTTPException(
                status_code=403,
                detail="Your role does not have access to this action",
            )
        return user

    return _guard


# Common guards. Roles are cumulative in capability, not hierarchy — list each.
require_admin = require_role(UserRole.ADMIN)
require_manager = require_role(UserRole.ADMIN, UserRole.RISK_MANAGER)
require_any_user = require_role(UserRole.ADMIN, UserRole.RISK_MANAGER, UserRole.ANALYST)


# ── Bootstrap ────────────────────────────────────────────────────────────────


async def ensure_bootstrap_admin() -> None:
    """Create the initial admin on an empty user table. Idempotent."""
    async with get_session() as db:
        count = await db.scalar(select(func.count()).select_from(User))
        if count:
            return

    password = config.get("secrets", "admin_bootstrap_password")
    await create_user(
        UserCreate(
            username="admin",
            email="admin@deepsentinel.io",
            full_name="DeepSentinel Admin",
            password=password,
            role=UserRole.ADMIN,
        ),
        created_by="system",
    )
    logger.warning(
        "Bootstrapped the initial admin account. Change its password before "
        "this instance is reachable by anyone else."
    )

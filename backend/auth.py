"""
User authentication and role-based access control (RBAC).
Secure user database with JWT tokens and password hashing.
"""

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import bcrypt
import jwt
from fastapi import HTTPException, Header
from pydantic import BaseModel, EmailStr

from backend import config

logger = logging.getLogger(__name__)

# --- Security configuration ---
# The JWT signing key MUST be set in production. A random key is generated if
# absent, which invalidates all tokens on restart — fine for dev, fatal for prod.
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
USERS_FILE = config.get_path("paths", "users_db")


class UserRole(str, Enum):
    """Role-based access control."""

    ADMIN = "admin"  # Full system access (DeepSentinel team)
    RISK_MANAGER = "risk_manager"  # Monitor & manage alerts (Bank)
    ANALYST = "analyst"  # View transactions & reports (Bank)


class User(BaseModel):
    """User model."""

    username: str
    email: EmailStr
    full_name: str
    role: UserRole
    hashed_password: str
    enabled: bool = True
    created_at: str = None
    last_login: Optional[str] = None


class UserCreate(BaseModel):
    """User creation request."""

    username: str
    email: EmailStr
    full_name: str
    password: str
    role: UserRole = UserRole.ANALYST


class LoginRequest(BaseModel):
    """Login request."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
    user: dict


def hash_password(password: str) -> str:
    """Hash password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash. Returns False on any malformed hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"), hashed_password.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def create_access_token(username: str, role: UserRole) -> str:
    """Create JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


def verify_token(token: str) -> dict:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": role}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def load_users() -> dict:
    """Load users from file."""
    if not USERS_FILE.exists():
        # Create default admin user
        bootstrap_password = config.get("secrets", "admin_bootstrap_password")
        default_admin = User(
            username="admin",
            email="admin@deepsentinel.io",
            full_name="DeepSentinel Admin",
            role=UserRole.ADMIN,
            hashed_password=hash_password(bootstrap_password),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        users = {default_admin.username: default_admin.model_dump()}
        save_users(users)
        logger.warning(
            "Bootstrapped default admin user. Change this password before deploying."
        )
        return users

    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load users: {e}")
        return {}


def save_users(users: dict) -> bool:
    """Save users to file."""
    try:
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save users: {e}")
        return False


def create_user(user_create: UserCreate) -> User:
    """Create new user."""
    users = load_users()

    if user_create.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_pw = hash_password(user_create.password)
    new_user = User(
        username=user_create.username,
        email=user_create.email,
        full_name=user_create.full_name,
        role=user_create.role,
        hashed_password=hashed_pw,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    users[new_user.username] = new_user.model_dump()
    save_users(users)
    logger.info(f"Created user: {new_user.username} ({new_user.role})")
    return new_user


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user by username and password."""
    users = load_users()
    user_data = users.get(username)

    if not user_data:
        return None

    user = User(**user_data)
    if not verify_password(password, user.hashed_password):
        return None

    # Update last login
    user.last_login = datetime.now(timezone.utc).isoformat()
    users[username] = user.model_dump()
    save_users(users)

    return user


def get_user(username: str) -> Optional[User]:
    """Get user by username."""
    users = load_users()
    user_data = users.get(username)
    return User(**user_data) if user_data else None


def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    """Get current user from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    payload = verify_token(token)
    user = get_user(payload["username"])

    if not user or not user.enabled:
        raise HTTPException(status_code=401, detail="User not found or disabled")

    return user


def require_role(*allowed_roles: UserRole):
    """Decorator to require specific role."""

    def decorator(func):
        async def wrapper(*args, current_user: User = None, **kwargs):
            if current_user.role not in allowed_roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Requires role: {', '.join(allowed_roles)}",
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator

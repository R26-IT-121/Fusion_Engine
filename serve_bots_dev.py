"""Development launcher for the two assistants — no RAG stack required.

The full application (`backend.main`) loads the FATF vector store at startup,
which needs chromadb and sentence-transformers. Neither assistant uses that
store, so this launcher brings up only what they do need — auth, the database
and the two routers — letting you demo both without installing the heavy
dependencies.

**Development only.** The real deployment runs `backend.main:app`, where both
routers are already mounted.

Usage:
    pip install fastapi uvicorn pydantic sqlalchemy aiosqlite bcrypt pyjwt \
                email-validator python-multipart
    GRAPH_API_BASE=http://localhost:8000 python serve_bots_dev.py

Then sign in with the bootstrap admin (default admin / admin123, or whatever
ADMIN_BOOTSTRAP_PASSWORD is set to) and open /assistant in the web app.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("bots-dev")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.auth import ensure_bootstrap_admin
    from backend.db.session import init_db

    await init_db()
    await ensure_bootstrap_admin()
    logger.info("Database ready; bootstrap admin ensured.")

    # Turn the assistant on so it is visible out of the box in development.
    # In production this stays off until an admin enables it.
    from assistant import entitlement

    if not entitlement.load().enabled:
        entitlement.save({"enabled": True})
        logger.info("Operator assistant enabled for development.")
    yield


app = FastAPI(title="DeepSentinel Assistants (dev)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

from assistant import router as assistant_router  # noqa: E402
from chatbot import router as chatbot_router  # noqa: E402

app.include_router(chatbot_router)
app.include_router(assistant_router)


# The real app defines auth inline in backend/main.py rather than as a router,
# so the two endpoints the web client needs to sign in are re-declared here
# using the same underlying functions — no duplicated auth logic.
from fastapi import Depends, Request  # noqa: E402

from backend.auth import LoginRequest, UserOut, get_current_user  # noqa: E402
from backend.db.models import User  # noqa: E402


@app.post("/auth/login", tags=["auth"])
async def login(req: LoginRequest, request: Request) -> dict:
    from backend.auth import authenticate_user, create_access_token

    client_ip = request.client.host if request.client else None
    user = await authenticate_user(req.username, req.password, client_ip=client_ip)
    return {
        "access_token": create_access_token(user),
        "token_type": "bearer",
        "expires_in": 8 * 3600,
        "user": {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    }


@app.get("/auth/me", tags=["auth"])
async def me(user: User = Depends(get_current_user)):
    return UserOut.from_model(user)


@app.post("/auth/logout", tags=["auth"])
async def logout(_: User = Depends(get_current_user)) -> dict:
    return {"status": "logged_out"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "assistants-dev"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8090")))

"""Routes for the operator assistant.

Access is decided in `entitlement.py`; this module only enforces it. Every
question is audited with the tools it invoked, because the assistant can read
customer transaction history and "who asked what about which account" is the
first thing an auditor wants.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from assistant import entitlement
from assistant.agent import Agent
from assistant.tools import available_tools
from backend.auth import audit, get_current_user, require_admin
from backend.db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/assistant", tags=["assistant"])


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class SettingsPatch(BaseModel):
    enabled: bool | None = None
    allowed_roles: list[str] | None = None
    max_steps: int | None = None
    allow_live_analysis: bool | None = None


def _llm():
    """Shared LLM backend; None when unconfigured so the agent can degrade."""
    from chatbot.llm import get_llm_backend

    backend, error = get_llm_backend()
    if backend is None and error:
        logger.info(f"Assistant running without an LLM: {error}")
    return backend


@router.get("/capabilities")
async def capabilities(user: User = Depends(get_current_user)) -> dict:
    """What this user may do — drives whether the UI renders the assistant."""
    status = entitlement.status_for(user.role)
    settings = entitlement.load()
    return {
        **status,
        "tools": [
            {"name": t.name, "description": t.description}
            for t in available_tools(settings.allow_live_analysis).values()
        ] if status["available"] else [],
        "llm_configured": _llm() is not None,
    }


@router.post("")
async def ask(req: AskRequest, user: User = Depends(get_current_user)) -> dict:
    settings = entitlement.require_entitled(user.role)

    agent = Agent(_llm(), settings)
    result = await agent.run(
        req.question, [t.model_dump() for t in req.history]
    )

    await audit(
        "assistant.query",
        actor=user.username,
        outcome="success",
        detail=f"tools={[s.tool for s in result.steps]} q={req.question[:120]}",
    )

    return {
        "answer": result.answer,
        "steps": [s.to_dict() for s in result.steps],
        "used_llm": result.used_llm,
        "truncated": result.truncated,
    }


@router.get("/settings")
async def get_settings(_: User = Depends(require_admin)) -> dict:
    return entitlement.load().to_dict()


@router.patch("/settings")
async def patch_settings(
    patch: SettingsPatch, user: User = Depends(require_admin)
) -> dict:
    changes = patch.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(422, "No settings supplied")
    updated = entitlement.save(changes)
    await audit(
        "assistant.settings",
        actor=user.username,
        outcome="success",
        detail=str(changes),
    )
    return updated.to_dict()

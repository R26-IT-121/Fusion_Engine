"""FastAPI routes for the project assistant.

Mount into the fusion-engine app with one line:

    from chatbot import router as chatbot_router
    app.include_router(chatbot_router)

The service is built lazily on first use so importing this module never blocks
application startup, and a corpus problem surfaces as a 503 on /api/chat rather
than a boot failure that takes the whole API down.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chatbot.service import ChatService

router = APIRouter(prefix="/api/chat", tags=["assistant"])

_service: ChatService | None = None
_init_error: str | None = None


def get_service() -> ChatService:
    global _service, _init_error
    if _service is None and _init_error is None:
        try:
            _service = ChatService()
        except Exception as exc:                      # noqa: BLE001
            _init_error = f"{type(exc).__name__}: {exc}"
    if _service is None:
        raise HTTPException(503, f"Assistant unavailable: {_init_error}")
    return _service


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool
    confident: bool


@router.post("", response_model=ChatResponse)
def ask(req: ChatRequest) -> ChatResponse:
    svc = get_service()
    result = svc.ask(req.question, [t.model_dump() for t in req.history])
    return ChatResponse(
        answer=result.text,
        sources=result.sources,
        grounded=result.grounded,
        confident=result.confident,
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok", **get_service().stats}


@router.get("/suggestions")
def suggestions() -> dict:
    """Starter questions — these double as a demo script for the panel."""
    return {
        "suggestions": [
            "What is DeepSentinel and how do the four components fit together?",
            "What F1 does the GraphSAGE model achieve and how was it measured?",
            "Why did the results drop after the leakage fix?",
            "Which novelty actually improves accuracy?",
            "How was the decision threshold chosen?",
            "What does the suspicious subgraph payload contain?",
            "What does NOT_APPLICABLE mean in the API response?",
        ]
    }

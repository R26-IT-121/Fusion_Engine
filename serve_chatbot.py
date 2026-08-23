"""Standalone launcher for the project assistant.

The fusion-engine backend source is not in this checkout yet, so this runs the
assistant on its own. When the backend lands, delete this file and mount the
router instead:

    from chatbot import router as chatbot_router
    app.include_router(chatbot_router)

Usage:
    pip install fastapi uvicorn pydantic          # google-generativeai optional
    export GEMINI_API_KEY=...                     # optional; without it the
                                                  # assistant answers extractively
    python serve_chatbot.py
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatbot import router as chatbot_router

app = FastAPI(title="DeepSentinel Project Assistant", version="1.0.0")

# The React dev server runs on another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(chatbot_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "chatbot-standalone"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("CHATBOT_HOST", "0.0.0.0"),
        port=int(os.getenv("CHATBOT_PORT", "8100")),
    )

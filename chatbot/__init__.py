"""DeepSentinel project assistant — grounded Q&A over the project's own docs."""

from chatbot.router import router
from chatbot.service import ChatService

__all__ = ["router", "ChatService"]

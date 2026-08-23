"""Operator assistant — a tool-using agent over the live DeepSentinel platform.

Distinct from `chatbot/`, which answers questions *about* the research from
documentation and is public. This assistant *operates* the platform: it runs
analyses, inspects fraud rings, and queries history. It is therefore gated —
disabled by default, admin-enabled, and restricted to entitled roles.
"""

from assistant.router import router

__all__ = ["router"]

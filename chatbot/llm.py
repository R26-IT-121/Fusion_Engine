"""LLM access for the two assistants.

Prefers the fusion engine's own factory so a deployment has a single LLM
configuration. That factory currently lives in `backend.llm.forensic_reporter`,
which imports the FATF RAG chain and therefore needs chromadb and
sentence-transformers — dependencies neither assistant uses. When those are not
installed the import fails, so we fall back to a minimal adapter that reads the
*same* config keys and speaks to the same providers.

Either way the object returned exposes `.generate(package)` where `package` has
`.system_prompt` and `.user_prompt`, so callers cannot tell the difference.
"""

from __future__ import annotations

import logging
import os

from backend import config

logger = logging.getLogger(__name__)

# The assistants may run on their own credentials, separate from the forensic
# reporter's. Two keys means chatbot traffic cannot exhaust the quota the
# report generator depends on, and each feature can be costed independently.
# When these are unset the assistants share the fusion engine's configuration.
CHATBOT_KEY_ENV = "CHATBOT_GEMINI_API_KEY"
CHATBOT_MODEL_ENV = "CHATBOT_GEMINI_MODEL"


class _GeminiBackend:
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(model_name=model)
        self.model_name = model

    def generate(self, package) -> str:
        resp = self._model.generate_content(
            f"{package.system_prompt}\n\n{package.user_prompt}",
            generation_config=self._genai.GenerationConfig(
                temperature=0.2,          # grounded, not creative
                max_output_tokens=1024,
            ),
        )
        return getattr(resp, "text", "") or ""


class _OllamaBackend:
    def __init__(self, base_url: str, model: str):
        import httpx

        self._client = httpx.Client(timeout=120.0)
        self._url = base_url.rstrip("/")
        self.model_name = model

    def generate(self, package) -> str:
        resp = self._client.post(
            f"{self._url}/api/chat",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": package.system_prompt},
                    {"role": "user", "content": package.user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _dedicated_key() -> str:
    """
    The chatbot's own API key, if one is set.

    Environment first, then config.ini. Everything else on this project is
    configured through config.ini, so requiring an environment variable for
    this one value would be an odd exception — and a separate key genuinely
    matters here, since chatbot traffic would otherwise consume the same daily
    quota the forensic reports depend on.

    Falls back to the shared key when neither is set.
    """
    from_env = (os.getenv(CHATBOT_KEY_ENV) or "").strip()
    if from_env:
        return from_env
    try:
        return str(config.get("secrets", "chatbot_gemini_api_key") or "").strip()
    except KeyError:
        return ""


def _fallback_backend():
    """Same providers and config keys as the fusion engine's factory."""
    provider = str(config.get("llm", "provider")).lower()

    if provider == "gemini":
        api_key = _dedicated_key() or str(config.get("secrets", "gemini_api_key") or "")
        if not api_key or api_key.startswith("your_"):
            raise ValueError(
                f"No Gemini API key configured. Set {CHATBOT_KEY_ENV} for the "
                "assistants specifically, or GEMINI_API_KEY to share the "
                "fusion engine's key, or switch to [llm] provider = ollama."
            )
        model = (
            os.getenv(CHATBOT_MODEL_ENV)
            or str(config.get("llm", "gemini_model"))
        )
        return _GeminiBackend(api_key, model)

    if provider == "ollama":
        return _OllamaBackend(
            str(config.get("llm", "ollama_base_url")),
            str(config.get("llm", "ollama_model")),
        )

    raise ValueError(f"Unknown LLM provider: '{provider}'. Choose 'gemini' or 'ollama'.")


def get_llm_backend() -> tuple[object | None, str | None]:
    """Return (backend, error). A None backend means answer without an LLM.

    A dedicated assistant key takes precedence: the fusion engine's factory
    would use the reporter's credentials, which is exactly the coupling the
    separate key exists to avoid.
    """
    if _dedicated_key():
        try:
            return _fallback_backend(), None
        except Exception as exc:                       # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    try:
        from backend.llm.forensic_reporter import create_llm_backend

        return create_llm_backend(), None
    except ImportError as exc:
        logger.info(f"Fusion engine LLM factory unavailable ({exc}); using fallback")
    except Exception as exc:                           # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

    try:
        return _fallback_backend(), None
    except Exception as exc:                           # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"

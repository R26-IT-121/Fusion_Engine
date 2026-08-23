"""
Forensic Reporter — LLM Integration
Supports Gemini API (default) and Ollama (local/offline).
Controlled by [llm] provider in config.ini, or the LLM_PROVIDER env var.
"""

import logging
from typing import Protocol

from backend import config
from backend.rag.prompt_builder import ForensicPromptPackage

logger = logging.getLogger(__name__)


class LLMBackend(Protocol):
    def generate(self, package: ForensicPromptPackage) -> str:
        ...


class GeminiBackend:
    """
    Gemini via the google-genai SDK.

    Replaces google-generativeai, which Google has discontinued — it warned on
    every import and no longer receives fixes.

    The system prompt is passed as a real system_instruction rather than
    concatenated into the user turn. That matters here: the Chain of Evidence
    constraints are the mechanism this project is testing, and instructions
    carry more weight in that channel than buried at the top of user content.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        logger.info(f"Gemini backend initialised: {model}")

    def generate(self, package: ForensicPromptPackage) -> str:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=package.user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=package.system_prompt,
                # Near-deterministic: a forensic report should not vary run to
                # run, and the ablation measurement depends on that stability.
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )

        text = response.text
        if not text:
            # An empty body usually means a safety block or a token ceiling hit
            # mid-generation. Silently returning "" would look like a model that
            # had nothing to say.
            reason = getattr(
                getattr(response, "candidates", [None])[0], "finish_reason", None
            )
            raise RuntimeError(
                f"Gemini returned no text (finish_reason={reason}). "
                f"Usually a safety block or the output limit being reached."
            )
        return text


class OllamaBackend:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        import httpx

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=120.0)
        logger.info(f"Ollama backend initialized: {model} @ {base_url}")

    def generate(self, package: ForensicPromptPackage) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": package.system_prompt},
                {"role": "user", "content": package.user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
        }
        response = self._client.post(
            f"{self._base_url}/api/chat", json=payload
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def create_llm_backend() -> LLMBackend:
    provider = str(config.get("llm", "provider")).lower()

    if provider == "gemini":
        api_key = config.get("secrets", "gemini_api_key")
        if not api_key or api_key.startswith("your_"):
            raise ValueError(
                "No Gemini API key configured. Set GEMINI_API_KEY, or "
                "[secrets] gemini_api_key in config.ini, or switch to "
                "[llm] provider = ollama."
            )
        model = config.get("llm", "gemini_model")
        return GeminiBackend(api_key=api_key, model=model)

    if provider == "ollama":
        base_url = config.get("llm", "ollama_base_url")
        model = config.get("llm", "ollama_model")
        return OllamaBackend(base_url=base_url, model=model)

    raise ValueError(f"Unknown LLM provider: '{provider}'. Choose 'gemini' or 'ollama'.")


class ForensicReporter:
    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def generate_report(self, package: ForensicPromptPackage) -> str:
        logger.info("Generating forensic report via LLM...")
        report = self._backend.generate(package)
        logger.info(f"Report generated ({len(report)} chars).")
        return report

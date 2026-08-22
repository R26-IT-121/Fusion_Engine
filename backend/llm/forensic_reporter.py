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
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model,
            system_instruction=None,  # injected per-call via contents
        )
        self._model_name = model
        logger.info(f"Gemini backend initialized: {model}")

    def generate(self, package: ForensicPromptPackage) -> str:
        import google.generativeai as genai

        # Combine system + user into a single structured request
        full_prompt = f"{package.system_prompt}\n\n{package.user_prompt}"
        response = self._model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,       # low temp = deterministic, grounded output
                max_output_tokens=2048,
            ),
        )
        return response.text


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

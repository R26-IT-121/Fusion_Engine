"""Answer synthesis for the project assistant.

Two modes, chosen automatically:

- **Grounded generation** when GEMINI_API_KEY is set. Retrieved passages are
  pasted into the prompt and the model is instructed to answer only from them.
- **Extractive** otherwise. The top passages are returned verbatim with their
  citations. Less fluent, but never wrong and never unavailable — which means
  the assistant still demonstrates in a room with no internet or no API key.

Either way the answer carries `sources`, so any claim can be traced back to a
line in the repository. That is the same discipline the forensic reports use:
an ungrounded assertion is worse than no answer.
"""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

from chatbot.knowledge import build_corpus
from chatbot.retriever import Retriever

from chatbot.llm import get_llm_backend

MIN_SCORE = 1.0   # below this the corpus has nothing useful to say

# Conversational openers deserve a reply, not a "not in the documentation"
# rejection — the corpus genuinely has nothing to say about "hi", but treating
# a greeting as a failed lookup makes the assistant feel broken.
GREETINGS = {
    "hi", "hey", "hello", "yo", "sup", "hiya", "howdy",
    "good morning", "good afternoon", "good evening",
}
THANKS = {"thanks", "thank you", "thx", "ty", "cheers", "nice", "great", "cool"}

GREETING_REPLY = (
    "Hello. Ask me anything about DeepSentinel — the architecture, the "
    "GraphSAGE results and how they were measured, the API contract, or the "
    "dataset. I answer from the project's documentation and cite where each "
    "answer came from."
)
THANKS_REPLY = "Happy to help. Ask me anything else about the project."

SYSTEM_PROMPT = """\
You are the DeepSentinel project assistant. DeepSentinel is an undergraduate \
research platform for explainable financial fraud detection, built from four \
components: an Edge-Enhanced GraphSAGE relational detector, a Stratified VAE \
behavioural detector, a Temporal Convolutional Network, and a fusion engine \
that combines them and writes forensic reports.

Answer ONLY from the CONTEXT below, which is taken from the project's own \
documentation.

Rules:
- If the context does not contain the answer, say so plainly and suggest which \
document might cover it. Never invent numbers, file names, or results.
- Quote metrics exactly as written. Where a metric has been superseded (the \
project fixed a data-leakage bug and re-measured), give the current figure and \
note the older one only if the user asks about it.
- Be concise and concrete. Prefer the specific number or file path over a \
general description.
- You are speaking to the research team or an examiner, so precision matters \
more than friendliness.
"""


@dataclass
class _PromptPackage:
    """Duck-types ForensicPromptPackage — the LLM backends read only these two."""

    system_prompt: str
    user_prompt: str


@dataclass
class Answer:
    text: str
    sources: list[str]
    grounded: bool          # True when an LLM synthesised the answer
    confident: bool         # False when retrieval found nothing relevant


class ChatService:
    """Owns the corpus and index. Construct once at application startup."""

    def __init__(self, repo_root: str | Path | None = None, top_k: int = 5):
        # None lets knowledge.discover_roots() find every documentation root
        # for this checkout, rather than assuming one repository shape.
        self.corpus = build_corpus(repo_root)
        self.repo_root = Path(repo_root) if repo_root else _find_repo_root()
        self.retriever = Retriever(self.corpus)
        self.top_k = top_k
        self._model = None
        self._model_error: str | None = None

    # -- stats for /api/chat/health ------------------------------------
    @property
    def stats(self) -> dict:
        return {
            "repo_root": str(self.repo_root),
            "documents": len({c.source for c in self.corpus}),
            "passages": len(self.corpus),
            "llm_enabled": self._llm() is not None,
            "llm_backend": type(self._model).__name__ if self._model else None,
            "model_error": self._model_error,
        }

    def _llm(self):
        """Lazily build the LLM backend; degrade to extractive on failure."""
        if self._model is None and not self._model_error:
            self._model, self._model_error = get_llm_backend()
        return self._model

    # -- main entry point ----------------------------------------------
    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer("Ask me something about the project.", [], False, False)

        # Small talk short-circuits retrieval.
        normalised = question.lower().strip(" .!?,")
        if normalised in GREETINGS:
            return Answer(GREETING_REPLY, [], False, True)
        if normalised in THANKS:
            return Answer(THANKS_REPLY, [], False, True)

        hits = self.retriever.search(question, top_k=self.top_k)
        hits = [(c, s) for c, s in hits if s >= MIN_SCORE]
        if not hits:
            return Answer(
                "I could not find that in the project documentation. I can answer "
                "questions about the GraphSAGE component, the ablation results and "
                "how they were measured, the API contract, the fusion engine, and "
                "the dataset.",
                [],
                False,
                False,
            )

        sources = [chunk.citation for chunk, _ in hits]
        model = self._llm()
        if model is None:
            return Answer(self._extractive(hits), sources, False, True)

        context = "\n\n".join(
            f"[{i + 1}] {chunk.citation}\n{chunk.text}" for i, (chunk, _) in enumerate(hits)
        )
        convo = ""
        for turn in (history or [])[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            convo += f"{role}: {turn.get('content', '')}\n"

        package = _PromptPackage(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=f"CONTEXT:\n{context}\n\n{convo}User: {question}",
        )
        try:
            text = (model.generate(package) or "").strip()
            if text:
                return Answer(text, sources, True, True)
        except Exception as exc:                      # noqa: BLE001
            self._model_error = f"{type(exc).__name__}: {exc}"
            # Quota exhaustion is not a corpus problem, and silently switching
            # to quoted passages hides it — the operator needs to know the key
            # is spent, not wonder why answers suddenly got worse.
            if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                return Answer(
                    "The language model's request quota is exhausted, so I am "
                    "quoting the documentation directly instead of summarising "
                    "it. It usually resets within a minute, or after a day on "
                    "the free tier.\n\n" + self._extractive(hits),
                    sources, False, True,
                )
        # Generation failed — still answer from the retrieved passages.
        return Answer(self._extractive(hits), sources, False, True)

    @staticmethod
    def _extractive(hits) -> str:
        parts = [
            "_No language model is configured, so I am quoting the most "
            "relevant documentation directly rather than summarising it._\n"
        ]
        for chunk, _ in hits[:3]:
            body = " ".join(chunk.text.split())
            parts.append(f"**{chunk.citation}**\n{textwrap.shorten(body, 700, placeholder=' …')}")
        return "\n\n".join(parts)


def _find_repo_root() -> Path:
    """Kept for compatibility; discovery now lives in knowledge.discover_roots."""
    from chatbot.knowledge import discover_roots

    roots = discover_roots()
    return roots[0] if roots else Path(__file__).resolve().parents[1]

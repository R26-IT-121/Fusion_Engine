"""The reasoning loop behind the operator assistant.

Neither LLM backend in this project exposes native function calling — both are
plain text in, text out — so tool use runs over a small JSON protocol the model
emits and we parse:

    {"tool": "get_fraud_ring", "arguments": {...}}      -> we execute, loop
    {"answer": "..."}                                    -> done

That keeps Gemini and Ollama interchangeable, and keeps the contract explicit
enough to validate: an unknown tool name or a malformed payload is rejected and
fed back as an observation rather than trusted.

Every run is bounded by `max_steps`, and each step is recorded in a trace the
UI can show. An operator acting on a fraud verdict needs to see which tools ran
and what they returned — an unexplained answer about someone's account is not
usable evidence.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from assistant.tools import Tool, available_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the DeepSentinel operations assistant, working alongside a bank fraud \
analyst. You have live access to the platform through tools.

DeepSentinel fuses three detectors: a GraphSAGE relational model (fraud rings), \
a VAE behavioural model, and a temporal convolutional model.

To use a tool, reply with ONLY this JSON and nothing else:
{"tool": "<tool_name>", "arguments": {...}}

When you can answer, reply with ONLY:
{"answer": "<your answer>"}

Rules:
- Prefer a tool over guessing. Never invent scores, account ids or history.
- Use each tool at most once unless arguments genuinely differ.
- Base the final answer strictly on tool results. If a tool failed or returned \
nothing, say so plainly rather than filling the gap.
- Be concise and specific: name the accounts, quote the scores, state the \
pattern. You are writing for someone who will act on this.
- A missing modality is not a low score. Say which models were unavailable.
"""


@dataclass
class Step:
    tool: str
    arguments: dict
    result: Any
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    used_llm: bool = True
    truncated: bool = False       # hit max_steps before concluding


@dataclass
class _PromptPackage:
    """Duck-types ForensicPromptPackage for the shared LLM backends."""

    system_prompt: str
    user_prompt: str


def _tool_manual(tools: dict[str, Tool]) -> str:
    lines = []
    for t in tools.values():
        params = ", ".join(f"{k} ({v})" for k, v in t.parameters.items()) or "none"
        lines.append(f"- {t.name}: {t.description}\n  arguments: {params}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose or fences despite instructions, so we scan for a
    balanced object rather than trusting the whole string to parse.
    """
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start : i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    start = None
    return None


class Agent:
    def __init__(self, llm, settings):
        self.llm = llm
        self.settings = settings
        self.tools = available_tools(settings.allow_live_analysis)

    async def run(self, question: str, history: list[dict] | None = None) -> AgentResult:
        if self.llm is None:
            return await self._without_llm(question)

        convo = ""
        for turn in (history or [])[-4:]:
            role = "User" if turn.get("role") == "user" else "Assistant"
            convo += f"{role}: {turn.get('content', '')}\n"

        transcript = (
            f"TOOLS AVAILABLE:\n{_tool_manual(self.tools)}\n\n"
            f"{convo}User: {question}\n"
        )
        steps: list[Step] = []

        for _ in range(self.settings.max_steps):
            try:
                reply = self.llm.generate(
                    _PromptPackage(SYSTEM_PROMPT, transcript)
                ) or ""
            except Exception as exc:                   # noqa: BLE001
                logger.warning(f"Assistant LLM call failed: {exc}")
                return AgentResult(
                    self._summarise(steps)
                    or "The language model is unavailable, so I cannot answer this.",
                    steps,
                    used_llm=False,
                )

            action = _extract_json(reply)
            if action is None:
                # Unparseable but non-empty: treat prose as the answer.
                return AgentResult(reply.strip(), steps)

            if "answer" in action:
                return AgentResult(str(action["answer"]).strip(), steps)

            name = str(action.get("tool", ""))
            args = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}

            tool = self.tools.get(name)
            if tool is None:
                observation = {
                    "error": f"Unknown tool '{name}'. Available: {', '.join(self.tools)}"
                }
                steps.append(Step(name or "(none)", args, None, observation["error"]))
            else:
                try:
                    observation = await tool.run(**args)
                except Exception as exc:               # noqa: BLE001
                    logger.exception(f"Tool {name} raised")
                    observation = {"error": f"{type(exc).__name__}: {exc}"}
                steps.append(
                    Step(name, args, observation, observation.get("error")
                         if isinstance(observation, dict) else None)
                )

            transcript += (
                f"Assistant: {json.dumps(action)}\n"
                f"Observation: {json.dumps(observation, default=str)[:3500]}\n"
            )

        # Ran out of steps — answer from what we gathered rather than nothing.
        return AgentResult(
            self._summarise(steps)
            or "I could not complete this within the configured step budget.",
            steps,
            truncated=True,
        )

    # ── No-LLM path ──────────────────────────────────────────────────────
    async def _without_llm(self, question: str) -> AgentResult:
        """Keyword routing so the assistant still works with no model configured.

        Not clever, but it keeps the feature demonstrable offline and makes the
        tool layer independently testable.
        """
        q = question.lower()
        accounts = re.findall(r"\bC\d{3,}\b", question)

        if any(w in q for w in ("status", "health", "running", "reachable",
                                "online", "offline", "down", "available",
                                "models up", "system")):
            name = "get_system_status"
            args: dict = {}
        elif any(w in q for w in ("ring", "network", "subgraph", "mule", "involved")):
            name, args = "get_fraud_ring", {}
        elif any(w in q for w in ("history", "before", "previously", "past",
                                  "seen", "recent", "flagged")):
            name, args = "search_analysis_history", {}
        elif accounts:
            name, args = "search_analysis_history", {"account": accounts[0]}
        else:
            name, args = "search_documentation", {"query": question}

        if accounts and name in ("get_fraud_ring", "get_model_scores"):
            args.setdefault("nameOrig", accounts[0])
            if len(accounts) > 1:
                args.setdefault("nameDest", accounts[1])
        if name == "search_analysis_history" and accounts:
            args.setdefault("account", accounts[0])

        tool = self.tools.get(name)
        if tool is None:
            return AgentResult(
                "No language model is configured and no tool matches this question.",
                [], used_llm=False,
            )
        try:
            result = await tool.run(**args)
        except Exception as exc:                       # noqa: BLE001
            result = {"error": f"{type(exc).__name__}: {exc}"}

        step = Step(name, args, result, result.get("error") if isinstance(result, dict) else None)
        return AgentResult(
            "No language model is configured, so here is the raw result of "
            f"`{name}`:\n\n```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```",
            [step],
            used_llm=False,
        )

    @staticmethod
    def _summarise(steps: list[Step]) -> str:
        if not steps:
            return ""
        parts = ["I gathered the following before running out of steps:\n"]
        for s in steps:
            parts.append(
                f"**{s.tool}** → "
                + (s.error or json.dumps(s.result, default=str)[:600])
            )
        return "\n\n".join(parts)

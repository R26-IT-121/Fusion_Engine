"""Builds the assistant's knowledge base from the project's own documentation.

The corpus is the repository itself — walkthroughs, API contracts, EDA findings,
component READMEs. Nothing is hand-written for the chatbot, so the answers stay
in sync with the project as the docs evolve.

Markdown is split on headings rather than fixed character windows. A heading is
a natural semantic boundary, and keeping the heading path lets every answer cite
where it came from ("system_walkthrough.md > 13. Results > 13.3 Finding 1"),
which is what makes a grounded answer checkable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Documentation worth answering from, relative to the repo root. Ordered by
# authority: when two documents disagree, the earlier one wins (see PRIORITY).
# Documentation lives in different places depending on how the project is
# checked out: the monorepo has every component side by side, while the fusion
# engine is also published as a standalone repository with only its own docs.
# Path-shaped globs silently matched nothing in the standalone layout and the
# assistant answered every question with "not in the documentation", so
# discovery walks whatever roots it is given instead of assuming a shape.
SKIP_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache",
    "dist", "build", "site-packages", ".qodo", "chroma_store",
    "chroma_store_test", ".ipynb_checkpoints",
}
MAX_DOCS = 400          # a runaway checkout should not blow up startup
DOCS_ROOT_ENV = "CHATBOT_DOCS_ROOT"   # os.pathsep-separated override

# Documents superseded by later work. Still indexed (someone may ask about the
# PP1 presentation) but down-weighted so current sources win by default.
STALE_MARKERS = ("Superseded metrics", "PP1", "Progress Presentation 1")

MAX_CHUNK_CHARS = 2400
MIN_CHUNK_CHARS = 120


@dataclass
class Chunk:
    """One retrievable passage."""

    text: str
    source: str          # repo-relative path
    heading: str         # heading path within the document
    stale: bool = False
    tokens: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.source}" + (f" › {self.heading}" if self.heading else "")


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, plus decimal numbers kept intact.

    Numbers matter here — users ask "what is the F1" and the answer hinges on
    0.4056 vs 0.5387, so digits must survive tokenisation.
    """
    return _TOKEN_RE.findall(text.lower())


def _split_markdown(text: str) -> list[tuple[str, str]]:
    """Split into (heading_path, body) on ATX headings, preserving hierarchy."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    buf: list[str] = []
    current = ""
    in_code = False

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if buf:
                sections.append((current, "\n".join(buf).strip()))
                buf = []
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            current = " › ".join(stack)
        else:
            buf.append(line)
    if buf:
        sections.append((current, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def _pack(heading: str, body: str, source: str, stale: bool) -> list[Chunk]:
    """Emit chunks under the size cap, splitting long sections on blank lines."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [Chunk(body, source, heading, stale)] if len(body) >= MIN_CHUNK_CHARS else []

    chunks, buf = [], []
    size = 0
    for para in body.split("\n\n"):
        if size + len(para) > MAX_CHUNK_CHARS and buf:
            chunks.append(Chunk("\n\n".join(buf), source, heading, stale))
            buf, size = [], 0
        buf.append(para)
        size += len(para)
    if buf:
        chunks.append(Chunk("\n\n".join(buf), source, heading, stale))
    return [c for c in chunks if len(c.text) >= MIN_CHUNK_CHARS]


def discover_roots(start: Path | None = None) -> list[Path]:
    """Where to look for documentation, most authoritative first.

    An explicit override wins. Otherwise we take the monorepo root when the
    components sit side by side, and always include the package's own
    repository so the assistant has something to say in any layout.
    """
    import os

    override = os.getenv(DOCS_ROOT_ENV, "").strip()
    if override:
        roots = [Path(p).expanduser() for p in override.split(os.pathsep) if p.strip()]
        return [r for r in roots if r.is_dir()]

    here = (start or Path(__file__)).resolve()
    roots: list[Path] = []

    for parent in here.parents:
        siblings = {c.name for c in parent.iterdir() if c.is_dir()} if parent.is_dir() else set()
        # Monorepo: components checked out together.
        if "GraphSage" in siblings and ("fusion_engine" in siblings or "TS-TCN" in siblings):
            roots.append(parent)
            break
        # Standalone repo root, identified by its own project manifest.
        if (parent / "pyproject.toml").exists() and parent not in roots:
            roots.append(parent)

    if not roots:
        roots.append(here.parents[1])
    return roots


def build_corpus(repo_root: str | Path | list | None = None) -> list[Chunk]:
    """Discover, split and tokenise every markdown document under the roots."""
    if repo_root is None:
        roots = discover_roots()
    elif isinstance(repo_root, (list, tuple)):
        roots = [Path(r) for r in repo_root]
    else:
        roots = [Path(repo_root)]

    seen: set[Path] = set()
    corpus: list[Chunk] = []

    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md")):
            if len(seen) >= MAX_DOCS:
                break
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                rel = path.name
            stale = any(marker in text[:4000] for marker in STALE_MARKERS)
            for heading, body in _split_markdown(text):
                for chunk in _pack(heading, body, rel, stale):
                    chunk.tokens = tokenize(f"{heading} {chunk.text}")
                    corpus.append(chunk)
    return corpus

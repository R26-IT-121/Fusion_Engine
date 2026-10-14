"""Per-run logs on disk, one folder per component, one tree per day.

    logs/
      2026-08-31/
        pipeline/    pipeline.log     every transaction, end to end
        graph/       graph.log        the network detector, call by call
        behaviour/   behaviour.log    the behaviour detector
        timing/      timing.log       the timing detector
        fusion/      fusion.log       fusion, retrieval, narrative, PDF, email

Why a file at all, when the console already shows a live feed: the feed is a
ring buffer in memory. It holds the last two hundred events, it is gone when
the process restarts, and it cannot answer "what did the system do during
yesterday's demo". These files can.

**What the detector folders hold.** Each detector runs in its own process,
owned by whoever built it, and this cannot reach inside them. What is recorded
here is *the platform's view* of each detector: what was sent, what came back,
how long it took, and what happened when it did not answer. That is the record
that matters for an integration problem, and it is the one nobody had. A
detector's own internal logging is a separate thing, in its own repository.

The fusion log is deliberately the most detailed, because its work is a
sequence of slow, failable steps — retrieval, prompt, LLM, diagram, PDF, mail —
and "the alert was slow" is not a diagnosis. Every stage is timed and its
outcome recorded, so the answer is a line rather than a guess.

Nothing here may break screening. Every write is wrapped: a full disk or a
read-only directory costs the log, never the verdict.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# The five streams. Order is the order a transaction moves through them, which
# is also the order they read best in a directory listing.
STREAMS = ("pipeline", "graph", "behaviour", "timing", "fusion")

_DEFAULT_ROOT = "./logs"
_lock = threading.Lock()
_handles: dict[str, object] = {}
_open_for: date | None = None
_root_override: Path | None = None
_disabled = False


def _root() -> Path:
    if _root_override is not None:
        return _root_override
    try:
        from backend import config

        return Path(str(config.get("paths", "run_logs") or _DEFAULT_ROOT))
    except Exception:                                    # noqa: BLE001
        return Path(_DEFAULT_ROOT)


def set_root(path: str | Path) -> None:
    """Point the logs somewhere else. For tests."""
    global _root_override
    with _lock:
        _close_all()
        _root_override = Path(path)


def today_dir() -> Path:
    """The dated folder for right now, created if missing."""
    return _root() / date.today().isoformat()


def _close_all() -> None:
    for fh in _handles.values():
        try:
            fh.close()                                   # type: ignore[attr-defined]
        except Exception:                                # noqa: BLE001
            pass
    _handles.clear()


def _handle(stream: str):
    """The open file for a stream, rolling to a new folder at midnight."""
    global _open_for, _disabled

    if _disabled:
        return None

    now = date.today()
    if _open_for != now:
        _close_all()
        _open_for = now

    fh = _handles.get(stream)
    if fh is not None:
        return fh

    try:
        folder = today_dir() / stream
        folder.mkdir(parents=True, exist_ok=True)
        fh = open(folder / f"{stream}.log", "a", encoding="utf-8")
        _handles[stream] = fh
        return fh
    except Exception as exc:                             # noqa: BLE001
        # Once, then never again — a log that cannot be written must not
        # produce a warning per transaction.
        logger.warning(f"Run logs unavailable ({exc}); continuing without them.")
        _disabled = True
        return None


def write(stream: str, message: str, *, txn: str | None = None,
          level: str = "INFO") -> None:
    """One line. Never raises."""
    try:
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        ref = f"{txn:<44}" if txn else " " * 44
        line = f"{stamp}  {level:<5} {ref} {message}\n"
        with _lock:
            fh = _handle(stream)
            if fh is None:
                return
            fh.write(line)                               # type: ignore[attr-defined]
            fh.flush()                                   # type: ignore[attr-defined]
    except Exception:                                    # noqa: BLE001
        pass


def event(streams: tuple[str, ...] | str, message: str, **kw) -> None:
    """The same line into more than one stream.

    A detector call belongs in that detector's log *and* in the pipeline
    narrative; writing it twice is what lets either file be read on its own.
    """
    for s in ((streams,) if isinstance(streams, str) else streams):
        write(s, message, **kw)


@contextmanager
def stage(stream: str, name: str, *, txn: str | None = None,
          also: tuple[str, ...] = ()):
    """Time one step and record how it ended.

    Emits a START line, then OK with a duration, or FAIL with the exception —
    so a stalled pipeline shows a START with no matching OK, and the last line
    in the file names the step it stopped on.

    The exception is always re-raised. This observes; it does not swallow.
    """
    # In its own file the step needs no prefix — every line there is that
    # component's. In the shared pipeline file it does: three consecutive
    # "classify" lines with nothing to separate them is unreadable, and it was
    # the first thing wrong with these logs.
    def emit(text: str, level: str = "INFO") -> None:
        write(stream, text, txn=txn, level=level)
        for other in also:
            write(other, f"{stream}: {text}", txn=txn, level=level)

    started = time.perf_counter()
    emit(f"START  {name}")
    detail_box: dict[str, str] = {}
    try:
        yield detail_box
    except Exception as exc:                             # noqa: BLE001
        ms = (time.perf_counter() - started) * 1000
        emit(f"FAIL   {name} after {_ms(ms)} — "
             f"{type(exc).__name__}: {str(exc)[:160]}", "ERROR")
        raise
    else:
        ms = (time.perf_counter() - started) * 1000
        note = detail_box.get("note", "")
        emit(f"OK     {name} in {_ms(ms)}" + (f" — {note}" if note else ""))


def _ms(ms: float) -> str:
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.2f}s"


def banner(message: str) -> None:
    """A separator in every stream — process start, monitor start, a reset."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for s in STREAMS:
        write(s, f"{'─' * 8} {message} · {stamp} {'─' * 8}")


def describe() -> dict:
    """Where the logs are and how big they have grown. For the console."""
    root = today_dir()
    out: dict = {"date": date.today().isoformat(), "path": str(root.resolve()),
                 "enabled": not _disabled, "streams": {}}
    for s in STREAMS:
        f = root / s / f"{s}.log"
        try:
            out["streams"][s] = {"file": str(f), "bytes": f.stat().st_size if f.exists() else 0}
        except Exception:                                # noqa: BLE001
            out["streams"][s] = {"file": str(f), "bytes": None}
    return out

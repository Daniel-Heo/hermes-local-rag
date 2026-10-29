"""Turn-local, opt-in Local RAG diagnostics.

This deliberately retains no transcript or provider request data.  It only
keeps the already-rendered recall and outcomes observed during one turn.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import html
import re
import threading

from .policy import IngestDecision, classify_text


_LIMIT = 1_800
_MAX_TURNS = 32


def _safe(value: object, limit: int = 240) -> str:
    """Render untrusted diagnostic text as inert Markdown, within a bound."""
    text = str(value).replace("\x00", "").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = html.escape(text[:limit], quote=False)
    return re.sub(r"([\\`*_{}\[\]#|!])", r"\\\1", text)


@dataclass
class _Turn:
    recall: str = ""
    changes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    attempted_save: bool = False
    truncated: bool = False
    search: str = ""

    def add(self, collection: list[str], text: str) -> None:
        entries = self.changes + self.failures + self.observations
        if len(entries) >= 12 or sum(map(len, entries)) + len(text) > 2800:
            self.truncated = True
            return
        collection.append(text)


class DebugTurns:
    """Bounded profile/session tracker used by one provider instance."""
    def __init__(self) -> None:
        self._turns: OrderedDict[str, _Turn] = OrderedDict()
        self._lock = threading.RLock()

    def start(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            self._turns.pop(key, None)  # interrupted/unfinalized prior turn
            self._turns[key] = _Turn()
            while len(self._turns) > _MAX_TURNS:
                self._turns.popitem(last=False)

    def recall(self, key: str, rendered: str) -> None:
        with self._lock:
            turn = self._turns.get(key)
            if turn:
                # Stored legacy records predate current ingest policy.  Check the
                # complete payload before line splitting or excerpt truncation.
                if classify_text(rendered) is IngestDecision.BLOCK:
                    turn.recall = "Context hidden by safety filter."
                    return
                lines = ["> " + _safe(line, 600) for line in rendered.splitlines() if line.strip()]
                safe = "\n".join(lines)
                turn.recall = safe[:_LIMIT]
                turn.truncated |= len(safe) > _LIMIT

    def change(self, key: str, fact: object, scope: object, *, action: str = "Saved") -> None:
        with self._lock:
            turn = self._turns.get(key)
            if turn:
                turn.attempted_save = True
                turn.add(turn.changes, f"• {action}: {_safe(fact)} — scope: {_safe(scope, 80)}")

    def search(self, key: str, results: list[dict]) -> None:
        with self._lock:
            turn = self._turns.get(key)
            if turn:
                lines = []
                for item in results[:5]:
                    excerpt = item.get('text') or item.get('description') or 'image; pixels not inspected'
                    if classify_text(str(excerpt)) is IngestDecision.BLOCK:
                        excerpt = 'content hidden by safety filter'
                    lines.append(f"• #{_safe(item.get('id', '?'), 30)}: {_safe(excerpt)}")
                text = '\n'.join(lines) if lines else 'No matches.'
                turn.search = text[:900]
                turn.truncated |= len(text) > 900 or len(results) > 5

    def forgotten(self, key: str, memory_id: int | str, *, kind: str = "record") -> None:
        with self._lock:
            turn = self._turns.get(key)
            if turn:
                # Do not create a fresh chat copy of forgotten data.
                turn.recall = "Context hidden after record deletion."
                turn.search = ""
                turn.changes.clear()
                self.change(key, f"{kind} #{memory_id}", "current accessible scope", action="Deleted")

    def failure(self, key: str, label: str, *, attempted_write: bool = True) -> None:
        with self._lock:
            turn = self._turns.get(key)
            if turn:
                turn.attempted_save |= attempted_write
                turn.add(turn.failures, f"• {_safe(label, 120)}")

    def observation(self, key: str, label: str) -> None:
        """Record an outcome whose storage mutation cannot be inferred."""
        with self._lock:
            turn = self._turns.get(key)
            if turn:
                turn.add(turn.observations, f"• {_safe(label, 180)}")

    def append(self, key: str, response: str, *, enabled: bool) -> str:
        with self._lock:
            turn = self._turns.pop(key, None)
        if not enabled or turn is None:
            return response
        parts = ["**Local RAG: diagnostics**"]
        if turn.recall:
            parts += ["\nRetrieved context:", turn.recall]
        else:
            parts.append("\nRetrieved context: none.")
        if turn.search:
            parts += ["\nExplicit search — passed to agent:", turn.search]
        if turn.changes:
            parts += ["\nMemory changes:", *turn.changes]
        if turn.observations:
            parts += ["\nObserved results:", *turn.observations]
        if not turn.changes and not turn.observations and turn.attempted_save:
            parts.append("\nNo saves: the operation did not change memory.")
        elif not turn.changes and not turn.observations:
            parts.append("\nNo saves: Local RAG received no successful write operation.")
        if turn.failures:
            parts += ["\nRejections/failures:", *turn.failures]
        if turn.truncated:
            parts.append("\nDiagnostics truncated.")
        return response.rstrip() + "\n\n---\n\n" + "\n".join(parts)

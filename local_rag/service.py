from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .chunking import chunk_text
from .policy import IngestDecision, classify_text, validate_memory_item
from .store import MemoryStore, SearchResult


_BLOCKED_NAMES = {".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"}
_ALLOWED_SUFFIXES = {".md", ".txt", ".rst", ".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml"}


class LocalRagService:
    def __init__(self, *, store: MemoryStore, embedder: Any, namespace: str, allowed_roots: list[Path] | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.namespace = namespace
        self.allowed_roots = [Path(root).expanduser().resolve() for root in (allowed_roots or [])]
        self.session_id = ""

    def update_memory_item(self, memory_id: int, *, expected_text: str, new_text: str, **fields: Any) -> dict:
        """Prepare outside SQLite, then compare against the preflight row.

        expected_text is content CAS, not a revision token. An A -> B -> A
        change completed before preflight cannot be detected by this API.
        """
        if type(memory_id) is not int or not 1 <= memory_id <= 2**63 - 1:
            raise ValueError("Memory ID must be an integer from 1 to 9223372036854775807")
        if not isinstance(expected_text, str) or not expected_text.strip() or len(expected_text) > 2000:
            raise ValueError("expected_text must contain the exact previously retrieved memory text")
        if fields.keys() - {"subject", "durability", "importance", "confidence", "tags"}:
            raise ValueError("Unknown atomic-update fields")
        item = validate_memory_item({"text": new_text, **fields}, partial_defaults=True)
        # Capture authority from runtime service state, never the semantic item.
        project = str(self.allowed_roots[0]) if self.allowed_roots else ""
        session_id = self.session_id
        original = self.store.memory_item_for_update(
            self.namespace, memory_id, project=project, session_id=session_id,
        )
        if original is None:
            return {"updated": False, "reason": "unavailable"}
        if original["text"] != expected_text:
            return {"updated": False, "reason": "conflict"}
        try:
            embedding = self.embedder.embed_document(item["text"])
            status = self.store.update_memory_item(
                self.namespace, memory_id, expected_text=expected_text, expected_row=original, item=item,
                embedding=embedding, project=project, session_id=session_id,
            )
        except Exception as exc:
            # Embedders/SQLite may include payloads or private paths in errors.
            raise RuntimeError("Atomic memory update failed; no change was saved") from exc
        return {"updated": True, "id": memory_id} if status == "updated" else {"updated": False, "reason": status}

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        project = str(self.allowed_roots[0]) if self.allowed_roots else ""
        return self.store.search(
            self.namespace,
            query,
            self.embedder.embed_query(query),
            limit=limit,
            project=project,
            session_id=self.session_id,
        )

    def index_text(
        self,
        text: str,
        *,
        source: str,
        kind: str = "document",
        ttl_seconds: float | None = None,
        project: str = "",
        metadata: dict | None = None,
    ) -> int:
        records = []
        for position, chunk in enumerate(chunk_text(text)):
            if classify_text(chunk) is not IngestDecision.INDEX:
                continue
            records.append(
                {
                    "text": chunk,
                    "embedding": self.embedder.embed_document(chunk),
                    "kind": kind,
                    "ttl_seconds": ttl_seconds,
                    "project": project,
                    "metadata": {**(metadata or {}), "chunk": position},
                }
            )
        return self.store.replace_source(self.namespace, source, records)

    def index_path(self, path: str | Path) -> int:
        resolved = Path(path).expanduser().resolve()
        if resolved.name.lower() in _BLOCKED_NAMES or resolved.suffix.lower() not in _ALLOWED_SUFFIXES:
            raise ValueError("File type is not allowed for local RAG indexing")
        project_root = next((root for root in self.allowed_roots if resolved.is_relative_to(root)), None)
        if project_root is None:
            raise ValueError("Path is not allowed for local RAG indexing")
        if not resolved.is_file() or resolved.stat().st_size > 5_000_000:
            raise ValueError("File is missing or exceeds the 5 MB indexing limit")
        raw = resolved.read_text(encoding="utf-8", errors="replace")
        if classify_text(raw) is IngestDecision.BLOCK:
            raise ValueError("File contains secret-like material and was not indexed")
        digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
        return self.index_text(
            raw,
            source=f"file:{digest}:{resolved}",
            kind="document",
            project=str(project_root),
            metadata={"path": str(resolved)},
        )

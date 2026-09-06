"""Frozen 1.4.0 search for temporary synthetic evaluation only.

Original store.py SHA256: 2898b3d5e0d3279f36bff09cd4e3b1e706ebccf5fe32290bc122a152f40e5a3b
"""
from __future__ import annotations

import re
import time
from array import array
from typing import Iterable

from local_rag.store import MemoryStore, SearchResult


class BaselineStore(MemoryStore):
    def search(
        self,
        namespace: str,
        query: str,
        query_embedding: Iterable[float],
        *,
        limit: int = 5,
        project: str | None = None,
        session_id: str | None = None,
    ) -> list[SearchResult]:
        qvec = list(query_embedding)
        if len(qvec) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(qvec)}")
        now = time.time()
        with self._lock:
            visibility = ""
            lexical_visibility = ""
            visibility_args: tuple[str, ...] = ()
            if project is not None and session_id is not None:
                def visibility_clause(prefix: str) -> str:
                    metadata = f"{prefix}metadata_json"
                    project_column = f"{prefix}project"
                    return (
                    " AND ("
                    f"(json_extract({metadata},'$.scope') IS NULL AND ({project_column}='' OR {project_column}=?)) OR "
                    f"json_extract({metadata},'$.scope')='global' OR "
                    f"(json_extract({metadata},'$.scope')='project' AND {project_column}=?) OR "
                    f"(json_extract({metadata},'$.scope')='session' AND json_extract({metadata},'$.session_id')=?)"
                    ")"
                    )
                visibility = visibility_clause("")
                lexical_visibility = visibility_clause("m.")
                visibility_args = (project, project, session_id)
            rows = self._conn.execute(
                "SELECT id,text,source,embedding,kind,confidence,importance,created_at "
                "FROM memories WHERE namespace=? AND (expires_at IS NULL OR expires_at>?)" + visibility,
                (namespace, now, *visibility_args),
            ).fetchall()
            lexical: dict[int, float] = {}
            terms = re.findall(r"[\w-]{2,}", query, flags=re.UNICODE)
            if terms:
                expression = " OR ".join(f'"{term}"' for term in terms[:16])
                for row in self._conn.execute(
                    "SELECT m.id,bm25(memories_fts) rank FROM memories_fts "
                    "JOIN memories m ON m.id=memories_fts.rowid "
                    "WHERE memories_fts MATCH ? AND m.namespace=? "
                    "AND (m.expires_at IS NULL OR m.expires_at>?)" + lexical_visibility + " LIMIT 100",
                    (expression, namespace, now, *visibility_args),
                ):
                    lexical[int(row["id"])] = 1.0
        ranked: list[SearchResult] = []
        for row in rows:
            memory_id = int(row["id"])
            semantic = self._cosine(qvec, array("f", row["embedding"]))
            age_days = max(0.0, (now - float(row["created_at"])) / 86400)
            freshness = 1.0 / (1.0 + age_days / 90.0)
            durable_bonus = 0.08 if row["kind"] == "durable" else 0.0
            score = 0.68 * semantic + 0.20 * lexical.get(memory_id, 0.0) + 0.07 * float(row["importance"]) + 0.05 * freshness + durable_bonus
            if score >= 0.34:
                ranked.append(SearchResult(memory_id, row["text"], row["source"], score, row["kind"], float(row["confidence"])))
        results = sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]
        if results:
            with self._lock, self._conn:
                self._conn.executemany(
                    "UPDATE memories SET accessed_at=?, access_count=access_count+1 WHERE id=?",
                    [(now, item.id) for item in results],
                )
        return results

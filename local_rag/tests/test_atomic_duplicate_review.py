"""Duplicate checks must not turn hidden text guesses into an existence oracle."""
from __future__ import annotations

import pytest

from local_rag.tests.test_atomic_update import NEW, OLD, provider, remember, rows, update


def add_duplicate(p, target, *, boundary, key_style):
    scope = target
    project = p._project if scope == "project" else ""
    session = "a"
    namespace = p._namespace
    ttl = None
    if boundary == "expired":
        ttl = -1
    elif boundary == "project-column":
        project = "/hidden-project"
    elif boundary == "other-session":
        session = "b"
    elif boundary == "other-project":
        project = "/hidden-project"
    elif boundary == "unknown-scope":
        scope = "unknown"
    elif boundary == "other-namespace":
        namespace = "other:local"
    identity = p._project if target == "project" else "a" if target == "session" else ""
    # Canonical deliberately occupies the target's hash, even for inconsistent
    # legacy scope columns. Historical hashes exercise the semantic scan alone.
    key = (f"memory_item\0{target}\0{identity}\0{' '.join(NEW.casefold().split())}"
           if key_style == "canonical" else f"historical-duplicate:{boundary}:{scope}")
    assert p._store.add(
        namespace, NEW, [0., 1., 0.], source=f"historical-session:{session}",
        kind="memory_item", project=project, ttl_seconds=ttl,
        metadata={"scope": scope, "session_id": session, "subject": "prior record"},
        dedupe_key=key,
    )
    return rows(p)[-1]


@pytest.mark.parametrize("key_style", ["historical", "canonical"])
@pytest.mark.parametrize("scope,boundary", [
    ("global", "expired"), ("project", "expired"), ("session", "expired"),
    ("global", "project-column"), ("session", "project-column"),
    ("session", "other-session"), ("project", "other-project"),
    ("global", "unknown-scope"), ("global", "other-namespace"),
])
def test_invisible_duplicate_behaves_like_absent_text_and_preserves_other_row(provider, scope, boundary, key_style):
    memory_id = remember(provider, scope)
    other = add_duplicate(provider, scope, boundary=boundary, key_style=key_style)
    assert provider._store.memory_item_for_update(
        provider._namespace, other["id"], project=provider._project, session_id="a",
    ) is None
    # A hidden/expired match must have the same successful outcome as no match,
    # including when UNIQUE(namespace, content_hash) would otherwise reject it.
    assert update(provider, memory_id) == {"updated": True, "id": memory_id}
    after = rows(provider)
    assert after[0]["id"] == memory_id and after[0]["text"] == NEW
    assert after[1] == other
    assert after[0]["content_hash"] != other["content_hash"] or boundary == "other-namespace"
    conn = provider._store._conn
    assert {r[0] for r in conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'cancelled'")} == {memory_id, other["id"]}
    assert not conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'approval'").fetchall()
    assert not conn.in_transaction


@pytest.mark.parametrize("key_style", ["historical", "canonical"])
@pytest.mark.parametrize("scope", ["global", "project", "session"])
def test_visible_same_scope_duplicate_still_conflicts_and_preserves_every_row(provider, scope, key_style):
    memory_id = remember(provider, scope)
    add_duplicate(provider, scope, boundary="visible", key_style=key_style)
    before = rows(provider)
    assert update(provider, memory_id) == {"updated": False, "reason": "conflict"}
    assert rows(provider) == before
    conn = provider._store._conn
    assert [r[0] for r in conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'approval'")] == [memory_id]
    assert [r[0] for r in conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'cancelled'")] == [before[1]["id"]]


@pytest.mark.parametrize("boundary", ["expired", "project-column", "unknown-scope", "other-namespace"])
def test_hidden_target_and_unknown_id_have_identical_nondisclosing_outcomes(provider, boundary):
    hidden = add_duplicate(provider, "global", boundary=boundary, key_style="canonical")
    before = rows(provider)
    assert update(provider, hidden["id"], expected_text=NEW) == update(provider, hidden["id"] + 999, expected_text=NEW) == {"updated": False, "reason": "unavailable"}
    assert rows(provider) == before


@pytest.mark.parametrize("failure", ["fts", "commit"])
def test_hidden_hash_collision_does_not_bypass_transaction_rollback(provider, failure):
    import sqlite3

    memory_id = remember(provider, "global")
    add_duplicate(provider, "global", boundary="project-column", key_style="canonical")
    before = rows(provider)
    conn = provider._store._conn
    if failure == "fts":
        conn.execute("CREATE TRIGGER reject_update AFTER UPDATE OF text ON memories BEGIN SELECT RAISE(ABORT, 'private failure'); END")
    else:
        conn.set_authorizer(lambda action, arg1, *rest: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT" else sqlite3.SQLITE_OK)
    try:
        assert update(provider, memory_id) == {"error": "Atomic memory update failed; no change was saved"}
    finally:
        conn.set_authorizer(None)
    assert rows(provider) == before
    assert not conn.in_transaction
    assert [r[0] for r in conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'approval'")] == [memory_id]
    assert [r[0] for r in conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'cancelled'")] == [before[1]["id"]]


def test_nonmemory_hash_owner_does_not_create_a_semantic_conflict(provider):
    memory_id = remember(provider, "global")
    other = add_duplicate(provider, "global", boundary="visible", key_style="canonical")
    conn = provider._store._conn
    with conn:
        conn.execute("UPDATE memories SET kind='document',source='file:legacy' WHERE id=?", (other["id"],))
    before = rows(provider)
    assert update(provider, memory_id) == {"updated": True, "id": memory_id}
    assert rows(provider)[1] == before[1]


def test_occupied_fallback_hash_is_preserved_and_next_update_still_succeeds(provider):
    memory_id = remember(provider, "global")
    other = add_duplicate(provider, "global", boundary="expired", key_style="canonical")
    assert update(provider, memory_id) == {"updated": True, "id": memory_id}
    fallback_hash = rows(provider)[0]["content_hash"]
    assert update(provider, memory_id, expected_text=NEW, new_text=OLD) == {"updated": True, "id": memory_id}
    assert provider._store.add(provider._namespace, "An unrelated legacy document must remain unchanged.", [1., 0., 0.], source="file:legacy", kind="document")
    conn = provider._store._conn
    with conn:
        conn.execute("UPDATE memories SET content_hash=? WHERE id=?", (fallback_hash, rows(provider)[-1]["id"]))
    before = rows(provider)
    assert update(provider, memory_id) == {"updated": True, "id": memory_id}
    assert rows(provider)[1:] == before[1:]
    assert rows(provider)[1] == other
    assert len({r["content_hash"] for r in rows(provider)}) == 3


def test_unrelated_unique_failure_is_generic_and_preserves_row_and_fts(provider):
    memory_id = remember(provider, "global")
    before = rows(provider)
    conn = provider._store._conn
    conn.executescript("""
        CREATE TABLE unique_guard (value TEXT UNIQUE);
        INSERT INTO unique_guard VALUES ('private payload');
        CREATE TRIGGER fail_unique AFTER UPDATE OF text ON memories BEGIN
            INSERT INTO unique_guard VALUES ('private payload');
        END;
    """)
    assert update(provider, memory_id) == {"error": "Atomic memory update failed; no change was saved"}
    assert rows(provider) == before
    assert not conn.in_transaction
    assert [r[0] for r in conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'approval'")] == [memory_id]
    assert not conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'cancelled'").fetchall()

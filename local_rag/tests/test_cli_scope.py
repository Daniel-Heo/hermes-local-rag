"""Standalone CLI scope uses the working directory, never caller session guesses."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import sys

import pytest

from local_rag import cli
from local_rag.config import LocalRagConfig
from local_rag.store import MemoryStore


class Embedder:
    dimensions = 128

    def __init__(self, *args, **kwargs):
        pass

    def embed_query(self, text):
        return [1.0] + [0.0] * (self.dimensions - 1)


@pytest.fixture
def cli_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project_a, project_b = tmp_path / "project-a", tmp_path / "project-b"
    project_a.mkdir()
    project_b.mkdir()
    LocalRagConfig(embedding_dimensions=128).save(home)
    db = home / "local-rag" / "memory.db"
    store = MemoryStore(db, dimensions=128)
    records = {}
    for name, scope, project, namespace in (
        ("project_a", "project", str(project_a.resolve()), "default:local"),
        ("project_b", "project", str(project_b.resolve()), "default:local"),
        ("session", "session", "", "default:local"),
        ("global", "global", "", "default:local"),
        ("other_namespace", "global", "", "other:local"),
    ):
        assert store.add(
            namespace, f"The {name} project requires the copper approval mark.",
            Embedder().embed_query("approval"), source="agent:session-a", kind="memory_item",
            project=project, metadata={"scope": scope, "session_id": "session-a"},
        )
        records[name] = store._conn.execute("SELECT max(id) FROM memories").fetchone()[0]
    store.close()
    # A known session in state.db is a stored identity, not proof the CLI owns it.
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, cwd TEXT, git_repo_root TEXT)")
        conn.execute("INSERT INTO sessions VALUES (?,?,NULL)", ("session-a", str(project_a.resolve())))
    monkeypatch.chdir(project_a)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(cli, "InferenceClient", Embedder)
    return home, db, project_a, project_b, records


def invoke(home, monkeypatch, capsys, *arguments):
    monkeypatch.setattr(sys, "argv", ["local-rag", "--home", str(home), *map(str, arguments)])
    assert cli.main() == 0
    return json.loads(capsys.readouterr().out)


def snapshot(db):
    with sqlite3.connect(db) as conn:
        return (
            conn.execute("SELECT * FROM memories ORDER BY id").fetchall(),
            conn.execute("SELECT rowid,text FROM memories_fts ORDER BY rowid").fetchall(),
        )


def test_cli_forget_uses_resolved_cwd_project_authority(cli_home, monkeypatch, capsys, tmp_path):
    home, db, project_a, project_b, records = cli_home
    alias = tmp_path / "project-alias"
    alias.symlink_to(project_a, target_is_directory=True)
    monkeypatch.chdir(alias)
    monkeypatch.setenv("PWD", str(project_b))
    result = invoke(home, monkeypatch, capsys, "forget", records["project_a"])
    assert result == {"removed": True}
    rows, fts = snapshot(db)
    assert records["project_a"] not in {row[0] for row in rows}
    assert records["project_a"] not in {row[0] for row in fts}
    assert len(rows) == len(fts) == 4


def test_cli_forget_denies_crossproject_without_disclosing_row(cli_home, monkeypatch, capsys):
    home, db, project_a, project_b, records = cli_home
    monkeypatch.setenv("PWD", str(project_b))
    before = snapshot(db)
    hidden = invoke(home, monkeypatch, capsys, "forget", records["project_b"])
    missing = invoke(home, monkeypatch, capsys, "forget", max(records.values()) + 100)
    assert hidden == missing == {"removed": False}
    assert snapshot(db) == before


@pytest.mark.parametrize("forged_session_env", [False, True])
def test_cli_forget_session_fails_closed_without_runtime_authority(
    cli_home, monkeypatch, capsys, forged_session_env,
):
    home, db, project_a, project_b, records = cli_home
    if forged_session_env:
        monkeypatch.setenv("HERMES_SESSION_ID", "session-a")
    before = snapshot(db)
    hidden = invoke(home, monkeypatch, capsys, "forget", records["session"])
    missing = invoke(home, monkeypatch, capsys, "forget", max(records.values()) + 100)
    assert hidden == missing == {"removed": False}
    assert snapshot(db) == before


def test_cli_search_uses_current_cwd_and_has_no_session_authority(cli_home, monkeypatch, capsys):
    home, db, project_a, project_b, records = cli_home
    monkeypatch.setenv("PWD", str(project_b))
    monkeypatch.setenv("HERMES_SESSION_ID", "session-a")
    result = invoke(home, monkeypatch, capsys, "search", "copper approval", "--limit", 10)
    assert {row["id"] for row in result["results"]} == {records["project_a"], records["global"]}
    monkeypatch.chdir(project_b)
    result = invoke(home, monkeypatch, capsys, "search", "copper approval", "--limit", 10)
    assert {row["id"] for row in result["results"]} == {records["project_b"], records["global"]}


def test_cli_forget_preserves_global_memory_behavior(cli_home, monkeypatch, capsys):
    home, db, project_a, project_b, records = cli_home
    assert invoke(home, monkeypatch, capsys, "forget", records["global"]) == {"removed": True}


def test_cli_forget_denies_other_namespace(cli_home, monkeypatch, capsys):
    home, db, project_a, project_b, records = cli_home
    before = snapshot(db)
    assert invoke(home, monkeypatch, capsys, "forget", records["other_namespace"]) == {"removed": False}
    assert snapshot(db) == before

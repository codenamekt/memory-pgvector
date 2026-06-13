"""tests/test_migration.py — schema migration contract tests.

The migration is the contract between the plugin code and the operator's
Postgres install. These tests pin the dim, the column types, the
indexes, and the idempotency so a future change that breaks any of
those things fails loudly here, not in production.

Tests run against a live PG_TEST_DSN. With no DSN set, all tests skip.
The migration SQL is the source of truth; we re-read the file and
apply it via psql so a drift between the .sql file and what we test
is impossible.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent / "hexus" / "migrations" / "001_schema.sql"
)


def _dsn_kwargs() -> dict:
    """Return env-style kwargs the rest of the file can splat into
    subprocess calls. Returns empty if PG_TEST_DSN is unset."""
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        return {"_skip": True}
    return {"dsn": dsn}


def _psql_env(dsn: str) -> dict:
    """Return env dict with PGPASSWORD extracted from the DSN if present.

    Keeps callers from having to parse the DSN themselves.
    """
    env = os.environ.copy()
    # Crude extraction: pgpass-style file path or password= param.
    if "password=" in dsn:
        # Convert "key=value" pairs into env vars
        for part in dsn.split():
            if "=" in part:
                k, v = part.split("=", 1)
                if k.lower() == "password":
                    env["PGPASSWORD"] = v
    return env


def _run_psql(sql: str, dsn: str) -> str:
    """Execute SQL via psql, return stdout. Raises on non-zero exit."""
    result = subprocess.run(
        ["psql", dsn, "-tAc", sql],
        capture_output=True, text=True, env=_psql_env(dsn), check=True,
    )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------

@pytest.fixture
def pg():
    """Skip-everywhere helper: skip the whole module if PG_TEST_DSN unset."""
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        pytest.skip("PG_TEST_DSN not set")
    return dsn


def test_migration_file_exists():
    """Sanity: the .sql file is where we expect it. A missing migration
    is the most common reason these tests fail elsewhere."""
    assert MIGRATION_PATH.exists(), f"migration file missing at {MIGRATION_PATH}"
    assert MIGRATION_PATH.is_file()
    assert MIGRATION_PATH.stat().st_size > 0


def test_migration_is_idempotent(pg):
    """Running the migration twice in a row succeeds. The second run
    hits every CREATE IF NOT EXISTS, every ADD CONSTRAINT IF NOT EXISTS,
    and is a no-op. This is the operator's safety net."""
    sql = MIGRATION_PATH.read_text(encoding="utf-8")
    # First run — may or may not be a no-op depending on whether
    # something created the tables already. Either way must succeed.
    _run_psql(sql, pg)
    # Second run — must be a clean no-op.
    _run_psql(sql, pg)


def test_memory_entries_dim_is_384(pg):
    """The embedding column must be vector(384) — matches the BERT model."""
    out = _run_psql(
        "SELECT format_type(atttypid, atttypmod) "
        "FROM pg_attribute "
        "WHERE attrelid = 'memory_entries'::regclass "
        "  AND attname = 'embedding'",
        pg,
    )
    assert out == "vector(384)", f"expected vector(384), got {out!r}"


def test_conversations_dim_is_384(pg):
    out = _run_psql(
        "SELECT format_type(atttypid, atttypmod) "
        "FROM pg_attribute "
        "WHERE attrelid = 'conversations'::regclass "
        "  AND attname = 'embedding'",
        pg,
    )
    assert out == "vector(384)", f"expected vector(384), got {out!r}"


def test_hnsw_index_on_memory_entries(pg):
    """The HNSW index is the recall hot path; verify it's there and
    uses the cosine distance operator class."""
    out = _run_psql(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = 'public' AND tablename = 'memory_entries' "
        "  AND indexname = 'ix_memory_entries_embedding_hnsw'",
        pg,
    )
    assert out, "HNSW index on memory_entries is missing"
    assert "using hnsw" in out.lower()
    assert "vector_cosine_ops" in out


def test_hnsw_index_on_conversations(pg):
    out = _run_psql(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = 'public' AND tablename = 'conversations' "
        "  AND indexname = 'ix_conversations_embedding_hnsw'",
        pg,
    )
    assert out, "HNSW index on conversations is missing"
    assert "using hnsw" in out.lower()
    assert "vector_cosine_ops" in out


def test_unique_constraint_on_memory_entries(pg):
    """The (agent_identity, target, content) unique constraint is what
    lets MemoryStore.add() do ON CONFLICT DO NOTHING for built-in
    dedup. Drop it and dedup silently breaks."""
    out = _run_psql(
        "SELECT conname FROM pg_constraint "
        "WHERE conrelid = 'memory_entries'::regclass "
        "  AND contype = 'u'",
        pg,
    )
    assert "memory_entries_unique" in out, f"unique constraint missing, got {out!r}"


def test_target_check_constraint(pg):
    """The CHECK constraint on target is what prevents `target='junk'`
    rows from leaking in. Belt-and-suspenders test."""
    out = _run_psql(
        "SELECT pg_get_constraintdef(oid) "
        "FROM pg_constraint "
        "WHERE conrelid = 'memory_entries'::regclass "
        "  AND contype = 'c'",
        pg,
    )
    assert "memory" in out and "user" in out, f"CHECK constraint missing or wrong: {out!r}"


def test_insert_384_dim_vector_works(pg):
    """Round-trip: insert a real 384-dim vector and read it back.
    Uses a temporary row to avoid polluting the table for the rest of
    the suite (the smoke tests clean up by agent_identity; this row
    has a unique random agent)."""
    import secrets
    agent = f"migration-test-{secrets.token_hex(4)}"
    try:
        # Build a 384-dim vector literal: "[0.1,0.1,...,0.1]"
        vec = "[" + ",".join(["0.1"] * 384) + "]"
        _run_psql(
            f"INSERT INTO memory_entries (agent_identity, target, content, embedding) "
            f"VALUES ('{agent}', 'memory', 'dim check', '{vec}'::vector)",
            pg,
        )
        out = _run_psql(
            f"SELECT vector_dims(embedding) FROM memory_entries "
            f"WHERE agent_identity = '{agent}'",
            pg,
        )
        assert out == "384", f"vector_dims returned {out!r}, expected 384"
    finally:
        _run_psql(
            f"DELETE FROM memory_entries WHERE agent_identity = '{agent}'", pg,
        )


def test_insert_768_dim_vector_rejected(pg):
    """If someone tries to insert a 768-dim vector (old shape), the DB
    must reject it. This is the safety net that the embed layer's
    dim check provides — even if a buggy embedder produces 768-dim
    vectors, the DB says no."""
    import secrets
    agent = f"migration-test-{secrets.token_hex(4)}"
    vec = "[" + ",".join(["0.1"] * 768) + "]"
    # Use a query that we expect to fail; capture the exit code.
    result = subprocess.run(
        ["psql", pg, "-tAc",
         f"INSERT INTO memory_entries (agent_identity, target, content, embedding) "
         f"VALUES ('{agent}', 'memory', 'dim mismatch test', '{vec}'::vector)"],
        capture_output=True, text=True, env=_psql_env(pg),
    )
    assert result.returncode != 0, "DB accepted a 768-dim vector; expected rejection"
    # Make sure the row didn't sneak in.
    rows = _run_psql(
        f"SELECT count(*) FROM memory_entries WHERE agent_identity = '{agent}'",
        pg,
    )
    assert rows == "0", f"row leaked despite rejection: {rows} rows present"

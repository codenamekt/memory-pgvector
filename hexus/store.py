"""store.py — Postgres ops for the hexus memory plugin.
#
# Forked from andreab67/hermes-hexus (BSD-3-Clause).
#
# Wraps psycopg3 + psycopg_pool. Mirrors hermes-agent's native built-in
# memory model (`memory` tool's add/replace/remove on targets 'memory' /
# 'user') into a single Postgres table with embeddings.
#
# Uses a small ConnectionPool because the plugin is touched from two
# threads at runtime: the agent thread (for prefetch / recall_memory /
# ensure_schema / health) and the async-writer drain thread (for the
# mirrored INSERTs / UPDATEs / DELETEs). Pooling beats short-lived
# connections under that two-thread pattern without adding much
# complexity.
#
# No SQLAlchemy, no LLM-mediated workers, no deriver loops.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

try:
    from .embed import to_hexus_literal
except ImportError:
    from embed import to_hexus_literal

logger = logging.getLogger(__name__)


class MemoryStore:
    """Postgres-backed mirror of hermes-agent's built-in memory entries."""

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 0,
        max_size: int = 4,
        timeout: float = 5.0,
        max_idle: float = 30.0,
        max_lifetime: float = 300.0,
    ):
        """Open a lazily-initialized, self-draining ConnectionPool.

        min_size=0 means an idle pool holds ZERO connections — critical so
        a pool that gets abandoned (a re-initialized provider, or a session
        the gateway never explicitly shuts down) cannot strand a warm
        backend in Postgres until the server's idle_session_timeout reaps
        it. Under load the pool still grows to max_size=4 so the agent
        thread and the async-writer drain thread can overlap.

        max_idle (30s) closes connections returned to the pool that then sit
        unused, shrinking back toward min_size. max_lifetime (300s) caps the
        absolute age of any pooled connection. Together these keep the
        connections "short-lived when idle, pooled under load" and bound the
        plugin's Postgres footprint to actual concurrent demand rather than
        to the number of sessions ever opened.
        """
        self._dsn = dsn
        self._lock = threading.Lock()
        self._pool: Optional[ConnectionPool] = None
        self._min_size = min_size
        self._max_size = max_size
        self._timeout = timeout
        self._max_idle = max_idle
        self._max_lifetime = max_lifetime

    # -- Pool lifecycle ------------------------------------------------------

    def _get_pool(self) -> ConnectionPool:
        """Return the live pool, constructing it on first call. Thread-safe."""
        if self._pool is not None:
            return self._pool
        with self._lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    conninfo=self._dsn,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    timeout=self._timeout,
                    max_idle=self._max_idle,
                    max_lifetime=self._max_lifetime,
                    open=True,
                    name="hexus-memory",
                )
        return self._pool

    def close(self) -> None:
        """Close the connection pool. Idempotent."""
        with self._lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("hexus pool close: %s", exc)
                finally:
                    self._pool = None

    # -- Schema --------------------------------------------------------------

    class SchemaNotApplied(RuntimeError):
        """Raised when memory_entries does not exist in the target DB."""

    def ensure_schema(self) -> None:
        """Verify the schema is in place. Does NOT run DDL."""
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('memory_entries')")
                if cur.fetchone()[0] is None:
                    raise self.SchemaNotApplied(
                        "memory_entries table missing. Apply the migration as DB admin: "
                        "psql -d <dbname> -f plugins/memory/hexus/migrations/001_schema.sql"
                    )
                cur.execute("SELECT to_regclass('delegations')")
                if cur.fetchone()[0] is None:
                    raise self.SchemaNotApplied(
                        "delegations table missing. Apply the migration as DB admin: "
                        "psql -d <dbname> -f plugins/memory/hexus/migrations/002_observability.sql"
                    )

    def apply_migration_as_admin(self, *, admin_dsn: str) -> None:
        """One-shot admin path: run the full migrations with privileged creds."""
        migrations_dir = Path(__file__).parent / "migrations"
        sql_files = sorted(migrations_dir.glob("*.sql"))
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                for sql_file in sql_files:
                    sql = sql_file.read_text(encoding="utf-8")
                    cur.execute(sql)

    # -- Built-in memory mirror (called by on_memory_write) ------------------

    def add(
        self,
        *,
        agent_identity: str,
        target: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Insert a memory entry. Returns row id, or None if duplicate (no-op).

        Matches the built-in tool's "reject exact duplicate" semantics via
        the (agent_identity, target, content) unique constraint + ON CONFLICT.
        """
        meta_json = json.dumps(metadata or {})
        vec_literal = to_hexus_literal(embedding) if embedding is not None else None

        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO memory_entries
                        (agent_identity, target, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                    ON CONFLICT (agent_identity, target, content) DO NOTHING
                    RETURNING id
                    """,
                    (agent_identity, target, content, vec_literal, meta_json),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0]) if row else None

    def replace(
        self,
        *,
        agent_identity: str,
        target: str,
        old_text: str,
        new_content: str,
        new_embedding: Optional[List[float]] = None,
    ) -> int:
        """Update entries in (agent_identity, target) where content contains old_text.

        Matches built-in semantics — old_text is a substring match. Returns
        the number of rows updated (built-in updates the FIRST match; we
        update all matches in the same scope for safety).
        """
        vec_literal = (
            to_hexus_literal(new_embedding) if new_embedding is not None else None
        )
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE memory_entries
                       SET content    = %s,
                           embedding  = %s::vector,
                           updated_at = now()
                     WHERE agent_identity = %s
                       AND target = %s
                       AND content LIKE %s
                    """,
                    (new_content, vec_literal, agent_identity, target, f"%{old_text}%"),
                )
                updated = cur.rowcount
                conn.commit()
                return int(updated)

    def remove(
        self,
        *,
        agent_identity: str,
        target: str,
        old_text: str,
    ) -> int:
        """Delete entries in (agent_identity, target) matching old_text substring.

        Returns the number of rows deleted.
        """
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM memory_entries
                     WHERE agent_identity = %s
                       AND target = %s
                       AND content LIKE %s
                    """,
                    (agent_identity, target, f"%{old_text}%"),
                )
                deleted = cur.rowcount
                conn.commit()
                return int(deleted)

    # -- Reads ---------------------------------------------------------------

    def list_entries(
        self,
        *,
        agent_identity: str,
        target: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List entries in an agent's scope. If target is None, both stores."""
        params: List[Any] = [agent_identity]
        target_clause = ""
        if target:
            target_clause = "AND target = %s"
            params.append(target)
        params.append(limit)
        params.append(offset)

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT id, agent_identity, target, content, created_at, updated_at, metadata
                    FROM memory_entries
                    WHERE agent_identity = %s
                    {target_clause}
                    ORDER BY updated_at DESC
                    LIMIT %s
                    OFFSET %s
                    """,
                    params,
                )
                return list(cur.fetchall())

    def search(
        self,
        *,
        query_embedding: List[float],
        agent_identity: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic recall via cosine distance.

        agent_identity=None → search across ALL agents (cross-theme recall).
        target=None → search both 'memory' and 'user'.
        Returns rows with `score` = 1 - cosine_distance ∈ [0, 1].
        """
        vec_literal = to_hexus_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if target:
            clauses.append("target = %s")
            params.append(target)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT id, agent_identity, target, content, created_at,
                           updated_at, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM memory_entries
                    {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [vec_literal, *params, vec_literal, limit],
                )
                rows = list(cur.fetchall())

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)

        if decay_half_life_days > 0.0 or recall_boost_weight > 0.0:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("memory_entries", [r["id"] for r in rows])

        return rows

    def hybrid_search(
        self,
        *,
        query_embedding: List[float],
        query_text: str,
        agent_identity: Optional[str] = None,
        target: Optional[str] = None,
        limit: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Blend semantic vector search and full-text search."""
        if not query_text or not query_text.strip():
            rows = self.search(
                query_embedding=query_embedding,
                agent_identity=agent_identity,
                target=target,
                limit=limit,
                min_similarity=min_similarity,
                decay_half_life_days=decay_half_life_days,
                recall_boost_weight=recall_boost_weight,
                platform=platform,
            )
            for r in rows:
                r["vector_score"] = r.get("vector_score", r.get("score"))
                r["text_score"] = 0.0
            return rows

        vec_literal = to_hexus_literal(query_embedding)
        
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if target:
            clauses.append("target = %s")
            params.append(target)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
            
        where = ("AND " + " AND ".join(clauses)) if clauses else ""

        sql = f"""
        WITH vector_search AS (
            SELECT id, agent_identity, target, content, created_at, updated_at, metadata,
                   1 - (embedding <=> %s::vector) AS vector_score
            FROM memory_entries
            WHERE 1=1 {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        text_search AS (
            SELECT id, agent_identity, target, content, created_at, updated_at, metadata,
                   ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', %s)) AS text_score
            FROM memory_entries
            WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
              {where}
            ORDER BY text_score DESC
            LIMIT %s
        )
        SELECT COALESCE(v.id, t.id) AS id,
               COALESCE(v.agent_identity, t.agent_identity) AS agent_identity,
               COALESCE(v.target, t.target) AS target,
               COALESCE(v.content, t.content) AS content,
               COALESCE(v.created_at, t.created_at) AS created_at,
               COALESCE(v.updated_at, t.updated_at) AS updated_at,
               COALESCE(v.metadata, t.metadata) AS metadata,
               COALESCE(v.vector_score, 0.0) AS vector_score,
               COALESCE(t.text_score, 0.0) AS text_score,
               (%s * COALESCE(v.vector_score, 0.0)) + (%s * COALESCE(t.text_score, 0.0)) AS score
        FROM vector_search v
        FULL OUTER JOIN text_search t ON v.id = t.id
        ORDER BY score DESC
        LIMIT %s
        """

        v_params = [vec_literal]
        for p in params:
            v_params.append(p)
        v_params.extend([vec_literal, limit])
        
        t_params = [query_text, query_text] + params + [limit]
        
        all_params = v_params + t_params + [vector_weight, text_weight, limit]

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, all_params)
                rows = list(cur.fetchall())

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)

        if decay_half_life_days > 0.0 or recall_boost_weight > 0.0:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("memory_entries", [r["id"] for r in rows])

        return rows

    # -- Bulk import from MEMORY.md / USER.md (v0.1.1) ----------------------

    # Matches tools/memory_tool.py:ENTRY_DELIMITER. Keep in sync if upstream
    # ever changes it (currently stable; been "\n§\n" since the tool shipped).
    ENTRY_DELIMITER = "\n§\n"

    def bulk_upsert_md(
        self,
        *,
        agent_identity: str,
        target: str,
        file_path: "Path | str",
        embed_fn,
    ) -> Dict[str, int]:
        """Parse a MEMORY.md / USER.md file and upsert each entry.

        Idempotent + cheap on re-run: we SELECT the existing content set
        for (agent_identity, target) once, then only embed + INSERT new
        entries. So initial install embeds everything; subsequent inits
        with no MD changes do zero embed calls.

        embed_fn is a callable taking a string and returning a 768-dim
        list (or raising — we catch and store text-only). Wired by the
        caller so the plugin can pass its `embed()` with the configured
        base_url + model.

        Returns: {'parsed': N, 'inserted': M, 'skipped': K} where N=M+K.
        """
        from pathlib import Path as _Path
        p = _Path(file_path)
        if not p.exists():
            return {"parsed": 0, "inserted": 0, "skipped": 0}

        raw = p.read_text(encoding="utf-8", errors="replace")
        entries = [e.strip() for e in raw.split(self.ENTRY_DELIMITER) if e.strip()]
        if not entries:
            return {"parsed": 0, "inserted": 0, "skipped": 0}

        # Single bulk SELECT of existing content for this scope. Beats N+1
        # by a wide margin and keeps re-init nearly free.
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM memory_entries WHERE agent_identity = %s AND target = %s",
                    (agent_identity, target),
                )
                existing = {row[0] for row in cur.fetchall()}

        inserted = 0
        skipped = 0
        for entry in entries:
            if entry in existing:
                skipped += 1
                continue
            vec = None
            try:
                vec = embed_fn(entry) if embed_fn else None
            except Exception:  # noqa: BLE001 — fail-soft on bulk embed
                vec = None
            row_id = self.add(
                agent_identity=agent_identity,
                target=target,
                content=entry,
                embedding=vec,
                metadata={"source": "bulk_import", "file": str(p)},
            )
            if row_id is not None:
                inserted += 1
            else:
                # Lost a race with another writer that inserted the same row.
                skipped += 1
        return {"parsed": len(entries), "inserted": inserted, "skipped": skipped}

    # -- Conversation turns (v0.2) ------------------------------------------

    def append_turn(
        self,
        *,
        session_id: str,
        agent_identity: str,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert one chat turn. Returns row id.

        No dedup (turns are inherently time-ordered events — same content
        twice is two distinct turns, even verbatim).
        """
        meta_json = json.dumps(metadata or {})
        vec_literal = to_hexus_literal(embedding) if embedding is not None else None

        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations
                        (session_id, agent_identity, role, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)
                    RETURNING id
                    """,
                    (session_id, agent_identity, role, content, vec_literal, meta_json),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0])

    def search_turns(
        self,
        *,
        query_embedding: List[float],
        agent_identity: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic recall over conversation turns. Same shape as `search()`."""
        vec_literal = to_hexus_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT id, session_id, agent_identity, role, content, ts, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM conversations
                    {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [vec_literal, *params, vec_literal, limit],
                )
                rows = list(cur.fetchall())

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)

        if decay_half_life_days > 0.0 or recall_boost_weight > 0.0:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("conversations", [r["id"] for r in rows])

        return rows

    def hybrid_search_turns(
        self,
        *,
        query_embedding: List[float],
        query_text: str,
        agent_identity: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Blend semantic vector search and full-text search over conversation turns."""
        if not query_text or not query_text.strip():
            rows = self.search_turns(
                query_embedding=query_embedding,
                agent_identity=agent_identity,
                session_id=session_id,
                limit=limit,
                min_similarity=min_similarity,
                decay_half_life_days=decay_half_life_days,
                recall_boost_weight=recall_boost_weight,
                platform=platform,
            )
            for r in rows:
                r["vector_score"] = r.get("vector_score", r.get("score"))
                r["text_score"] = 0.0
            return rows

        vec_literal = to_hexus_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
            
        where = ("AND " + " AND ".join(clauses)) if clauses else ""

        sql = f"""
        WITH vector_search AS (
            SELECT id, session_id, agent_identity, role, content, ts, metadata,
                   1 - (embedding <=> %s::vector) AS vector_score
            FROM conversations
            WHERE 1=1 {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        ),
        text_search AS (
            SELECT id, session_id, agent_identity, role, content, ts, metadata,
                   ts_rank(to_tsvector('english', content), websearch_to_tsquery('english', %s)) AS text_score
            FROM conversations
            WHERE to_tsvector('english', content) @@ websearch_to_tsquery('english', %s)
              {where}
            ORDER BY text_score DESC
            LIMIT %s
        )
        SELECT COALESCE(v.id, t.id) AS id,
               COALESCE(v.session_id, t.session_id) AS session_id,
               COALESCE(v.agent_identity, t.agent_identity) AS agent_identity,
               COALESCE(v.role, t.role) AS role,
               COALESCE(v.content, t.content) AS content,
               COALESCE(v.ts, t.ts) AS ts,
               COALESCE(v.metadata, t.metadata) AS metadata,
               COALESCE(v.vector_score, 0.0) AS vector_score,
               COALESCE(t.text_score, 0.0) AS text_score,
               (%s * COALESCE(v.vector_score, 0.0)) + (%s * COALESCE(t.text_score, 0.0)) AS score
        FROM vector_search v
        FULL OUTER JOIN text_search t ON v.id = t.id
        ORDER BY score DESC
        LIMIT %s
        """

        v_params = [vec_literal]
        for p in params:
            v_params.append(p)
        v_params.extend([vec_literal, limit])
        
        t_params = [query_text, query_text] + params + [limit]
        
        all_params = v_params + t_params + [vector_weight, text_weight, limit]

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, all_params)
                rows = list(cur.fetchall())

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)

        if decay_half_life_days > 0.0 or recall_boost_weight > 0.0:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("conversations", [r["id"] for r in rows])

        return rows

    def record_delegation(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        agent_identity: str = "default",
        task: str,
        result: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a delegation entry. Returns row id."""
        vec_literal = to_hexus_literal(embedding) if embedding is not None else None
        meta_json = json.dumps(metadata or {})
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO delegations
                        (parent_session_id, child_session_id, agent_identity, task, result, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s::jsonb)
                    RETURNING id
                    """,
                    (parent_session_id, child_session_id, agent_identity, task, result, vec_literal, meta_json),
                )
                row = cur.fetchone()
                conn.commit()
                return int(row[0])

    def search_delegations(
        self,
        *,
        query_embedding: List[float],
        agent_identity: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
        platform: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic recall over delegations."""
        vec_literal = to_hexus_literal(query_embedding)
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if parent_session_id:
            clauses.append("parent_session_id = %s")
            params.append(parent_session_id)
        if platform:
            clauses.append("metadata->>'platform' = %s")
            params.append(platform)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._get_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    f"""
                    SELECT id, parent_session_id, child_session_id, agent_identity, task, result, ts, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM delegations
                    {where}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    [vec_literal, *params, vec_literal, limit],
                )
                rows = list(cur.fetchall())

        # Apply boost & decay
        rows = self._apply_recall_boost(rows, recall_boost_weight)
        rows = self._apply_temporal_decay(rows, decay_half_life_days)

        if decay_half_life_days > 0.0 or recall_boost_weight > 0.0:
            rows = sorted(rows, key=lambda r: r.get("score", 0.0), reverse=True)

        if min_similarity > 0:
            rows = [r for r in rows if (r.get("score") or 0) >= min_similarity]

        if rows:
            self.increment_recall_counts("delegations", [r["id"] for r in rows])

        return rows

    def cleanup_stale_records(
        self,
        *,
        conversations_ttl_days: Optional[int] = None,
        memories_ttl_days: Optional[int] = None,
        delegations_ttl_days: Optional[int] = None,
    ) -> Dict[str, int]:
        """Delete records older than the specified TTL. Returns counts of deleted items."""
        deleted = {"conversations": 0, "memory_entries": 0, "delegations": 0}
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                if conversations_ttl_days is not None and conversations_ttl_days > 0:
                    limit_date = datetime.now(timezone.utc) - timedelta(days=conversations_ttl_days)
                    cur.execute("DELETE FROM conversations WHERE ts < %s", (limit_date,))
                    deleted["conversations"] = cur.rowcount
                if memories_ttl_days is not None and memories_ttl_days > 0:
                    limit_date = datetime.now(timezone.utc) - timedelta(days=memories_ttl_days)
                    cur.execute("DELETE FROM memory_entries WHERE updated_at < %s", (limit_date,))
                    deleted["memory_entries"] = cur.rowcount
                if delegations_ttl_days is not None and delegations_ttl_days > 0:
                    limit_date = datetime.now(timezone.utc) - timedelta(days=delegations_ttl_days)
                    cur.execute("DELETE FROM delegations WHERE ts < %s", (limit_date,))
                    deleted["delegations"] = cur.rowcount
                conn.commit()
        return deleted

    # -- Maintenance ---------------------------------------------------------

    def count_turns(
        self,
        *,
        agent_identity: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if session_id:
            clauses.append("session_id = %s")
            params.append(session_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM conversations {where}", params)
                return int(cur.fetchone()[0])

    def count(
        self,
        *,
        agent_identity: Optional[str] = None,
        target: Optional[str] = None,
    ) -> int:
        clauses: List[str] = []
        params: List[Any] = []
        if agent_identity:
            clauses.append("agent_identity = %s")
            params.append(agent_identity)
        if target:
            clauses.append("target = %s")
            params.append(target)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM memory_entries {where}", params)
                return int(cur.fetchone()[0])

    def health(self) -> Dict[str, Any]:
        """Liveness probe — pool reachable + table exists. Never raises."""
        try:
            with self._get_pool().connection(timeout=3.0) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('memory_entries') IS NOT NULL")
                    has_table = bool(cur.fetchone()[0])
                    if not has_table:
                        return {"ok": False, "error": "memory_entries table missing", "row_count": 0}
                    cur.execute("SELECT COUNT(*) FROM memory_entries")
                    return {"ok": True, "error": "", "row_count": int(cur.fetchone()[0])}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:200], "row_count": 0}

    def _apply_recall_boost(self, rows: List[Dict[str, Any]], boost_weight: float) -> List[Dict[str, Any]]:
        if boost_weight <= 0.0:
            return rows
        for r in rows:
            meta = r.get("metadata") or {}
            try:
                recall_count = int(meta.get("recall_count", 0))
            except (ValueError, TypeError):
                recall_count = 0
            # log-based boost: score * (1 + boost_weight * log(1 + recall_count))
            r["score"] = r["score"] * (1.0 + boost_weight * math.log(1 + recall_count))
        return rows

    def _apply_temporal_decay(self, rows: List[Dict[str, Any]], half_life_days: float) -> List[Dict[str, Any]]:
        if half_life_days <= 0.0:
            return rows
        now = datetime.now(timezone.utc)
        for r in rows:
            ts_val = r.get("updated_at") or r.get("ts") or r.get("created_at")
            if isinstance(ts_val, str):
                try:
                    from datetime import datetime as dt
                    ts_val = dt.fromisoformat(ts_val)
                except Exception:
                    continue
            if not ts_val:
                continue
            
            # Ensure ts_val has timezone info (psycopg datetimes are timezone-aware, now is utc)
            if ts_val.tzinfo is None:
                ts_val = ts_val.replace(tzinfo=timezone.utc)
                
            age_days = (now - ts_val).total_seconds() / 86400.0
            # exponential decay: score * 2^(-age/half_life)
            decay = math.exp(-math.log(2.0) * age_days / half_life_days)
            r["score"] = r["score"] * decay
        return rows

    def increment_recall_counts(self, table: str, ids: List[int]) -> None:
        if not ids:
            return
        sql = f"""
        UPDATE {table}
        SET metadata = jsonb_set(
            metadata, 
            '{{recall_count}}', 
            (COALESCE(metadata->>'recall_count', '0')::int + 1)::text::jsonb
        )
        WHERE id = ANY(%s)
        """
        try:
            with self._get_pool().connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (ids,))
                    conn.commit()
        except Exception as exc:
            logger.warning("Failed to increment recall counts for %s: %s", table, exc)

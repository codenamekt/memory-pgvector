"""mcp_server.server — FastMCP wiring for hexus.

Forked from andreab67/hermes-hexus (BSD-3-Clause).

Each `@server.tool()` registers one of the pure functions in `tools.py`
as an MCP tool. The MCP transport (stdio or streamable-http) is selected
at run() time by `cli.py`.

Multi-agent: the `agent_identity` parameter on every write/read tool
keeps each connected client's data isolated. The server process can be
the same for N agents — agent isolation lives in the DB, not the
process. One model load (the LocalBertEmbedder singleton) is shared
across all of them.

The server has no opinion on transport, so the same FastMCP instance
works with:
  - `mcp.run(transport='stdio')`             for Claude Desktop / Cursor
  - `mcp.run(transport='streamable-http')`  for fleet use
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from hexus.store import MemoryStore

from . import tools

logger = logging.getLogger(__name__)


def _build_server(
    store: MemoryStore,
    *,
    name: str = "hexus",
    instructions: Optional[str] = None,
):
    """Build and return a configured `mcp.server.fastmcp.FastMCP` instance.

    The server is wired to the supplied `MemoryStore` (closed-over into
    each tool handler). Reuses the `LocalBertEmbedder` singleton if any
    of the tools trigger an embed — the first embed() call loads the
    model into the process; subsequent calls reuse it.
    """
    # Imported lazily so `pip install hexus` (no [mcp] extra)
    # doesn't pull mcp as a transitive runtime dep.
    from mcp.server.fastmcp import FastMCP

    if instructions is None:
        instructions = (
            "hexus exposes a Postgres + hexus shared knowledge "
            "base as MCP tools. All tools take an optional `agent_identity` "
            "argument that scopes writes/reads — every connected agent is "
            "isolated by default, and passes can use `agent_identity=None` "
            "(or omit it) on `memory_recall` / `memory_search` to query "
            "across all agents. Embeddings are produced locally by "
            "sentence-transformers MiniLM-L6-v2 (384-dim, no network)."
        )

    mcp = FastMCP(name=name, instructions=instructions, host="0.0.0.0", port=8000)

    # -- tools -------------------------------------------------------------

    @mcp.tool()
    def memory_health() -> Dict[str, Any]:
        """Liveness + capability check. Returns DB status, embedder model/dim, row counts."""
        return tools.memory_health(store, {})

    @mcp.tool()
    def memory_retain(
        contents: list[str],
        target: str = "memory",
        agent_identity: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        doc_type: str = "memory",
        source_url: str = "",
    ) -> Dict[str, Any]:
        """Add one or many memory entries. Each content becomes one row.

        Args:
          contents: list of non-empty strings, one per row.
          target: 'memory' (default — the agent's MEMORY.md mirror) or
                  'user' (the agent's USER.md mirror). Omit to default
                  to 'memory'.
          agent_identity: which agent's scope to write into. Defaults to
                          the env var HEXUS_AGENT_IDENTITY, then
                          'default'. Pick a stable lowercase-dashed name
                          per agent (e.g. 'marketing', 'sales',
                          'intraday-trading').
          metadata: optional dict applied to every row, or a list of
                    dicts (one per content) for per-item metadata. Each
                    dict is stored as JSONB alongside the content.
          doc_type: optional tag stored in metadata (default 'memory').
          source_url: optional URL stored in metadata['source_url'].

        Returns: {"inserted": N, "duplicates": K, "errors": [...]}
        """
        return tools.memory_retain(
            store,
            {
                "contents": contents,
                "target": target,
                "agent_identity": agent_identity,
                "metadata": metadata,
                "doc_type": doc_type,
                "source_url": source_url,
            },
        )

    @mcp.tool()
    def memory_recall(
        query: str,
        top_k: int = 5,
        agent_identity: str = "",
        target: str = "",
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """Semantic search over memory entries.

        Args:
          query: the natural-language search query.
          top_k: 1..100, default 5.
          agent_identity: scope to one agent, or empty / None to search
                          across every agent in the store.
          target: 'memory' | 'user' | '' (both).
          min_similarity: 0..1, default 0. Filter out lower-scored hits.

        Returns: {"query", "count", "results": [{id, agent_identity, target,
                                                  content, score, metadata, ...}]}
        """
        return tools.memory_recall(
            store,
            {
                "query": query,
                "top_k": top_k,
                "agent_identity": agent_identity,
                "target": target,
                "min_similarity": min_similarity,
            },
        )

    @mcp.tool()
    def memory_hybrid_search(
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        agent_identity: str = "",
        target: str = "",
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """Hybrid search blending semantic vector search and full-text search over memory entries.

        Args:
          query: the natural-language search query.
          top_k: 1..100, default 5.
          vector_weight: weight for semantic similarity (0..1, default 0.7).
          text_weight: weight for full-text search rank (0..1, default 0.3).
          agent_identity: scope to one agent, or empty / None to search all.
          target: 'memory' | 'user' | '' (both).
          min_similarity: 0..1, default 0. Filter out lower-scored hits.

        Returns: {"query", "count", "results": [{id, agent_identity, target,
                                                  content, score, vector_score,
                                                  text_score, metadata, ...}]}
        """
        return tools.memory_hybrid_search(
            store,
            {
                "query": query,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "text_weight": text_weight,
                "agent_identity": agent_identity,
                "target": target,
                "min_similarity": min_similarity,
            },
        )

    @mcp.tool()
    def memory_search(
        agent_identity: str = "",
        target: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Browse memory entries without semantic search (list / paginate).

        Returns: {"count", "limit", "offset", "rows": [...]}
        """
        return tools.memory_search(
            store,
            {
                "agent_identity": agent_identity,
                "target": target,
                "limit": limit,
                "offset": offset,
            },
        )

    @mcp.tool()
    def memory_forget(
        id: int,
        confirm: bool = False,
        agent_identity: str = "",
    ) -> Dict[str, Any]:
        """Delete a memory entry by id. Pass confirm=true to actually delete.

        Dry-run by default (returns what would happen). Restricted to the
        caller's agent_identity scope — you can only delete rows you
        own.
        """
        return tools.memory_forget(
            store,
            {
                "id": id,
                "confirm": confirm,
                "agent_identity": agent_identity,
            },
        )

    @mcp.tool()
    def memory_recall_turns(
        query: str,
        top_k: int = 5,
        agent_identity: str = "",
        session_id: str = "",
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """Semantic search over past chat turns (every user/assistant exchange).

        Args:
          query: natural-language search.
          top_k: 1..100, default 5.
          agent_identity: scope to one agent, or '' / None to search all.
          session_id: optional — restrict to one session id.
          min_similarity: 0..1, default 0.

        Returns: {"query", "count", "results": [{id, session_id,
                                                  agent_identity, role,
                                                  content, score, ts, ...}]}
        """
        return tools.memory_recall_turns(
            store,
            {
                "query": query,
                "top_k": top_k,
                "agent_identity": agent_identity,
                "session_id": session_id,
                "min_similarity": min_similarity,
            },
        )

    @mcp.tool()
    def memory_hybrid_recall_turns(
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        agent_identity: str = "",
        session_id: str = "",
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """Hybrid search blending semantic vector search and full-text search over conversation turns.

        Args:
          query: natural-language search.
          top_k: 1..100, default 5.
          vector_weight: weight for semantic similarity (0..1, default 0.7).
          text_weight: weight for full-text search rank (0..1, default 0.3).
          agent_identity: scope to one agent, or '' / None to search all.
          session_id: optional — restrict to one session id.
          min_similarity: 0..1, default 0.

        Returns: {"query", "count", "results": [{id, session_id,
                                                  agent_identity, role,
                                                  content, score, vector_score,
                                                  text_score, ts, ...}]}
        """
        return tools.memory_hybrid_recall_turns(
            store,
            {
                "query": query,
                "top_k": top_k,
                "vector_weight": vector_weight,
                "text_weight": text_weight,
                "agent_identity": agent_identity,
                "session_id": session_id,
                "min_similarity": min_similarity,
            },
        )

    @mcp.tool()
    def memory_append_turn(
        session_id: str,
        role: str,
        content: str,
        agent_identity: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append one chat turn. Use this to capture a (user, assistant)
        exchange into the conversation log for later semantic recall.

        Args:
          session_id: a stable per-conversation id (e.g. UUID).
          role: 'user' | 'assistant' | 'system' | 'tool'.
          content: the turn text.
          agent_identity: which agent's log to append to.
          metadata: optional dict, stored as JSONB.

        Returns: {"id", "session_id", "role"}
        """
        return tools.memory_append_turn(
            store,
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "agent_identity": agent_identity,
                "metadata": metadata,
            },
        )

    @mcp.tool()
    def memory_record_delegation(
        parent_session_id: str,
        child_session_id: str,
        task: str,
        result: str,
        agent_identity: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a subagent delegation.

        Args:
          parent_session_id: parent session ID delegating the task.
          child_session_id: child session ID executing the task.
          task: task description prompt.
          result: task output response.
          agent_identity: agent identity/theme.
          metadata: optional dict metadata.

        Returns: {"id", "parent_session_id", "child_session_id", "agent_identity"}
        """
        return tools.memory_record_delegation(
            store,
            {
                "parent_session_id": parent_session_id,
                "child_session_id": child_session_id,
                "task": task,
                "result": result,
                "agent_identity": agent_identity,
                "metadata": metadata,
            },
        )

    @mcp.tool()
    def memory_recall_delegations(
        query: str,
        top_k: int = 5,
        agent_identity: str = "",
        parent_session_id: str = "",
        min_similarity: float = 0.0,
        decay_half_life_days: float = 0.0,
        recall_boost_weight: float = 0.0,
    ) -> Dict[str, Any]:
        """Recall subagent delegations by semantic similarity query.

        Args:
          query: natural-language search query.
          top_k: max results to return.
          agent_identity: scope search to specific agent theme.
          parent_session_id: scope search to specific parent session.
          min_similarity: similarity score threshold.
          decay_half_life_days: temporal decay half life parameter.
          recall_boost_weight: recall boost parameter.

        Returns: {"query", "count", "results": [{id, parent_session_id, child_session_id, agent_identity, task, result, score, ts, metadata}]}
        """
        return tools.memory_recall_delegations(
            store,
            {
                "query": query,
                "top_k": top_k,
                "agent_identity": agent_identity,
                "parent_session_id": parent_session_id,
                "min_similarity": min_similarity,
                "decay_half_life_days": decay_half_life_days,
                "recall_boost_weight": recall_boost_weight,
            },
        )

    @mcp.tool()
    def memory_count(
        agent_identity: str = "",
        target: str = "",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """Return row counts for memory_entries and conversations, scoped as requested.

        Args:
          agent_identity: default = env / 'default'.
          target: 'memory' | 'user' | '' (both).
          session_id: optional session id, restricts the conversation count.

        Returns: {"memory_entries": N, "conversations": M, ...}
        """
        return tools.memory_count(
            store,
            {
                "agent_identity": agent_identity,
                "target": target,
                "session_id": session_id,
            },
        )

    @mcp.tool()
    def memory_cleanup(
        conversations_ttl_days: Optional[int] = None,
        memories_ttl_days: Optional[int] = None,
        delegations_ttl_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Delete stale records from conversations, memory_entries, and delegations based on TTL.

        Args:
          conversations_ttl_days: delete conversations older than this many days (default None/disabled).
          memories_ttl_days: delete memory entries older than this many days (default None/disabled).
          delegations_ttl_days: delete delegations older than this many days (default None/disabled).

        Returns: {"status": "ok", "deleted": {"conversations": N, "memory_entries": M, "delegations": K}}
        """
        deleted = store.cleanup_stale_records(
            conversations_ttl_days=conversations_ttl_days,
            memories_ttl_days=memories_ttl_days,
            delegations_ttl_days=delegations_ttl_days,
        )
        return {"status": "ok", "deleted": deleted}

    @mcp.tool()
    def memory_metrics() -> str:
        """Return operational metrics in a Prometheus-compatible format."""
        health = store.health()
        m_entries = store.count(agent_identity=None, target=None)
        m_turns = store.count_turns(agent_identity=None)
        
        lines = [
            "# HELP hexus_db_reachable Liveness check for the Postgres database (1=ok, 0=error)",
            "# TYPE hexus_db_reachable gauge",
            f"hexus_db_reachable {1 if health.get('ok') else 0}",
            "# HELP hexus_memory_entries_total Total number of stored memory entries",
            "# TYPE hexus_memory_entries_total counter",
            f"hexus_memory_entries_total {m_entries}",
            "# HELP hexus_conversation_turns_total Total number of stored conversation turns",
            "# TYPE hexus_conversation_turns_total counter",
            f"hexus_conversation_turns_total {m_turns}",
        ]
        return "\n".join(lines)

    return mcp


def build_server(
    dsn: str,
    *,
    name: str = "hexus",
    instructions: Optional[str] = None,
) -> Any:
    """Build (but don't run) an MCP server for the given DSN.

    The MemoryStore is constructed lazily on first use; closing the
    server is the caller's job (or the process exit, which is fine for
    the typical stdio / one-shot http deployments).
    """
    store = MemoryStore(dsn)
    return _build_server(store, name=name, instructions=instructions)

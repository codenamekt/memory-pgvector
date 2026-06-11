"""mcp_server — Model Context Protocol server for memory-pgvector.

Forked from andreab67/hermes-memory-pgvector (BSD-3-Clause).

Exposes the same Postgres + pgvector memory store that the Hermes plugin
uses to any MCP client (Claude Desktop, Cursor, custom agents) as a set
of `memory_*` tools. One `MemoryStore` + one `LocalBertEmbedder` instance
is shared by every connected client per process — multiple agents pointing
at the same MCP server share the storage layer but each is scoped by the
`agent_identity` argument on every tool call (matching the plugin's
multi-tenant model).

Transports:
  - stdio           : the default; one process per client, easiest for
                      Claude Desktop / Cursor / editor integration
  - streamable-http : one shared process, N clients; for fleets of agents
                      that want a single canonical memory store

Install:  pip install "memory-pgvector[mcp]"
Run:      memory-pgvector-mcp serve --transport stdio --dsn "..."
          memory-pgvector-mcp serve --transport http  --host 0.0.0.0 --port 8000
"""

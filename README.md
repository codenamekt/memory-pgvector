# hexus

**Postgres + hexus memory substrate for [hermes-agent](https://github.com/NousResearch/hermes-agent) AND a standalone MCP server for any MCP client (Claude Desktop, Cursor, fleet agents).** A shared knowledge base for a fleet of cooperating agents — built on Postgres and **local sentence-transformers MiniLM-L6-v2** (no cloud, no LLM in the hot path).

```text
                                                ┌──────────────────────────────────┐
each minion → X-Hermes-Session-Key: <theme>     │  hexus                 │
            → hermes-agent gateway              │  (single process, one model)    │
            → hexus plugin   ──────────────► ├──────────────────────────────────┤
                                                │  ┌────────────────────────────┐  │
Claude Desktop ──── stdio MCP ──┐                │  │ LocalBertEmbedder          │  │
                              │                │  │ (sentence-transformers)    │  │
Cursor       ──── stdio MCP ──┼────► mcp_server │  └────────────────────────────┘  │
                              │       (mcp_)   │                                  │
custom agent ── http MCP  ────┘       server    │  ┌────────────────────────────┐  │
                                                │  │ MemoryStore (psycopg pool) │◄─┤
                                                │  │  memory_entries            │  │
                                                │  │  conversations             │  │
                                                │  └────────────────────────────┘  │
                                                │           │                      │
                                                │           ▼                      │
                                                │  Postgres 16 + hexus           │
                                                └──────────────────────────────────┘
```

## Two surfaces, one store

This repo ships two integration paths that share the exact same `MemoryStore` and the exact same `LocalBertEmbedder` instance per process:

1. **Hermes plugin** (`hexus/__init__.py`) — drop into `~/.hermes/plugins/hexus/`. Mirrors built-in `memory` writes, captures conversation turns, provides `recall_memory` + `recall_conversation` tools. Per-minion scoping via `X-Hermes-Session-Key`.
2. **MCP server** (`mcp_server/`) — `hexus-mcp serve --transport stdio|http`. Exposes the same store as eight MCP tools (`memory_retain`, `memory_recall`, `memory_search`, `memory_forget`, `memory_recall_turns`, `memory_append_turn`, `memory_count`, `memory_health`). Multi-agent: each connected client picks its own `agent_identity`.

The local BERT swap (sentence-transformers MiniLM-L6-v2, 384-dim, ~88MB on disk, <500MB RAM on CPU) replaces the upstream HTTP-embedder. The HTTP path is preserved as a fallback for operators with an existing Ollama / OpenAI-compatible endpoint.

## Why it exists

Existing memory providers each solve a piece of the problem; the gap for **fleet deployments** is wide:

- **Built-in `memory` tool** persists to per-host `MEMORY.md` / `USER.md`. Two minions on the same host stomp on each other; minions on different hosts have no shared substrate.
- **Honcho** offers cross-session user modelling but requires a full external service, an LLM in the memory hot path for its deriver + dialectic loops, and its own ontology layered on top of the built-in tool. In high-concurrency fleet use it produces retry storms, embedding-endpoint queue backups, and gateway↔Honcho circular dependencies.
- **Holographic** is a fine in-process fact store but uses SQLite — a poor fit for many minions writing concurrently from many hosts.
- **Other providers** (Mem0, Hindsight, OpenViking, ByteRover, RetainDB, Supermemory) all either require a paid cloud, require LLM mediation for memory ops, or both.

What was missing: a **storage layer** that gives the built-in `memory` model durable, multi-tenant, semantically-searchable backing, with no LLM in the hot path, scoped cleanly per-minion so a marketing agent's notes don't pollute a trading agent's recall. That's what this plugin provides.

## Design philosophy

1. **Storage layer, not a memory model.** The agent keeps using `memory(action='add', target='memory'|'user', …)`. We mirror those writes via `on_memory_write`. No new ontology for the agent to learn.
2. **No LLM in the memory hot path.** Embeddings are vector math, not LLM calls. There is no deriver, no dialectic, no dream cycle — the failure modes that hurt Honcho cannot occur here by construction.
3. **Per-agent themes by default, cross-theme recall on explicit demand.** Every row carries `agent_identity` (resolved from `X-Hermes-Session-Key` header, profile name, workspace, or `'default'`). Recall is scoped to the current theme unless the agent asks for `scope='all'`.
4. **Fail-soft everywhere.** Embed endpoint down → degrade to text-only writes. Async writer queue full → drop with a one-time warning. DB down → log + skip. No exception escapes into the agent loop.
5. **Admin/runtime separation.** DDL (`CREATE EXTENSION vector`, `CREATE TABLE`, `CREATE INDEX`) runs once with superuser. The runtime user has DML only on the migrated schema. `ensure_schema()` at runtime is verify-only with a clear `SchemaNotApplied` error if the operator forgot the migration.

## Features (v0.4.0)

Two integration surfaces, one shared store. The fork's v0.4.0 also replaced the upstream 768-dim HTTP embedder with a **local 384-dim sentence-transformers MiniLM-L6-v2** model (88MB on disk, ~500MB RAM on CPU, no GPU required).

### Hermes plugin surface

| Hook / surface | Behavior |
|---|---|
| `initialize()` | Verifies schema, opens `psycopg_pool.ConnectionPool`, bulk-imports existing `MEMORY.md` + `USER.md` content. |
| `on_memory_write(action, target, content, meta)` | Mirrors built-in `memory` writes into `memory_entries` (add / replace / remove). |
| `sync_turn(user, assistant, session_id)` | Captures every substantive (`>= 40` chars + not boilerplate) chat turn into `conversations`. |
| `prefetch(query)` | Top-K semantically similar `memory_entries` in current theme, injected ambient. |
| `recall_memory(query, scope, target, limit)` tool | Explicit cross-theme search of durable memory entries. |
| `recall_conversation(query, scope, limit)` tool | Explicit search over past chat turns. `scope ∈ {current, session, all, <theme>}`. |

### MCP server surface (NEW in v0.4.0)

Eight tools exposed to any MCP client (Claude Desktop, Cursor, custom agents). All take an optional `agent_identity` argument so each connected client is isolated by default.

| Tool | Purpose |
|---|---|
| `memory_health` | Liveness + capability check (DB status, embedder model/dim, row counts). |
| `memory_retain` | Add one or many memory entries. `target='memory'|'user'`, optional metadata, `doc_type`, `source_url`. |
| `memory_recall` | Semantic search over `memory_entries`. `top_k` (cap 100), `min_similarity` (0..1). |
| `memory_search` | Browse entries (no embedding) — pagination, scoping. |
| `memory_forget` | Delete by id. **Dry-run by default**; pass `confirm=true` to actually delete. |
| `memory_recall_turns` | Semantic search over past chat turns. |
| `memory_append_turn` | Append one chat turn. Mirrors the plugin's `sync_turn`. |
| `memory_count` | Row counts for entries + turns, scoped. |

### Internals (shared by both surfaces)

- **`psycopg_pool.ConnectionPool`** (min=0, max=4, lazy + thread-safe, `max_idle=30s` / `max_lifetime=300s`) shared across the agent thread and the async-writer drain thread. `min_size=0` keeps an idle — or abandoned — pool at **zero** open connections, so a session the gateway never explicitly shuts down cannot strand a Postgres backend (see *Fixed in v0.3.1* below).
- **`AsyncWriter`** — bounded queue + daemon drain thread. Memory write hooks return in microseconds. Worker embeds + writes in the background. Crash-resilient (auto-restart on next enqueue).
- **Single migration** (`hexus/migrations/001_schema.sql`) — `memory_entries` + `conversations` + HNSW indexes. Same tuning operators typically use elsewhere.
- **Boilerplate filter** for turn capture — length floor + acknowledgement regex (`"ok"`, `"thanks"`, `"continue"`, …) so the recall table stays high-signal.

### Fixed in v0.3.1 — connection-leak hotfix

A single registered provider has `initialize()` called again for each new session. It previously
reassigned `self._store` / `self._writer` without closing the prior ones, **abandoning a
`ConnectionPool`** whose warm (`min_size=1`) connection lingered in Postgres — committed-but-idle —
until the server's `idle_session_timeout`. Under a burst of concurrent sessions (e.g. a swarm of
systemd-run minions firing on the same minute) these orphaned backends saturated the database's
connection slots. Fixed by:

1. **`initialize()` teardown** — drain the prior `AsyncWriter` + close the prior pool before
   re-initializing (the call is idempotent and skipped on first init).
2. **Self-draining pool** — `min_size=0` (an idle or abandoned pool holds *zero* connections) plus
   `max_idle=30s` / `max_lifetime=300s`, so connections are short-lived when idle and pooled only
   under active load.

## Multi-agent / per-minion themes

Each systemd-run minion sets one header on its OpenAI client; everything else flows automatically:

```python
client = AsyncOpenAI(
    base_url="http://127.0.0.1:8642/v1",
    api_key=API_KEY,
    default_headers={"X-Hermes-Session-Key": "marketing"},   # ← theme
)
```

The gateway plumbs `X-Hermes-Session-Key` through as `gateway_session_key=…` in `MemoryProvider.initialize` kwargs. The plugin reads it with **priority over the profile default**, so `agent_identity='default'` from unprofiled API traffic does not collapse every minion into one shared scope.

Convention: lowercase, dash-separated, stable. Examples that work well:

- `marketing`, `sales`, `morning-report`, `incident`
- `intraday-<agent_name>` for fan-out workers (e.g. `intraday-trading`, `intraday-sre`, `intraday-marketing`)

## MCP server

The same `MemoryStore` is exposed to non-hermes agents as a standard MCP server. One process, one model load (the local BERT is shared), one shared knowledge base for every connected client.

### Quickstart (docker)

```bash
# Build the image (downloads MiniLM-L6-v2 into /home/hermes/.cache/huggingface/
# during build; offline at runtime thanks to HF_HUB_OFFLINE=1).
docker compose -f docker/compose.yml --profile mcp up -d --build

# Tail the server's log to confirm it started
docker compose -f docker/compose.yml --profile mcp logs -f mcp
# [entrypoint] starting MCP server (transport=http)

# Health check (one-shot, JSON, exit 0 if healthy)
docker compose -f docker/compose.yml --profile mcp exec mcp \
    hexus-mcp doctor --dsn "dbname=hermes_test user=postgres password=postgres host=pg"
```

The MCP server listens on `0.0.0.0:8000` inside the compose network (streamable-http transport). To expose it to the host, add a `ports: ["8000:8000"]` mapping to the `mcp` service in your compose override.

### Claude Desktop / Cursor (stdio)

Claude Desktop and Cursor speak MCP over stdio — one MCP server per client process. The cleanest bridge is to run `hexus-mcp serve --transport stdio` on demand from the editor's MCP config:

```jsonc
// ~/.config/claude_desktop_config.json (Claude Desktop)
// or ~/.cursor/mcp.json (Cursor)
{
  "mcpServers": {
    "hexus": {
      "command": "hexus-mcp",
      "args": ["serve", "--transport", "stdio"],
      "env": {
        "HEXUS_DSN": "dbname=hermes_memory user=hermes host=/var/run/postgresql",
        "HEXUS_AGENT_IDENTITY": "claude-desktop"
      }
    }
  }
}
```

The `HEXUS_AGENT_IDENTITY` env var is the default `agent_identity` for tool calls that don't supply one — pick a stable name per editor so a Claude Desktop session can read/write its own memory without colliding with Cursor's. To share memory between two editors, set both to the same identity. To query across all agents, pass `agent_identity=""` on `memory_recall` / `memory_search`.

### Multi-agent fleet (streamable-http)

One long-lived server, many connected clients (custom agents, web clients, CI bots). Each picks its own `agent_identity` per call (or via the `HEXUS_AGENT_IDENTITY` env var on the client process). The server process is shared, so a single BERT model (~500MB resident) serves every client.

```bash
# Start the server (docker compose --profile mcp, or directly)
hexus-mcp serve --transport http --host 0.0.0.0 --port 8000 \
    --dsn "dbname=hermes_memory user=hermes host=/var/run/postgresql"

# An agent talking to it — point your MCP client at http://localhost:8000/mcp
# (the /mcp path is the streamable-http endpoint set by FastMCP).
```

### Example tool calls

```python
# Python: connect via the official `mcp` client SDK
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(
        command="hexus-mcp",
        args=["serve", "--transport", "stdio"],
        env={"HEXUS_DSN": "dbname=hermes_test user=postgres host=pg",
             "HEXUS_AGENT_IDENTITY": "demo-agent"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 1. Health
            h = await session.call_tool("memory_health", {})
            # 2. Add some memory
            await session.call_tool("memory_retain", {
                "contents": ["Postgres + hexus is a great choice for semantic memory."],
                "target": "memory",
            })
            # 3. Recall
            r = await session.call_tool("memory_recall", {
                "query": "what should I use for semantic search",
                "top_k": 5,
            })
            print(r.content[0].text)

asyncio.run(main())
```

### `doctor` for ops + CI

```bash
# One-shot health check; prints JSON, exits 0 if healthy
hexus-mcp doctor --dsn "dbname=hermes_memory user=hermes host=/var/run/postgresql"
# {
#   "status": "ok",
#   "schema_ok": true,
#   "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "eager_loaded": false},
#   "row_counts": {"memory_entries": 1234}
# }
```

The same command is the docker `healthcheck` for the `mcp` service, so a misbehaving server gets marked unhealthy automatically and your orchestrator can restart it.

## Install

### Option 0: docker (recommended for v0.4.0+)

The whole stack — Postgres + hexus + the MCP server, or just the test runner — runs through docker compose. This is the only path the fork actively tests.

```bash
git clone https://github.com/codenamekt/hexus.git
cd hexus

# Run the test suite (15 + 19 + 10 + 8 = ~50 tests, all green in 6-7s)
docker compose -f docker/compose.yml --profile test up --abort-on-container-exit --exit-code-from test

# Or start the MCP server + Postgres (long-lived)
docker compose -f docker/compose.yml --profile mcp up -d --build

# Or just the DB for host-side development
docker compose -f docker/compose.yml --profile dev up -d pg
PG_TEST_DSN="dbname=hermes_test user=postgres host=localhost" pytest tests/
```

The image is multi-stage: deps pre-downloads MiniLM-L6-v2 (88MB) into the runtime stage's hermes user cache. Runtime runs offline (`HF_HUB_OFFLINE=1`) and as non-root.

### Option 1: clone + run the installer script (legacy)

```bash
git clone https://github.com/andreab67/hermes-hexus.git
cd hermes-hexus
./scripts/install.sh
```

That:

1. `pip install`s `psycopg[binary]`, `psycopg-pool`, `PyYAML` (with the upper-bound pins).
2. Copies `hexus/` into `$HERMES_HOME/plugins/hexus/` (defaults to `~/.hermes/plugins/hexus/`).
3. Prints the admin migration + activation commands you run next.

For the MCP server, additionally:

```bash
pip install "hexus[mcp]"   # adds mcp[cli] + uvicorn + starlette
```

### Option 2: manual

```bash
# Python deps
pip install 'psycopg[binary]>=3.3.4,<4' 'psycopg-pool>=3.3.1,<4' 'PyYAML>=6.0,<7'

# Plugin module
mkdir -p ~/.hermes/plugins
cp -r hexus ~/.hermes/plugins/hexus
```

### Then (admin once)

```bash
# Apply the schema migration (CREATE EXTENSION needs superuser)
sudo -u postgres psql -d <your-memory-db> \
     -f ~/.hermes/plugins/hexus/migrations/001_schema.sql

# Hand ownership of the new tables to the hermes runtime role
sudo -u postgres psql -d <your-memory-db> -c "
ALTER TABLE memory_entries OWNER TO hermes;
ALTER SEQUENCE memory_entries_id_seq OWNER TO hermes;
ALTER TABLE conversations OWNER TO hermes;
ALTER SEQUENCE conversations_id_seq OWNER TO hermes;
"

# Activate
hermes config set memory.provider hexus
sudo systemctl restart hermes.service     # or however you run hermes
hermes memory status                       # expect: Provider: hexus; Status: available
```

## Configuration

Lives in `$HERMES_HOME/config.yaml` under `plugins.hexus` — every value optional, sensible defaults shown:

```yaml
plugins:
  hexus:
    dsn: "dbname=hermes_memory user=hermes host=/var/run/postgresql"
    # No embed_url → use the local sentence-transformers MiniLM-L6-v2
    # model (default, recommended). Set embed_url to override with an
    # OpenAI-compatible or Ollama-native endpoint that returns 384-dim
    # vectors (the schema is hard-coded to vector(384)).
    embed_url: null
    embed_model: "sentence-transformers/all-MiniLM-L6-v2"
    prefetch_limit: 5
    min_similarity: 0.30
    embed_on_write: true
    scope_default: "current"
    write_queue_maxsize: 256
    bulk_sync_on_init: true
    sync_turns: true
    turn_min_chars: 40
    # Set true to pre-load the BERT model at plugin init (saves ~1-2s
    # on the first embed call). Default false = lazy on first use.
    embed_eager_load: false
```

The embed endpoint can be any OpenAI-compatible `/v1/embeddings` or Ollama-native `/api/embed` URL that returns **384-dim vectors** (the schema is hard-coded to `vector(384)` to match the local MiniLM-L6-v2 model). Use a different model only if it produces 384-dim output, or edit the migration before applying it.

## Schema

```sql
CREATE TABLE memory_entries (
  id              BIGSERIAL PRIMARY KEY,
  agent_identity  TEXT NOT NULL DEFAULT 'default',
  target          TEXT NOT NULL CHECK (target IN ('memory', 'user')),
  content         TEXT NOT NULL,
  embedding       vector(384),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (agent_identity, target, content)
);

CREATE TABLE conversations (
  id              BIGSERIAL PRIMARY KEY,
  session_id      TEXT NOT NULL,
  agent_identity  TEXT NOT NULL DEFAULT 'default',
  role            TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
  content         TEXT NOT NULL,
  ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
  embedding       vector(384),
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);
```

Indexes: HNSW on each `embedding` column (m=16, ef_construction=64) plus per-agent + per-session btree timelines. Full DDL in [`hexus/migrations/001_schema.sql`](hexus/migrations/001_schema.sql).

## Tests

The full test suite runs in docker — see the [docker quickstart](#option-0-docker-recommended-for-v040) above. Three test files, ~50 tests total, all green in ~7s:

- `tests/test_smoke.py` — MemoryStore + AsyncWriter + embed dispatch (16 tests)
- `tests/test_embedder.py` — LocalBertEmbedder class, lazy load, batch, dispatch (19 tests)
- `tests/test_migration.py` — schema contract, idempotency, dim enforcement (10 tests)
- `tests/test_mcp_server.py` — every MCP tool, multi-agent isolation, FastMCP wiring (8+ tests)

```bash
# All in one shot
docker compose -f docker/compose.yml --profile test up --abort-on-container-exit --exit-code-from test

# Or, against a host-side Postgres+hexus (the test container doesn't pull the deps):
PG_TEST_DSN="dbname=hermes_test user=postgres host=localhost" \
  pytest tests/ -v
```

The host-side mode requires `pip install -e ".[test,mcp]"` once. DB tests skip when `PG_TEST_DSN` is unset; the HTTP-embed test skips when `PG_TEST_EMBED_URL` is unset (the local BERT path runs unconditionally — it uses the preloaded model in the docker image).

## Benchmarks

See [`BENCHMARK.md`](BENCHMARK.md) for detailed latency, throughput, and scaling numbers covering:

- LocalBertEmbedder (MiniLM-L6-v2) single/batch embed latency
- MemoryStore insert + recall at various corpus sizes
- Multi-agent isolation verification
- Resource usage (RAM, disk) and scaling projections
- Instructions for running the benchmark yourself

```bash
# Quick run (requires docker image + running pg)
cat benchmarks/bench.py | docker run --rm --entrypoint python \
  --network hexus_default \
  -e PG_TEST_DSN="dbname=hermes_test user=postgres password=postgres host=hexus-pg" \
  hexus:test
```

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the full milestone table. Highlights:

- **M1 (v0.1, v0.1.1)** ✅ Shared storage with per-agent themes, async writer, connection pool, bulk import from `MEMORY.md`/`USER.md`
- **M2 (v0.2)** ✅ Conversation transcript table with `sync_turn` capture + `recall_conversation` tool
- **M3 (v0.3)** ✅ Identity propagation for stateless API minions via `X-Hermes-Session-Key`
- **M3.5 (v0.4 fork)** ✅ Local BERT swap (MiniLM-L6-v2, 384-dim) + MCP server (`hexus-mcp serve`) for non-hermes clients
- **M4 (v0.5 fork)** ⏳ Hybrid search, temporal decay, TTL/memory decay tool — [see upcoming features](UPCOMING.md)
- **M5 (v0.6 fork)** ⏳ Entity tagging, co-occurrence graph, confidence scoring, session summaries
- **M6 (v0.7 fork)** ⏳ Cross-encoder reranker, event webhooks, PyPI/GHCR publish
- **M7 (v1.0)** ⏳ Stable config schema, full docs, CI coverage

The roadmap exists so the multi-agent positioning isn't a one-off claim — each milestone has to pass the test *"does this make N cooperating agents more capable?"* before it lands. The `What's not on the roadmap` section in `ROADMAP.md` lists what was deliberately rejected (LLM-mediated dialectic, fact-store ontologies, background derivers, in-plugin RBAC) so the boundaries are explicit.

## Upcoming Features

See [`UPCOMING.md`](UPCOMING.md) for a detailed breakdown of 9 planned features across 3 phases — from hybrid search (BM25 + vector) to entity co-occurrence graphs and conversation summaries. Every feature is designed for **dual-surface availability**: each ships as both an MCP tool and a Hermes memory plugin tool, backed by a single shared implementation in `MemoryStore`.

## Rollback

```bash
hermes config set memory.provider none
sudo systemctl restart hermes.service

# Optional — drop the tables (data loss, irreversible)
sudo -u postgres psql -d <your-memory-db> -c "
DROP TABLE IF EXISTS conversations;
DROP TABLE IF EXISTS memory_entries;
"

# Optional — remove the plugin files
rm -rf ~/.hermes/plugins/hexus
```

## Why a standalone plugin (not an upstream PR)?

Per the hermes-agent [`CONTRIBUTING.md`](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md):

> We are no longer accepting new memory providers into this repo. The set of built-in providers under `plugins/memory/` is closed. If you want to add a new memory backend, publish it as a standalone plugin repo that users install into `~/.hermes/plugins/` (or via a pip entry point).

The discovery system (`plugins/memory/__init__.py` in hermes-agent) scans `$HERMES_HOME/plugins/<name>/` for any directory whose `__init__.py` calls `register_memory_provider`. This plugin's `hexus/__init__.py` does exactly that — no upstream change required.

## Contributing

Bug reports + PRs welcome. Open an issue describing the failure mode + your environment (hermes-agent version, Postgres version, embed endpoint), or a PR with a focused change + test.

## License

[BSD 3-Clause](LICENSE) © 2026 Andrea Borghi.

# hexus — Roadmap

A versioned plan for evolving the plugin from "Postgres storage for a single agent" to "shared memory substrate for a fleet of cooperating hermes-agent minions AND any MCP client." The driving constraint throughout is **deliver multi-agent memory on the resources you already have** — your existing Postgres, a single in-process embedder, no LLM costs in the memory hot path, no third-party services.

The plugin is built around a clear separation of concerns:

- The **agent's mental model** stays unchanged — keep using the built-in `memory(action='add', target='memory'|'user', …)` tool.
- The **storage backbone** moves from per-host markdown files to a centralized Postgres table that every minion can read from and write to.
- The **scoping mechanism** (`agent_identity`) keeps each minion's working memory clean while still allowing explicit cross-theme recall when an agent needs the bigger picture.
- The **transport surface** is two-headed: hermes-agent hooks (for the gateway path) AND a standard MCP server (for Claude Desktop, Cursor, fleet agents). Both talk to the same `MemoryStore` and share the same `LocalBertEmbedder` instance per process.

---

## Milestones

### M1 — Shared storage with per-agent themes (v0.1 → v0.1.1) ✅ DONE

**Goal:** every minion's `memory(action='add', …)` write lands in a single Postgres table with semantic search on top, scoped so marketing's notes don't pollute trading's recall.

| Capability | Version |
|---|---|
| Mirror built-in `memory.add` / `replace` / `remove` via `on_memory_write` hook | v0.1.0 |
| Per-`agent_identity` row scoping (marketing / sales / trading / incident / `default`) | v0.1.0 |
| 768-dim embeddings via OpenAI-compatible or Ollama-native endpoint | v0.1.0 |
| HNSW index for sub-millisecond recall up to ~1M rows | v0.1.0 |
| Async writer (bounded queue + daemon drain) — agent loop never blocks on slow embeds | v0.1.0 |
| `psycopg_pool.ConnectionPool` — small pool shared across agent + writer threads | v0.1.0 |
| Admin/runtime DDL split (`apply_migration_as_admin()` + verify-only `ensure_schema()`) | v0.1.0 |
| `recall_memory(query, scope, target, limit)` cross-theme search tool | v0.1.0 |
| Bulk-import existing `MEMORY.md` + `USER.md` from disk on init (idempotent + cheap on re-init) | v0.1.1 |

**Why this matters for multi-agent deployments:** before this milestone, every hermes-agent process kept its own markdown files. Two agents on the same host stomped on each other; agents on different hosts had no shared substrate at all. M1 gives every minion a single source of truth, scoped per-theme so isolation is the default and cross-theme recall is an explicit opt-in.

---

### M2 — Conversation-history substrate (v0.2) ✅ DONE

**Goal:** every substantive chat turn across every minion becomes semantically searchable. Filter the boilerplate (`"ok"`, `"thanks"`, sub-40-char turns) so recall stays high-signal.

| Capability | Version |
|---|---|
| `conversations` table: id, session_id, agent_identity, role, content, ts, embedding, metadata | v0.2.0 |
| `sync_turn(user, assistant, session_id)` hook captures every turn pair | v0.2.0 |
| Boilerplate filter (length floor + acknowledgement regex) | v0.2.0 |
| Per-session + per-agent timeline indexes | v0.2.0 |
| HNSW index on conversation embeddings — same tuning as `memory_entries` | v0.2.0 |
| `recall_conversation(query, scope, limit)` tool with `scope ∈ {current, session, all, <theme>}` | v0.2.0 |

**Why this matters for multi-agent deployments:** in a fleet, *the chat is the memory*. A finance agent can pull "what did marketing say about last quarter's CAC" without anybody having to copy-paste between systems. The agent fetches it as a tool call, exactly when it needs it.

---

### M3 — Identity propagation for stateless API minions (v0.3) ✅ DONE

**Problem solved:** before v0.3, every systemd-run minion (`marketing-daily.py`, `sales-daily.py`, intraday workers, morning-report enrich) hit the gateway via `POST /v1/chat/completions` and the gateway forwarded a `gateway_session_key` kwarg (from the built-in `X-Hermes-Session-Key` header), but the hexus plugin didn't consume it — every API-routed write collapsed to `agent_identity='default'`. The per-theme isolation promised in M1 was theoretical for fleet-style use.

| Capability | Version |
|---|---|
| Plugin reads `kwargs.get('gateway_session_key')` in the `agent_identity` fallback chain | v0.3.0 ✅ |
| Each minion sets `default_headers={'X-Hermes-Session-Key': '<theme>'}` on its OpenAI client | v0.3.0 ✅ |
| `intraday_loop._get_client_for(agent_name)` per-agent client cache so `intraday-trading`, `intraday-sre`, `intraday-marketing`, etc. are each scoped | v0.3.0 ✅ |
| Optional: per-theme allow-list in plugin config so a typo'd header can't silently create a new theme | v0.3.1 ⏳ |

**Theme naming convention:** lowercase, dash-separated, stable. Established themes:

- `marketing` — `scripts/marketing-daily.py`, `scripts/marketing/content-factory.py`
- `sales` — `scripts/sales-daily.py`
- `morning-report` — `scripts/morning-report/enrich.py`
- `intraday-{agent_name}` — every `scripts/agents/agent-*-worker.py` via `intraday_loop.run(agent_name=…)`. Concretely: `intraday-trading`, `intraday-sre`, `intraday-marketing`, `intraday-gitlab`, `intraday-cloud`, `intraday-hermes`.
- `incident` — reserved for incident-responder (Phase 1 observe-only today, will activate when its Phase 2 ships LLM hypothesis-forming)
- `default` — last-resort bucket for interactive `hermes chat` without `--profile` or any caller that doesn't set the header
- `claude-desktop`, `cursor`, etc. — when a non-hermes MCP client is connected via the MCP server (M3.5), this is how it scopes itself

**Why this matters for multi-agent deployments:** v0.3 is the smallest change that makes the M1 design true in practice. Without it, a marketing-daily run could surface a trading-agent's notes in its recall, defeating the isolation. With it, each minion has its own memory pool by default while still being able to ask cross-theme questions through `recall_memory(scope='all')`.

---

### M3.5 — Local BERT + MCP server (v0.4 fork) ✅ DONE

**Goal:** make the memory substrate a real fleet product. Two changes shipped together because they're complementary:

1. **Local BERT swap** — replace the upstream 768-dim HTTP embedder (Ollama / OpenAI-compatible) with a local sentence-transformers MiniLM-L6-v2 (384-dim, ~88MB on disk, <500MB RAM on CPU, no GPU required). The HTTP path is preserved as a fallback for operators with an existing embed service. Why: the CPU model runs in well under 500MB, the test loop doesn't depend on an external service being reachable, and the v0.4.0 docker image is fully air-gapped at runtime (`HF_HUB_OFFLINE=1`).
2. **MCP server** — expose the same `MemoryStore` to any MCP client (Claude Desktop, Cursor, custom agents) as eight tools (`memory_retain`, `memory_recall`, `memory_search`, `memory_forget`, `memory_recall_turns`, `memory_append_turn`, `memory_count`, `memory_health`). Multi-agent: each connected client picks its own `agent_identity` per call (or via the `HEXUS_AGENT_IDENTITY` env var on the client process). The server is a single process — one model load, one shared knowledge base — and supports both `stdio` (Claude Desktop / Cursor) and `streamable-http` (fleet use) transports.

| Capability | Version |
|---|---|
| `LocalBertEmbedder` (sentence-transformers MiniLM-L6-v2, 384-dim) with lazy load + thread-safe singleton | v0.4.0 ✅ |
| `embed.py` dual-path dispatch: `base_url=None` → local BERT, else HTTP fallback | v0.4.0 ✅ |
| Schema migrates `vector(768)` → `vector(384)` in the same idempotent migration | v0.4.0 ✅ |
| Multi-stage Dockerfile: deps pre-downloads MiniLM, runtime is hermes user + HF_HUB_OFFLINE=1 | v0.4.0 ✅ |
| `mcp_server/` package: pure tool functions in `tools.py`, FastMCP wiring in `server.py`, CLI in `cli.py` | v0.4.0 ✅ |
| `hexus-mcp serve` (stdio / streamable-http) + `hexus-mcp doctor` (one-shot health) | v0.4.0 ✅ |
| Eight MCP tools covering retain / recall / search / forget / turn-append / turn-recall / count / health | v0.4.0 ✅ |
| Per-tool `agent_identity` arg + `HEXUS_AGENT_IDENTITY` env var — multi-agent out of the box | v0.4.0 ✅ |
| Docker `mcp` profile: pg + long-lived server, with `doctor` as the healthcheck | v0.4.0 ✅ |
| 50+ tests (smoke + embedder + migration + mcp server + FastMCP wiring) all green in docker in ~7s | v0.4.0 ✅ |

**Why this matters for multi-agent deployments:** before M3.5, only hermes-agent minions could write to the store. After M3.5, any MCP client can — Claude Desktop can read and write shared memory across a fleet, CI bots can `memory_retain` test fixtures, web clients can `memory_recall` from a marketing session. The hermes plugin still works exactly as before (drop-in replacement), but it's no longer the only on-ramp. Local BERT also means the whole stack runs offline on a NUC with no cloud dependencies.

---

### M4 — Agent-of-agents observability (v0.5 fork) ⏳ PROPOSED

**Goal:** when one minion delegates to another (subagent pattern), capture the task/result pair so the parent can recall "what did I ask my research subagent last week and what did it find." Extend this to the MCP server so a Claude Desktop session can query the same delegation history.

| Capability | Version |
|---|---|
| `on_delegation(task, result, child_session_id)` hook → row in `conversations` (or a separate `delegations` table) | v0.5.0 |
| `on_session_end(messages)` hook → optional session summary row, agent-decided | v0.5.0 |
| `recall_delegation(query, scope, limit)` tool — surfaces past delegation transcripts | v0.5.0 |
| Parent ↔ child session linkage in metadata for traceback | v0.5.0 |
| New MCP tool `memory_recall_delegations` mirroring the plugin tool | v0.5.0 |

**Why this matters for multi-agent deployments:** orchestrator patterns (one supervisor minion fanning out to N specialists) become much more reviewable when delegations are first-class durable records. Without this, the only place the supervisor remembers a delegation is the immediate conversation context, which compresses away. With this, a Claude Desktop session in the MCP client can `memory_recall_delegations` to see "what did marketing ask research last week" — cross-tooling observability that v0.4 already enables the data layer for.

---

### M5 — Production hardening at scale (v0.6 → v0.7 fork) ⏳ PROPOSED

**Goal:** survive a fleet of dozens of minions, hundreds of writes per minute, multi-million-row tables. Plus publish artifacts so the plugin is a one-liner install.

| Capability | Version |
|---|---|
| TTL / decay policy on `memory_entries.updated_at` and `conversations.ts` so stale entries surface less | v0.6.0 |
| Optional partial HNSW indexes per high-volume `agent_identity` when cross-theme search becomes the slow query | v0.6.0 |
| Periodic re-sync of `MEMORY.md` / `USER.md` (not just on init) for callers that edit the markdown directly | v0.6.0 |
| Bulk-import CLI for migrating from Holographic / Honcho / Mem0 / Hindsight installations | v0.6.0 |
| Metrics: queue depth, drop count, embed latency p50/p95, recall hit rate (Prometheus-friendly) | v0.7.0 |
| Per-platform metadata facets (CLI vs cron vs telegram vs API) for richer recall filtering | v0.7.0 |
| PyPI publish (`pip install hexus` and `pip install hexus[mcp]`) | v0.7.0 |
| GitHub Actions CI: build the image, run the test suite, publish the image to GHCR on tag | v0.7.0 |

**Why this matters for multi-agent deployments:** a memory store that's fast for one user often falls over under fleet load. M5 is the slow + boring work that turns "works on my hermes" into "works for ten agents writing concurrently — and for ten non-hermes MCP clients reading concurrently."

---

### M6 — Public release (v1.0) ⏳ PROPOSED

**Goal:** documentation, contract guarantees, and a CHANGELOG good enough that someone landing on this plugin from the hermes-agent docs (or from the MCP server directory) can deploy it cleanly without reading the source.

| Capability | Version |
|---|---|
| Stable config schema (semver guarantees on `plugins.hexus.*` keys + MCP tool input schemas) | v1.0.0 |
| Full hermes-agent docs page with config reference, scaling guide, troubleshooting | v1.0.0 |
| Coverage: store + mcp tests in upstream CI against a Postgres service container | v1.0.0 |
| Conformance test: validate `MemoryProvider` ABC contract against current upstream | v1.0.0 |
| MCP server conformance: validate the FastMCP wiring against the official MCP spec test suite | v1.0.0 |

---

## What's *not* on the roadmap (and why)

These were considered and rejected. Keeping the list visible so it's clear the omissions are deliberate, not gaps.

- **A `fact_store` ontology with trust scoring, entity resolution, and HRR algebra.** Holographic does that well. Layering it in hexus would duplicate Holographic's surface and force agents to learn a second memory model when the built-in `memory` tool already serves the same need.
- **LLM-mediated dialectic recall** (à la Honcho). The synchronous LLM call in the memory hot path is exactly the failure mode that motivated this plugin. We embed text; we don't reason about it. The agent reasons.
- **Background deriver / fact-extraction pipelines.** Same reason: any background LLM loop becomes a retry-storm liability. Fact extraction stays explicit (the agent decides to call `memory.add`) instead of implicit (a daemon thread scrapes turns).
- **Multi-tenant authentication / RBAC at the plugin layer.** Postgres roles + `agent_identity` scoping are sufficient. Anything fancier belongs in a separate access layer, not in a memory provider.
- **Re-implementing the hermes-agent `memory` tool inside MCP.** The MCP server exposes retain/recall/search/forget — primitive operations the agent (hermes or otherwise) composes into whatever mental model it wants. We don't ship an opinionated "this is how an agent should organize its memory" — that's the agent's call, not ours.

## Operating principle

Every milestone has to answer: **does this make N cooperating agents more capable, or does it just add features?** If it doesn't pass that test, it goes in the "Not on the roadmap" pile.

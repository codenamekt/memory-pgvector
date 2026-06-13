# Upcoming Features

Features planned for the next phases of hexus, ordered by impact-to-effort ratio. Every feature is designed for **dual-surface availability** — it is exposed as both an MCP tool (for Claude Desktop, Cursor, fleet agents) and as a Hermes memory plugin tool or hook (for `hermes-agent` minions via the gateway). Both surfaces call the same underlying `MemoryStore` methods; no feature is exclusive to one transport.

## Architecture Principle: Dual-Surface by Default

```
operator configures feature once (pyproject.toml or config.yaml)
          │
          ▼
    MemoryStore / shared module (single implementation)
         ╱           ╲
   MCP tool       Hermes plugin hook / tool
   (mcp_server/)  (hexus/__init__.py)
```

Every feature below lives in the shared core. The MCP tool and the Hermes hook are thin wrappers — typically 5-10 lines each — that marshal arguments into the same method call.

---

## Phase 5: High Impact, Low Effort (Next Week)

### 1. Hybrid Search: Semantic + Full-Text (configurable weights)

**What it does:** Every query runs both HNSW vector search AND Postgres full-text search (`tsvector` / `tsquery`), then merges results with a configurable weight. This gives operators the retrieval precision of RetainDB's BM25 + cross-encoder pipeline without leaving Postgres or adding a new dependency.

**Dual-surface exposure:**

| Surface | Tool / Hook | Args |
|---------|------------|------|
| MCP | `memory_hybrid_search` | `query`, `top_k`, `vector_weight` (default 0.7), `text_weight` (default 0.3), `agent_identity`, `target`, `min_similarity`, `filters` |
| Hermes plugin | `hybrid_search` tool | Same signature, registered via `recall_memory` pattern |

**Implementation sketch:**

```sql
SELECT id, content, metadata,
       ($vector_weight * (1 - (embedding <=> $1::vector)))
       + ($text_weight * ts_rank(to_tsvector('english', content), plainto_tsquery('english', $2)))
       AS hybrid_score
FROM memory_entries
WHERE agent_identity = $3
  AND ($2::text = '' OR to_tsvector('english', content) @@ plainto_tsquery('english', $2))
ORDER BY hybrid_score DESC
LIMIT $4
```

**Config (both surfaces):**
```yaml
# hermes config.yaml
plugins:
  hexus:
    hybrid_search:
      enabled: true
      vector_weight: 0.7
      text_weight: 0.3
      language: "english"        # Postgres text search config

# MCP server env
HEXUS_HYBRID_VECTOR_WEIGHT=0.7
HEXUS_HYBRID_TEXT_WEIGHT=0.3
```

**Effort:** ~80 lines (new `hybrid_search()` method on `MemoryStore`, one MCP tool, one Hermes tool schema)

---

### 2. Temporal Decay Scoring

**What it does:** Older memory entries surface lower in search results. An exponential decay curve (`score × 2^(-age/half_life)`) is applied to every search result after the DB query. Disabled by default (`half_life_days: 0`); operators set a value (e.g., 30 days) to enable it.

**Dual-surface exposure:**

| Surface | Mechanism |
|---------|-----------|
| MCP | Applied implicitly to all `memory_recall` / `memory_hybrid_search` results. Configurable via `HEXUS_DECAY_HALF_LIFE_DAYS` env var. |
| Hermes plugin | Applied via `decay_half_life_days` config key in `plugins.hexus`. Same decay function. |

**Implementation:** Pure post-processing on `MemoryStore.search()` and `MemoryStore.search_turns()` results. No DB changes needed.

```python
def _apply_decay(rows: list[dict], half_life_days: int) -> list[dict]:
    if half_life_days <= 0:
        return rows
    now = datetime.now(tz=timezone.utc)
    for row in rows:
        updated = row.get("updated_at") or row.get("ts")
        if updated is None:
            continue
        age_days = (now - updated).days
        row["raw_score"] = row.get("score", 0)
        row["score"] = row.get("score", 0) * (2 ** (-age_days / half_life_days))
    return sorted(rows, key=lambda r: r.get("score", 0), reverse=True)
```

**Config:**
```yaml
plugins:
  hexus:
    decay_half_life_days: 0    # 0 = disabled; 30 = entries half as relevant after 30 days
```

**Effort:** ~40 lines (decay function + config key + apply in `search()` / `search_turns()`)

---

### 3. TTL / Memory Decay Tool

**What it does:** Give every memory entry an optional TTL (time-to-live) after which it is auto-deleted or its score is permanently reduced. The agent can set TTL at write time or adjust it later. This is Supermemory's "forgetfulness" without the knowledge graph dependency.

**Dual-surface exposure:**

| Surface | Tool | Behavior |
|---------|------|----------|
| MCP | `memory_decay` | `id`, `reduce_by` (0-1), `ttl_days`, `confirm` |
| Hermes plugin | `decay_memory` | Same signature, registered as a memory plugin tool |

**Implementation:** Stores TTL deadline in `metadata.ttl_deadline` (ISO timestamp). A lightweight cleanup query runs on `MemoryStore` init (or via a periodic background task):

```sql
DELETE FROM memory_entries
WHERE (metadata->>'ttl_deadline')::timestamptz < now();
```

**Config:**
```yaml
plugins:
  hexus:
    ttl_cleanup_interval_hours: 24   # how often to run the cleanup query (0 = on init only)
```

**Effort:** ~60 lines (tool handler + cleanup query + config)

---

## Phase 6: Medium Impact, Medium Effort (Next 2 Weeks)

### 4. Entity Tagging (regex-based, no LLM)

**What it does:** Extract named entities (URLs, domains, email addresses, IPs, file paths, version numbers, Docker images, hostnames) from every memory entry at write time using configurable regex patterns. Store entities in `metadata.entities` as a JSONB array. Expose entity-based filters on all search tools.

**Dual-surface exposure:**

| Surface | Mechanism |
|---------|-----------|
| MCP | `memory_retain` auto-extracts. `memory_search` / `memory_hybrid_search` accept `filters=[{entity_type: "url", entity_value: "github.com/..."}]`. New tool: `memory_list_entities(agent_identity, entity_type)` for browsing. |
| Hermes plugin | Same extraction in `on_memory_write`. Same filter support in `recall_memory`. New tool: `list_entities`. |

**Config (both surfaces):**
```yaml
plugins:
  hexus:
    entity_extraction:
      enabled: true
      patterns:
        url: 'https?://[^\s<>"]+'
        domain: '\b([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        email: '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        file_path: '(?:/[\w.-]+)+\.\w+'
        version: '\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b'
        ip_address: '\b(?:\d{1,3}\.){3}\d{1,3}\b'
        docker_image: '\b[a-zA-Z0-9][a-zA-Z0-9_.-]*(?::[a-zA-Z0-9_.-]+)?\b'
        hostname: '\b[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\b'
      # Operators can add custom patterns:
      # custom_project_prefix: 'PROJ-\d{4,6}'
```

**Effort:** ~150 lines (entity extractor module + integration into `add()` / `memory_retain` + `list_entities` tool + MCP tool wrapper)

---

### 5. Entity Co-occurrence Graph

**What it does:** Given an entity (e.g., `docker_image:traefik`), find all other entities that co-occur with it across memory entries, ranked by co-occurrence count. This gives the agent a "what else is related to X?" query without a separate graph database. See the [full design discussion](https://github.com/codenamekt/hexus/issues) for the SQL under the hood.

**Dual-surface exposure:**

| Surface | Tool | Returns |
|---------|------|---------|
| MCP | `memory_entity_graph` | `{entity: {type, value}, related: [{type, value, co_occurrences, sample_content}]}` |
| Hermes plugin | `entity_graph` | Same structure, registered as memory plugin tool |

**Implementation:** Pure SQL aggregation over `metadata.entities` JSONB column with a GIN index. No new tables, no batch processing, no LLM. Co-occurrence is simply "entities that appear together in the same memory entry."

```sql
WITH source_entries AS (
    SELECT id, content, metadata
    FROM memory_entries
    WHERE metadata @> $1::jsonb
      AND ($2::text IS NULL OR agent_identity = $2)
),
related_entities AS (
    SELECT
        e->>'type' AS entity_type,
        e->>'value' AS entity_value,
        COUNT(*) AS co_occurrences,
        (ARRAY_AGG(content ORDER BY updated_at DESC))[1] AS sample_content
    FROM source_entries,
         jsonb_array_elements(metadata->'entities') AS e
    WHERE (e->>'type', e->>'value') != ($3, $4)
    GROUP BY e->>'type', e->>'value'
)
SELECT entity_type, entity_value, co_occurrences, sample_content
FROM related_entities
ORDER BY co_occurrences DESC
LIMIT $5
```

**Effort:** ~80 lines (one new `MemoryStore` method + one MCP tool + one Hermes tool schema)

---

### 6. Confidence / Recall Counter

**What it does:** Every time a memory entry is returned in a search result, its `metadata.recall_count` increments. The agent can also explicitly confirm ("this was relevant") or reject ("this was noise") an entry, stored as `metadata.confirm_count` / `metadata.reject_count`. This builds a lightweight trust signal over time — purely additive, no LLM.

**Dual-surface exposure:**

| Surface | Tool | Behavior |
|---------|------|----------|
| MCP | `memory_confirm(id, score=1.0)`, `memory_reject(id)` | Increments confirm/reject counters; adjusts effective score for future searches |
| Hermes plugin | `confirm_memory`, `reject_memory` | Same, registered as memory plugin tools |

**Implementation:** JSONB updates in the `search()` method (on every recall) and dedicated tool handlers. A new `min_confidence` filter on search tools uses the ratio `confirm_count / (confirm_count + reject_count)` when available.

**Config:**
```yaml
plugins:
  hexus:
    confidence:
      enabled: true
      min_confidence: 0.0    # 0 = disabled; 0.5 = only surface entries with >50% confirm rate
```

**Effort:** ~100 lines (metadata updates in `search()` + two tool handlers + `min_confidence` filter)

---

### 7. Conversation Summaries (extractive, no LLM)

**What it does:** Given a session ID, return the top-K most semantically central turns in that session. Computed by finding the centroid vector of all turns in the session, then selecting the K turns closest to the centroid. Pure vector math — fast, deterministic, no LLM call.

**Dual-surface exposure:**

| Surface | Tool | Returns |
|---------|------|---------|
| MCP | `memory_summarize_session` | `{session_id, turn_count, summary_turns: [{role, content, centrality_score}]}` |
| Hermes plugin | `summarize_session` | Same, registered as memory plugin tool |

**Implementation:** SQL with vector aggregation:

```sql
-- Compute session centroid
SELECT AVG(embedding) INTO session_centroid
FROM conversations
WHERE session_id = $1;

-- Find K turns closest to centroid
SELECT id, role, content, ts, 
       1 - (embedding <=> session_centroid) AS centrality_score
FROM conversations
WHERE session_id = $1
ORDER BY embedding <=> session_centroid
LIMIT $2
```

**Effort:** ~60 lines (one new `MemoryStore` method + one MCP tool + one Hermes tool schema)

---

## Phase 7: Nice to Have, Higher Effort (Future)

### 8. Cross-Encoder Reranker (optional, local)

**What it does:** Opt-in lightweight cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB, CPU-friendly) that reranks the top-N HNSW results for higher retrieval precision. Only loaded if the operator configures it. Shared across both surfaces (one model load per process).

**Dual-surface exposure:**

| Surface | Mechanism |
|---------|-----------|
| MCP | `memory_hybrid_search(rerank=true)` or `HEXUS_RERANK=true` env var |
| Hermes plugin | Config key `reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"`; if set, all search results are reranked |

**Implementation:** Load the cross-encoder once (shared singleton, like `LocalBertEmbedder`). On every search, rerank the top-50 HNSW results → return the top-K final results.

**Config:**
```yaml
plugins:
  hexus:
    reranker:
      model: ""                      # empty = disabled
      max_input_length: 512          # tokens per (query, document) pair
      top_n_before_rerank: 50        # how many HNSW results to rerank
```

**Effort:** ~120 lines (cross-encoder class + integration into `search()` + `hybrid_search()` + config)

---

### 9. Event Webhooks

**What it does:** POST a JSON payload to a configurable webhook URL when memory events occur (retain, forget, new session, session end). Operators plug in their own processing — log analysis, alerting, external indexing, Slack/Discord notifications. No LLM, no deriver loop.

**Dual-surface exposure:**

| Surface | Mechanism |
|---------|-----------|
| MCP | Server-side only; configured via env vars. Not a tool. |
| Hermes plugin | Plugin config; fires from the same hook points. |

**Config:**
```yaml
plugins:
  hexus:
    webhooks:
      url: ""                        # POST target
      secret: ""                     # HMAC-SHA256 signing secret
      events:
        - memory_retain
        - memory_forget
        - session_new
        - session_end
      retry_max: 3
      retry_backoff_seconds: 5
```

**Effort:** ~100 lines (webhook dispatch module + integration into `add()` / `memory_forget()` / `append_turn()`)

---

## Summary: Impact vs Effort

| # | Feature | Lines | Adds to Competitiveness | Phase |
|---|---------|-------|------------------------|-------|
| 1 | Hybrid Search (BM25 + vector) | ~80 | RetainDB-level recall precision | 5 |
| 2 | Temporal Decay Scoring | ~40 | Supermemory "forgetfulness" | 5 |
| 3 | TTL / memory_decay tool | ~60 | Supermemory "forgetfulness" | 5 |
| 4 | Entity Tagging (regex) | ~150 | Mnemosyne entity extraction | 6 |
| 5 | Entity Co-occurrence Graph | ~80 | Unique: knowledge graph lite, no LLM | 6 |
| 6 | Confidence / Recall Counter | ~100 | Holographic trust scoring | 6 |
| 7 | Conversation Summaries (extractive) | ~60 | Honcho session summaries (no LLM) | 6 |
| 8 | Cross-Encoder Reranker | ~120 | RetainDB reranker (optional, local) | 7 |
| 9 | Event Webhooks | ~100 | Extensibility: operator-owned processing | 7 |

**Total:** ~890 lines across 3 phases. Every feature available on both MCP and Hermes plugin surfaces.

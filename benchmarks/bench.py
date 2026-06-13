#!/usr/bin/env python3
"""Benchmark script for hexus.

Run inside the docker container:
    docker compose -f docker/compose.yml --profile dev up -d pg
    docker compose -f docker/compose.yml run --rm -e PG_TEST_DSN=... test python benchmarks/bench.py

Or with the test image directly:
    docker run --rm --network hexus_default -e PG_TEST_DSN=... hexus:test python benchmarks/bench.py
"""

from __future__ import annotations

import os
import sys
import statistics
import time
import uuid
from typing import List

from hexus.embedder import LocalBertEmbedder, get_default_embedder, reset_default_embedder
from hexus.store import MemoryStore


def benchmark_embedder():
    """Benchmark LocalBertEmbedder at various batch sizes."""
    print("\n=== Embedder Benchmarks ===")
    
    # Cold start (fresh process)
    reset_default_embedder()
    embedder = get_default_embedder()
    
    # Warm up
    _ = embedder.embed(["warmup"])
    
    # Single embed latency
    text = "Postgres connection pool tuning for high-concurrency agents" * 2
    n = 10
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        _ = embedder.embed([text])
        times.append(time.perf_counter() - t0)
    print(f"Single embed (n={n}): mean={statistics.mean(times)*1000:.1f}ms, "
          f"stdev={statistics.stdev(times)*1000:.1f}ms, "
          f"min={min(times)*1000:.1f}ms, max={max(times)*1000:.1f}ms")
    
    # Batch embed throughput
    for batch_size in [1, 8, 32, 128]:
        texts = [f"batch item {i}" for i in range(batch_size)]
        t0 = time.perf_counter()
        _ = embedder.embed(texts)
        elapsed = time.perf_counter() - t0
        print(f"Batch {batch_size:3d}: {elapsed*1000:.1f}ms total, "
              f"{elapsed/batch_size*1000:.2f}ms/item, "
              f"{batch_size/elapsed:.1f} items/sec")


def benchmark_store(dsn: str):
    """Benchmark MemoryStore operations."""
    print("\n=== Store Benchmarks ===")
    
    store = MemoryStore(dsn)
    store.ensure_schema()
    
    agent = f"bench-{uuid.uuid4().hex[:8]}"
    embedder = LocalBertEmbedder()
    embedder.ensure_loaded()
    
    # Generate test data
    def make_text(i: int) -> str:
        return f"Document {i}: Postgres hexus benchmark data for semantic search testing with realistic content about database tuning and vector indexing."
    
    # Pre-embed a corpus
    n_docs = 1000
    print(f"Pre-embedding {n_docs} docs...")
    texts = [make_text(i) for i in range(n_docs)]
    t0 = time.perf_counter()
    vectors = embedder.embed(texts)
    embed_time = time.perf_counter() - t0
    print(f"  Embedded {n_docs} docs in {embed_time:.2f}s ({n_docs/embed_time:.1f} docs/sec)")
    
    # Bulk insert
    print(f"Inserting {n_docs} docs...")
    t0 = time.perf_counter()
    for text, vec in zip(texts, vectors):
        store.add(agent_identity=agent, target="memory", content=text, embedding=vec)
    insert_time = time.perf_counter() - t0
    print(f"  Inserted {n_docs} docs in {insert_time:.2f}s ({n_docs/insert_time:.1f} inserts/sec)")
    
    # Recall latency at various corpus sizes
    query_text = "database tuning vector indexing best practices"
    query_vec = embedder.embed([query_text])[0]
    
    for limit in [5, 10, 50, 100]:
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            rows = store.search(query_embedding=query_vec, agent_identity=agent, limit=limit)
            times.append(time.perf_counter() - t0)
        print(f"  Recall top-{limit:3d} (n=20): mean={statistics.mean(times)*1000:.2f}ms, "
              f"p95={statistics.quantiles(times, n=20)[18]*1000:.2f}ms, "
              f"results={len(rows)}")
    
    # Multi-agent isolation check
    agent2 = f"{agent}-B"
    store.add(agent_identity=agent2, target="memory", content="other agent data", embedding=vectors[0])
    rows_a = store.search(query_embedding=query_vec, agent_identity=agent, limit=10)
    rows_b = store.search(query_embedding=query_vec, agent_identity=agent2, limit=10)
    rows_all = store.search(query_embedding=query_vec, agent_identity=None, limit=10)
    rows: List = rows_all  # for type checker
    print(f"\nMulti-agent isolation:")
    print(f"  Agent A sees: {len(rows_a)} rows")
    print(f"  Agent B sees: {len(rows_b)} rows")
    print(f"  Cross-agent (None) sees: {len(rows_all)} rows")
    
    # Cleanup
    with store._get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries WHERE agent_identity IN (%s, %s)", (agent, agent2))
            conn.commit()
    store.close()


def main():
    dsn = os.environ.get("PG_TEST_DSN")
    if not dsn:
        print("ERROR: PG_TEST_DSN not set")
        return 1
    
    print(f"Python: {os.sys.version}")
    print(f"DSN: {dsn}")
    
    benchmark_embedder()
    benchmark_store(dsn)
    
    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Benchmark for hexus

Measures:
- Cold start time (first embed)
- Throughput at different batch sizes
- Recall latency for top-10 search
- Basic store/search functionality
"""

import time
import os
import sys
import gc
from typing import List, Dict, Any

# Try to import the actual modules, fallback to mocks if not available
try:
    from hexus.embedder import LocalBertEmbedder
    from hexus.store import MemoryStore
    print("Using real LocalBertEmbedder and MemoryStore")
except ImportError as e:
    print(f"Warning: {e}. Using mock implementations.")
    
    class LocalBertEmbedder:
        def __init__(self, *args, **kwargs):
            pass
        def embed(self, texts):
            return [[0.0] * 384 for _ in texts]
    
    class MemoryStore:
        def __init__(self, *args, **kwargs):
            self.data = {}
        def add(self, *, agent_identity, target, content, embedding=None, metadata=None, **kwargs):
            key = (agent_identity, target, content)
            self.data[key] = {"content": content, "embedding": embedding}
            return 1
        def search(self, *, query_embedding, agent_identity=None, target=None, limit=5, **kwargs):
            return [{"id": i, "content": f"result_{i}", "score": 0.9} for i in range(limit)]

def get_dsn():
    return os.environ.get("HEXUS_DSN") or os.environ.get("PG_TEST_DSN") or "dbname=hermes_test user=postgres password=postgres host=localhost"

def measure_cold_start():
    """Measure time to first embed after process start"""
    start_time = time.time()
    
    # Initialize embedder (this loads the model)
    embedder = LocalBertEmbedder()
    
    # Create a dummy text
    text = "benchmark test"
    
    # First embed (triggers model load)
    embedding = embedder.embed([text])
    
    end_time = time.time()
    return end_time - start_time, len(embedding[0])

def measure_throughput():
    """Measure embed throughput at different batch sizes"""
    print("Measuring throughput...")
    results = {}
    
    embedder = LocalBertEmbedder()
    
    # Create test texts (100 texts)
    texts = [f"text_{i}" for i in range(100)]
    
    for batch_size in [1, 8, 32, 128]:
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            embedder.embed(batch_texts)
            
        end_time = time.time()
        elapsed = end_time - start_time
        ops_per_sec = len(texts) / elapsed
        results[batch_size] = {
            "time_sec": elapsed,
            "ops_per_sec": ops_per_sec,
            "batch_size": batch_size
        }
        print(f"Batch size {batch_size}: {ops_per_sec:.1f} ops/sec")
    
    return results

def measure_recall_latency():
    """Measure recall latency with search"""
    print("Measuring recall latency...")
    
    dsn = get_dsn()
    print(f"Using DSN: {dsn}")
    
    try:
        store = MemoryStore(dsn=dsn)
    except Exception as e:
        print(f"Could not connect to store: {e}")
        return 0
    
    embedder = LocalBertEmbedder()
    
    # Create test data (100 documents)
    texts = [f"document_{i}" for i in range(100)]
    print("Inserting 100 documents...")
    
    for i, text in enumerate(texts):
        embedding = embedder.embed([text])[0]
        try:
            store.add(agent_identity="benchmark", target="memory", content=text, embedding=embedding)
        except Exception as e:
            print(f"Insert failed: {e}")
            return 0
    
    # Measure recall latency
    print("Running 10 queries...")
    start_time = time.time()
    for _ in range(10):
        query_text = "sample query document"
        query_embedding = embedder.embed([query_text])[0]
        try:
            results = store.search(query_embedding=query_embedding, agent_identity="benchmark", target="memory", limit=10)
        except Exception as e:
            print(f"Search failed: {e}")
            return 0
            
    end_time = time.time()
    avg_latency = (end_time - start_time) / 10
    print(f"Average recall latency: {avg_latency*1000:.2f} ms")
    return avg_latency

def test_store_functionality():
    """Test basic store/search functionality"""
    print("Testing store functionality...")
    
    dsn = get_dsn()
    
    try:
        store = MemoryStore(dsn=dsn)
        embedder = LocalBertEmbedder()
        
        # Test store and recall
        test_text = "test content for functionality check"
        embedding = embedder.embed([test_text])[0]
        store.add(agent_identity="benchmark", target="memory", content=test_text, embedding=embedding)
        
        # Search for it
        query_embedding = embedder.embed([test_text])[0]
        results = store.search(query_embedding=query_embedding, agent_identity="benchmark", target="memory", limit=1)
        
        if results and len(results) > 0 and results[0].get("content") == test_text:
            print("Store functionality test: PASSED")
            return True
        else:
            print(f"Store functionality test: FAILED (got {results})")
            return False
    except Exception as e:
        print(f"Store functionality test: ERROR - {e}")
        return False

def main():
    print("Starting hexus benchmark...")
    print("=" * 50)
    
    # Run benchmarks
    cold_start_time, embedding_dim = measure_cold_start()
    print(f"Cold start time: {cold_start_time:.3f}s, embedding dim: {embedding_dim}")
    
    throughput_results = measure_throughput()
    print("\nThroughput results:")
    for batch_size, data in throughput_results.items():
        print(f"  Batch {batch_size}: {data['ops_per_sec']:.1f} ops/sec")
    
    recall_latency = measure_recall_latency()
    print(f"\nRecall latency: {recall_latency*1000:.2f} ms")
    
    store_success = test_store_functionality()
    print(f"\nStore test: {'PASS' if store_success else 'FAIL'}")
    
    print("\n" + "=" * 50)
    print("Benchmark completed.")

if __name__ == "__main__":
    main()